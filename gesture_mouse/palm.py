"""PALM-mode system-gesture detectors (pure logic, no macOS/cv2/numpy).

Detects the six trackpad-parity gestures from raw landmark frames:

- Open-palm swipe left/right/up/down (Space switch, Mission Control,
  App Expose) via an ARM-AT-REST -> NET-DISPLACEMENT state machine (below).
- Five-finger pinch-in (Launchpad) and spread-out (Show Desktop): threshold
  crossings of the spread metric ``m`` = mean distance of the five
  fingertips to the palm centroid (mean of landmarks 0, 5, 9, 13, 17),
  normalized by ``LandmarkFrame.scale``.

Swipe model — the consensus recipe from shipped MediaPipe gesture projects,
each point the OPPOSITE of the per-frame rule stack it replaced:

- The open-palm pose is tested ONLY AT REST ("arming"): pose + palm-center
  speed below ``arm_max_speed_fw_s``, held ``arm_hold_ms``, arms the
  detector. The swipe motion itself requires NO pose: fast lateral motion
  blurs fingertip landmarks (and finger-extension tests) precisely during
  the motion being detected — wrist/palm gross position survives blur.
- The tracked point is the PALM CENTER (mean of landmarks 0,5,9,13,17),
  never a fingertip.
- Hand-absent frames NEVER clear the trajectory or disarm unless the gap
  exceeds ``swipe_gap_bridge_ms``: MediaPipe dropping the hand for a few
  frames mid-swipe is documented-normal under fast motion (the hand outruns
  the VIDEO-mode tracking ROI; blur breaks the landmark model). One good
  observation on each side of the motion is enough.
- Firing is NET displacement from the arming origin — >=
  ``swipe_min_disp_frac`` of the frame span on a dominant axis within
  ``swipe_max_duration_ms`` — never instantaneous velocity, which is
  meaningless across dropped frames and variable fps.
- After a fire: ``swipe_cooldown_ms``, then a FULL re-arm (pose at rest
  again) for the next swipe.

State machine::

    IDLE --(open palm AND at rest, held arm_hold_ms)--> ARMED
    ARMED: origin refreshed every frame while still at rest (drift can't
           accumulate into a fire); window countdown starts when motion does
    ARMED --(net disp >= threshold, dominant axis)--> fire -> COOLDOWN
    ARMED --(window expires / gap > bridge)--> IDLE
    COOLDOWN --(cooldown elapses)--> IDLE

``update`` is called EVERY frame (not only in PALM mode) with the engine's
(jitter-debounced) open-palm flag. The spread deque for pinch-in/spread-out
IS still cleared on any hand-absent frame — those gestures happen at rest,
where an absent frame means something real.

Units: times are ms of ``ts_ms``; displacement fractions are of ``img_w``
horizontally and ``img_h`` vertically; speed is frame-widths/second.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from .config import PalmConfig
from .types import (
    FINGER_TIPS,
    Intent,
    LandmarkFrame,
    Phase,
    Point,
    dist,
)

# Palm centroid = mean of wrist + the four finger MCPs. This is BOTH the
# spread-metric reference and the swipe-trajectory point.
_PALM_CENTROID_IDS = (0, 5, 9, 13, 17)

# The at-rest speed estimate compares the newest sample against one at least
# this much older — a shorter baseline is dominated by landmark jitter.
_SPEED_BASELINE_MS = 80.0
# A single noisy speed/pose frame must not restart the arming hold.
_ARM_GRACE_MS = 80.0


class SwipePhase(Enum):
    IDLE = "idle"
    ARMING = "arming"
    ARMED = "armed"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class _TrajSample:
    ts_ms: float
    x: float          # palm-center pixel coords
    y: float


class PalmDetector:
    """Self-contained detector for PALM-mode system gestures.

    ``bindings`` maps gesture keys (``swipe_left/right/up/down``,
    ``pinch_in``, ``spread_out``) to Intent names (see config.DEFAULT_BINDINGS).
    A recognized gesture with no binding is consumed (cooldown still starts)
    but emits nothing.

    ``debug=True`` appends human-readable event strings to ``self.events``
    (a bounded deque the caller drains and prints — this module stays pure).
    """

    def __init__(
        self, cfg: PalmConfig, bindings: dict[str, str], debug: bool = False
    ) -> None:
        self.cfg = cfg
        self.bindings = bindings
        self.active = False  # last frame's open-palm flag (preview readout)
        self.last_m: float | None = None  # latest spread metric, for tuning UI
        self.phase = SwipePhase.IDLE
        self.debug_enabled = debug
        self.events: deque[tuple[float, str]] = deque(maxlen=256)
        self.last_disp_frac: float = 0.0  # |major-axis| net travel, for the UI
        # Swipe trajectory: palm-center samples from hand-PRESENT frames only.
        self._traj: deque[_TrajSample] = deque()
        self._last_seen_ts: float | None = None
        self._arm_since: float | None = None
        self._arm_break_since: float | None = None
        self._origin: _TrajSample | None = None
        self._cooldown_until: float = float("-inf")
        # Five-finger pinch/spread (separate history; clears on hand loss).
        self._spread: deque[tuple[float, float]] = deque()
        self._spread_block_until: float = float("-inf")
        self._spread_reset_ts: float = float("-inf")

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        self.active = False
        self.last_m = None
        self.phase = SwipePhase.IDLE
        self.last_disp_frac = 0.0
        self._traj.clear()
        self._last_seen_ts = None
        self._arm_since = None
        self._arm_break_since = None
        self._origin = None
        self._cooldown_until = float("-inf")
        self._spread.clear()
        self._spread_block_until = float("-inf")
        self._spread_reset_ts = float("-inf")

    def disarm(self) -> None:
        """Engine calls this on pinch/scroll entry: a stale armed state must
        not fire a swipe the instant an unrelated gesture releases."""
        if self.phase in (SwipePhase.ARMING, SwipePhase.ARMED):
            self._event(self._last_seen_ts or 0.0, "disarmed (pinch/scroll)")
            self.phase = SwipePhase.IDLE
        self._arm_since = None
        self._arm_break_since = None
        self._origin = None

    @property
    def armed(self) -> bool:
        return self.phase is SwipePhase.ARMED

    @property
    def engaged(self) -> bool:
        """True while the swipe system is mid-episode (arming through
        cooldown) — drives the engine's cursor freeze."""
        return self.phase is not SwipePhase.IDLE

    @property
    def debug(self) -> dict[str, float | bool | str]:
        out: dict[str, float | bool | str] = {
            "phase": self.phase.value,
            "armed": self.armed,
            "disp_frac": self.last_disp_frac,
        }
        if self.last_m is not None:
            out["m"] = self.last_m
        return out

    def _event(self, ts: float, msg: str) -> None:
        if self.debug_enabled:
            self.events.append((ts, msg))

    # -- per-frame update ----------------------------------------------------

    def update(self, frame: LandmarkFrame, open_palm: bool) -> list[Intent]:
        now = frame.ts_ms

        if not frame.hand_present or frame.scale <= 0.0:
            # Spread gestures happen at rest: an absent frame there is real,
            # so that history clears. The swipe trajectory does NOT clear —
            # losing the hand mid-swipe is normal; only a gap longer than
            # the bridge abandons the episode.
            self._spread.clear()
            self.active = False
            self.last_m = None
            if (
                self._last_seen_ts is not None
                and now - self._last_seen_ts > self.cfg.swipe_gap_bridge_ms
                and self.phase in (SwipePhase.ARMING, SwipePhase.ARMED)
            ):
                self._event(
                    now,
                    f"gap {now - self._last_seen_ts:.0f}ms > "
                    f"{self.cfg.swipe_gap_bridge_ms:.0f} -> disarmed",
                )
                self.phase = SwipePhase.IDLE
                self._traj.clear()
                self._arm_since = None
                self._arm_break_since = None
                self._origin = None
            return []

        landmarks = frame.landmarks
        assert landmarks is not None  # guaranteed by hand_present
        self.active = open_palm
        self._last_seen_ts = now

        center = self._palm_center(landmarks)
        self._traj.append(_TrajSample(now, center.x, center.y))
        horizon = self.cfg.swipe_max_duration_ms + self.cfg.swipe_gap_bridge_ms
        while self._traj and now - self._traj[0].ts_ms > horizon:
            self._traj.popleft()

        m = self._spread_metric(landmarks, frame.scale)
        self.last_m = m
        self._spread.append((now, m))
        while self._spread and now - self._spread[0][0] > self.cfg.spread_window_ms:
            self._spread.popleft()

        intents: list[Intent] = []
        swipe = self._tick_swipe(frame, now, center, open_palm)
        if swipe is not None:
            intents.append(swipe)
        intents.extend(self._detect_spread_gestures(now, m))
        return intents

    # -- swipe state machine --------------------------------------------------

    def _tick_swipe(
        self, frame: LandmarkFrame, now: float, center: Point, open_palm: bool
    ) -> Intent | None:
        cfg = self.cfg

        if self.phase is SwipePhase.COOLDOWN:
            if now >= self._cooldown_until:
                self.phase = SwipePhase.IDLE
                self._event(now, "cooldown over -> idle")
            else:
                return None

        speed = self._palm_speed_fw_s(now, frame.img_w)
        at_rest = speed is not None and speed < cfg.arm_max_speed_fw_s
        arm_condition = open_palm and at_rest

        if self.phase in (SwipePhase.IDLE, SwipePhase.ARMING):
            if arm_condition:
                self._arm_break_since = None
                if self._arm_since is None:
                    self._arm_since = now
                    self.phase = SwipePhase.ARMING
                    self._event(now, "arming (open palm at rest)")
                if now - self._arm_since >= cfg.arm_hold_ms:
                    self.phase = SwipePhase.ARMED
                    self._origin = _TrajSample(now, center.x, center.y)
                    self._event(
                        now, f"ARMED origin=({center.x:.0f},{center.y:.0f})"
                    )
            elif self._arm_since is not None:
                # Tolerate a brief noisy frame before abandoning the hold.
                if self._arm_break_since is None:
                    self._arm_break_since = now
                elif now - self._arm_break_since > _ARM_GRACE_MS:
                    reason = "pose lost" if not open_palm else (
                        f"speed {speed:.2f} >= {cfg.arm_max_speed_fw_s:.2f}"
                        if speed is not None else "no speed baseline"
                    )
                    self._event(now, f"arming aborted ({reason})")
                    self._arm_since = None
                    self._arm_break_since = None
                    self.phase = SwipePhase.IDLE
            return None

        # ARMED.
        assert self._origin is not None
        if arm_condition:
            # Still resting: refresh the origin so slow drift never
            # accumulates into a phantom swipe, and the fire window only
            # starts counting when real motion does.
            self._origin = _TrajSample(now, center.x, center.y)
            self.last_disp_frac = 0.0
            return None

        if now - self._origin.ts_ms > cfg.swipe_max_duration_ms:
            self._event(now, "window expired -> idle (re-arm to retry)")
            self.phase = SwipePhase.IDLE
            self._arm_since = None
            self._arm_break_since = None
            self._origin = None
            self.last_disp_frac = 0.0
            return None

        fx = (center.x - self._origin.x) / float(frame.img_w)
        fy = (center.y - self._origin.y) / float(frame.img_h)
        afx, afy = abs(fx), abs(fy)
        self.last_disp_frac = max(afx, afy)
        if self.debug_enabled and self.last_disp_frac > 0.05:
            self._event(
                now,
                f"candidate dx={fx:+.2f} dy={fy:+.2f} "
                f"need {cfg.swipe_min_disp_frac:.2f}",
            )

        if self.last_disp_frac < cfg.swipe_min_disp_frac:
            return None
        if afx >= afy * cfg.swipe_axis_dominance:
            key = "swipe_left" if fx < 0 else "swipe_right"
        elif afy >= afx * cfg.swipe_axis_dominance:
            key = "swipe_up" if fy < 0 else "swipe_down"
        else:
            # Far enough but diagonal: keep waiting inside the window — the
            # dominant axis usually wins as the motion completes.
            return None

        self.phase = SwipePhase.COOLDOWN
        self._cooldown_until = now + cfg.swipe_cooldown_ms
        self._arm_since = None
        self._arm_break_since = None
        self._origin = None
        self.last_disp_frac = 0.0
        # The hand relaxing/returning after a swipe sweeps the spread metric
        # through the pinch-in/spread-out trajectories; wall that history off.
        self._spread_reset_ts = now
        name = self.bindings.get(key)
        self._event(now, f"FIRE {key} -> {name or '(unbound)'}")
        if name is None:
            return None
        return Intent(name=name, phase=Phase.TRIGGER, ts_ms=now)

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _palm_center(landmarks: tuple[Point, ...]) -> Point:
        n = len(_PALM_CENTROID_IDS)
        return Point(
            sum(landmarks[i].x for i in _PALM_CENTROID_IDS) / n,
            sum(landmarks[i].y for i in _PALM_CENTROID_IDS) / n,
        )

    def _palm_speed_fw_s(self, now: float, img_w: int) -> float | None:
        """Palm-center speed in frame-widths/s over a >=80ms baseline; None
        when history is too short to judge (treated as NOT at rest)."""
        newest = self._traj[-1]
        baseline: _TrajSample | None = None
        for s in reversed(self._traj):
            if now - s.ts_ms >= _SPEED_BASELINE_MS:
                baseline = s
                break
        if baseline is None:
            return None
        dt_s = (newest.ts_ms - baseline.ts_ms) / 1000.0
        if dt_s <= 0.0:
            return None
        d = ((newest.x - baseline.x) ** 2 + (newest.y - baseline.y) ** 2) ** 0.5
        return d / float(img_w) / dt_s

    @staticmethod
    def _spread_metric(landmarks: tuple[Point, ...], scale: float) -> float:
        centroid = PalmDetector._palm_center(landmarks)
        total = sum(dist(landmarks[i], centroid) for i in FINGER_TIPS)
        return total / len(FINGER_TIPS) / scale

    # -- five-finger pinch/spread (unchanged semantics) -------------------------

    def _detect_spread_gestures(self, now: float, m: float) -> list[Intent]:
        cfg = self.cfg
        if now < self._spread_block_until:
            return []
        # Only history from after the last firing counts: the reset marker
        # keeps the pre-gesture extreme (open palm before a pinch-in, curled
        # fist before a spread-out) from qualifying the recovery motion as the
        # opposite gesture once the refractory lapses.
        window = [
            (ts, sm) for (ts, sm) in self._spread
            if now - ts <= cfg.spread_window_ms and ts > self._spread_reset_ts
        ]

        def fire(key: str) -> list[Intent]:
            self._spread_block_until = now + cfg.spread_refractory_ms
            self._spread_reset_ts = now
            name = self.bindings.get(key)
            self._event(now, f"FIRE {key} -> {name or '(unbound)'}")
            if name is None:
                return []
            return [Intent(name=name, phase=Phase.TRIGGER, ts_ms=now)]

        # Pinch-in: m was above spread_open within the window, now below
        # spread_closed. Refractory (800ms) outlasts the window (500ms), so
        # once it expires the qualifying start sample has aged out and a held
        # fist cannot re-fire.
        if m < cfg.spread_closed and any(sm > cfg.spread_open for _, sm in window):
            return fire("pinch_in")

        # Spread-out: m was below spread_in_start within the window (curled
        # hand), now above spread_out.
        if m > cfg.spread_out and any(sm < cfg.spread_in_start for _, sm in window):
            return fire("spread_out")

        return []
