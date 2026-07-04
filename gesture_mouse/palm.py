"""PALM-mode system-gesture detectors (pure logic, no macOS/cv2/numpy).

Detects the six trackpad-parity gestures from raw landmark frames:

- Open-palm swipe left/right/up/down (Space switch, Mission Control,
  App Expose): net anchor displacement + mean speed + dominant-axis test
  over a sliding window, gated on the engine's open-palm pose flag.
- Five-finger pinch-in (Launchpad) and spread-out (Show Desktop): threshold
  crossings of the spread metric ``m`` = mean distance of the five
  fingertips to the palm centroid (mean of landmarks 0, 5, 9, 13, 17),
  normalized by ``LandmarkFrame.scale``.

``update`` is called EVERY frame (not only in PALM mode) with the engine's
open-palm flag; the detector keeps its own sliding windows over
``(ts_ms, anchor_x, anchor_y, m)`` so that:

- Swipes only measure the trailing contiguous run of open-palm samples —
  motion made before the pose formed (or after it broke) never counts.
- Pinch-in / spread-out ignore the pose flag entirely: pinch-in starts open
  but the pose drops as the fingers curl, and spread-out starts from a
  curled hand where the pose was never present.

Units: all times are milliseconds of ``ts_ms``; swipe displacement is a
fraction of the dominant-axis frame dimension (``img_w`` horizontal,
``img_h`` vertical) and speed is that same fraction per second — i.e.
frame-widths/s horizontally, frame-heights/s vertically.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import PalmConfig
from .types import (
    FINGER_TIPS,
    INDEX_MCP,
    Intent,
    LandmarkFrame,
    Phase,
    Point,
    dist,
)

# Palm centroid = mean of wrist + the four finger MCPs (plan doc §Launchpad).
_PALM_CENTROID_IDS = (0, 5, 9, 13, 17)

# A swipe reference sample closer than this to "now" is too short a baseline
# to distinguish a deliberate flick from one frame of jitter.
_MIN_SWIPE_DT_MS = 60.0


@dataclass(frozen=True)
class _Sample:
    ts_ms: float
    x: float          # anchor (INDEX_MCP) pixel coords
    y: float
    m: float          # spread metric, dimensionless
    open_palm: bool   # engine's all-5-extended pose flag for this frame


class PalmDetector:
    """Self-contained sliding-window detector for PALM-mode system gestures.

    ``bindings`` maps gesture keys (``swipe_left/right/up/down``,
    ``pinch_in``, ``spread_out``) to Intent names (see config.DEFAULT_BINDINGS).
    A recognized gesture with no binding is consumed (refractory still starts)
    but emits nothing.
    """

    def __init__(self, cfg: PalmConfig, bindings: dict[str, str]) -> None:
        self.cfg = cfg
        self.bindings = bindings
        self.active = False  # True while the last frame had the open-palm pose
        self._samples: deque[_Sample] = deque()
        self._swipe_block_until: float = float("-inf")
        self._need_pose_drop = False  # open_palm must go False before next swipe
        # Pinch-in and spread-out share ONE refractory and one history-reset
        # marker: the natural hand recovery after either gesture (reopening
        # after a pinch-in, re-curling after a spread-out) is exactly the
        # opposite gesture's trajectory and must never fire it.
        self._spread_block_until: float = float("-inf")
        self._spread_reset_ts: float = float("-inf")

    def reset(self) -> None:
        self.active = False
        self._samples.clear()
        self._swipe_block_until = float("-inf")
        self._need_pose_drop = False
        self._spread_block_until = float("-inf")
        self._spread_reset_ts = float("-inf")

    def update(self, frame: LandmarkFrame, open_palm: bool) -> list[Intent]:
        if not frame.hand_present or frame.scale <= 0.0:
            # No usable hand: windows restart from scratch, and losing the
            # hand counts as dropping the pose for the swipe re-arm rule.
            self._samples.clear()
            self._need_pose_drop = False
            self.active = False
            return []

        landmarks = frame.landmarks
        assert landmarks is not None  # guaranteed by hand_present
        now = frame.ts_ms
        anchor = landmarks[INDEX_MCP]
        m = self._spread_metric(landmarks, frame.scale)
        self._samples.append(_Sample(now, anchor.x, anchor.y, m, open_palm))
        self._prune(now)

        if not open_palm:
            self._need_pose_drop = False
        self.active = open_palm

        intents: list[Intent] = []
        swipe = self._detect_swipe(frame, now, open_palm)
        if swipe is not None:
            intents.append(swipe)
        intents.extend(self._detect_spread_gestures(now, m))
        return intents

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _spread_metric(landmarks: tuple[Point, ...], scale: float) -> float:
        n = len(_PALM_CENTROID_IDS)
        cx = sum(landmarks[i].x for i in _PALM_CENTROID_IDS) / n
        cy = sum(landmarks[i].y for i in _PALM_CENTROID_IDS) / n
        centroid = Point(cx, cy)
        total = sum(dist(landmarks[i], centroid) for i in FINGER_TIPS)
        return total / len(FINGER_TIPS) / scale

    def _prune(self, now: float) -> None:
        horizon = max(self.cfg.swipe_window_ms, self.cfg.spread_window_ms)
        while self._samples and now - self._samples[0].ts_ms > horizon:
            self._samples.popleft()

    def _detect_swipe(
        self, frame: LandmarkFrame, now: float, open_palm: bool
    ) -> Intent | None:
        cfg = self.cfg
        if not open_palm or self._need_pose_drop or now < self._swipe_block_until:
            return None

        cur = self._samples[-1]
        # Candidate reference samples: the trailing contiguous open-palm run
        # inside the swipe window (a pose break resets the run so pre-pose
        # motion never contributes). Each candidate is tested independently —
        # anchoring only on the OLDEST sample would average a fast flick with
        # any preceding still-palm dwell and never fire (still 250 ms + flick
        # 100 ms = high displacement but low window-average speed).
        for s in reversed(self._samples):
            if now - s.ts_ms > cfg.swipe_window_ms or not s.open_palm:
                break
            if s is cur:
                continue
            dt_s = (now - s.ts_ms) / 1000.0
            if dt_s * 1000.0 < _MIN_SWIPE_DT_MS:
                continue  # too short a baseline to call it deliberate

            dx = cur.x - s.x
            dy = cur.y - s.y
            adx, ady = abs(dx), abs(dy)
            # Dominant axis must carry at least 2x the minor axis.
            if adx >= 2.0 * ady and adx > 0.0:
                span, disp = float(frame.img_w), adx
                key = "swipe_left" if dx < 0 else "swipe_right"
            elif ady >= 2.0 * adx and ady > 0.0:
                span, disp = float(frame.img_h), ady
                key = "swipe_up" if dy < 0 else "swipe_down"
            else:
                continue

            if disp <= cfg.swipe_min_disp_frac * span:
                continue
            if disp / span / dt_s <= cfg.swipe_min_vel_fw_s:
                continue

            self._swipe_block_until = now + cfg.swipe_refractory_ms
            self._need_pose_drop = True
            name = self.bindings.get(key)
            if name is None:
                return None
            return Intent(name=name, phase=Phase.TRIGGER, ts_ms=now)
        return None

    def _detect_spread_gestures(self, now: float, m: float) -> list[Intent]:
        cfg = self.cfg
        if now < self._spread_block_until:
            return []
        # Only history from after the last firing counts: the reset marker
        # keeps the pre-gesture extreme (open palm before a pinch-in, curled
        # fist before a spread-out) from qualifying the recovery motion as the
        # opposite gesture once the refractory lapses.
        window = [
            s for s in self._samples
            if now - s.ts_ms <= cfg.spread_window_ms
            and s.ts_ms > self._spread_reset_ts
        ]

        def fire(key: str) -> list[Intent]:
            self._spread_block_until = now + cfg.spread_refractory_ms
            self._spread_reset_ts = now
            name = self.bindings.get(key)
            if name is None:
                return []
            return [Intent(name=name, phase=Phase.TRIGGER, ts_ms=now)]

        # Pinch-in: m was above spread_open within the window, now below
        # spread_closed. Refractory (800ms) outlasts the window (500ms), so
        # once it expires the qualifying start sample has aged out and a held
        # fist cannot re-fire.
        if m < cfg.spread_closed and any(s.m > cfg.spread_open for s in window):
            return fire("pinch_in")

        # Spread-out: m was below spread_in_start within the window (curled
        # hand), now above spread_out.
        if m > cfg.spread_out and any(s.m < cfg.spread_in_start for s in window):
            return fire("spread_out")

        return []
