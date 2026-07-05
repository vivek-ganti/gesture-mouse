"""Pure gesture engine: (LandmarkFrame, CursorSample) -> EngineOutput.

No macOS / cv2 / mediapipe / numpy imports and no wall-clock reads — all
temporal logic is in milliseconds of ``frame.ts_ms``, never frame counts.

State machine (see plan doc §State machine, CONTRACTS.md §engine.py):
CLUTCH_WAIT -> pointer pose held clutch.engage_ms -> POINTER (rebase, no
cursor jump). Engagement-gating only: a relaxed hand keeps POINTER. Left
pinch (dist(4,8)/scale, One-Euro filtered via the injected ``pinch_filter``)
with conditional position latch, release latch, hysteresis, debounce and a
scale-stability gate; drag with distance-only 12px unfreeze, in-drag release
threshold, minor-axis dead-band; double-click with relaxed thresholds inside
the 500ms/15px window (click_count=2, single clicks never delayed); right
click = thumb-middle pinch requiring a clearly-extended index, with argmin
arbitration against the left pinch; two-finger joystick scroll; HANDS_LOST
with auto-mouseUp after cfg.hands_lost_ms and 250ms clutch reacquire; PALM
(open-hand system gestures) delegated to an injected, duck-typed detector.

Freeze semantics: ``EngineOutput.freeze`` asks the cursor pipeline to hold
its *output* (used in CLUTCH_WAIT / SCROLL / PALM / HANDS_LOST, where the
held position is also the rebase target on exit). While a pinch is held but
not yet dragging the engine instead keeps the pipeline live and simply emits
no move intents — the on-screen cursor is parked at the mouse-down point,
while the live samples let the engine measure the drag-unfreeze distance in
true screen pixels; on unfreeze it rebases the pipeline back to the frozen
point, so the pinch approach travel is swallowed exactly as if the pipeline
had been frozen.

``notify_suspended()`` clears held-button state WITHOUT emitting UP intents
(``synth.release_all()`` posts the actual UPs) and resets to a CLUTCH_WAIT
that requires ``clutch.reacquire_ms`` of pointer pose.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

from .config import Config
from .types import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    CursorSample,
    EngineState,
    Intent,
    LandmarkFrame,
    Phase,
    Point,
    dist,
)

# Radial finger-extension test factor (plan doc §Gesture vocabulary).
_EXTENDED_RATIO = 1.15
# Sustained low tracker confidence degrades like a missing hand.
_LOW_CONF = 0.4
_LOW_CONF_MS = 165.0
# Anchor teleporting >25% of the frame width in one frame = hand swap.
_ANCHOR_JUMP_FRAC = 0.25
# Internal history horizons (ms).
_HISTORY_MS = 600.0
_SCALE_WINDOW_MS = 120.0

_ACTIVE_STATES = frozenset(
    {
        EngineState.POINTER,
        EngineState.PINCHED,
        EngineState.RIGHT_PINCH,
        EngineState.SCROLL,
        EngineState.PALM,
    }
)


@dataclass
class EngineOutput:
    intents: list[Intent]
    freeze: bool          # cursor pipeline frozen next frame
    drag: bool            # drag filter mode next frame
    rebase: tuple[float, float] | None   # rebase request (screen coords)


@dataclass
class _Arbitration:
    """Left-vs-right pinch argmin hold (plan doc: right-click arbitration)."""

    until: float
    left_min: float
    right_min: float


class _Debounced:
    """Tracks how long a per-frame boolean condition has been continuously
    true, tolerating up to ``grace_ms`` of false readings without resetting
    the hold — single/double-frame hand-tracking jitter (a finger's radial
    extension test flickering false for one frame) must not restart a
    multi-hundred-ms pose hold from zero. Requiring several landmarks to
    agree at once (all four fingers extended for the open-palm pose) makes
    this far more likely per-frame than the single-landmark pointer pose, so
    every pose hold in the engine goes through this, not just palm's.

    ``since`` is the timestamp of the FIRST true sample in the current
    unbroken (within grace) run — ``ts - since`` is the effective hold
    duration; ``None`` means "not currently holding"."""

    __slots__ = ("grace_ms", "since", "_false_since")

    def __init__(self, grace_ms: float) -> None:
        self.grace_ms = grace_ms
        self.since: float | None = None
        self._false_since: float | None = None

    def update(self, ts: float, value: bool) -> float | None:
        if value:
            self._false_since = None
            if self.since is None:
                self.since = ts
        elif self.since is not None:
            if self._false_since is None:
                self._false_since = ts
            elif ts - self._false_since > self.grace_ms:
                self.since = None
                self._false_since = None
        return self.since

    def reset(self) -> None:
        self.since = None
        self._false_since = None


class GestureEngine:
    def __init__(
        self,
        cfg: Config,
        pinch_filter: Callable[[str, float, float], float],
        palm_detector=None,
    ) -> None:
        """``pinch_filter``: (name, raw, ts_ms) -> filtered — pass
        ``CursorPipeline.pinch``. ``palm_detector`` is duck-typed
        (.update(frame, open_palm) -> list[Intent], .reset(), .active) and
        may be None, in which case PALM transitions are skipped entirely.
        """
        self.cfg = cfg
        self._pinch_filter = pinch_filter
        self._palm = palm_detector
        self.state = EngineState.CLUTCH_WAIT
        self._reset_transients(clutch_required=cfg.clutch.engage_ms)

    # -- lifecycle ----------------------------------------------------------

    def _reset_transients(self, clutch_required: float) -> None:
        grace = self.cfg.pose_jitter_grace_ms
        self._clutch_required = clutch_required
        self._clutch_hold = _Debounced(grace)
        self._cursor_hist: deque[tuple[float, float, float]] = deque()
        self._scale_hist: deque[tuple[float, float]] = deque()
        self._last_ts: float | None = None
        self._prev_anchor: Point | None = None
        self._low_conf_since: float | None = None
        # Pinch candidacy (POINTER).
        self._left_val = float("inf")
        self._right_val = float("inf")
        self._left_since: float | None = None
        self._left_latch: tuple[float, float] | None = None
        self._left_cc = 1
        self._right_since: float | None = None
        self._arb: _Arbitration | None = None
        # Held button / drag (PINCHED, RIGHT_PINCH).
        self._held: tuple[str, int, tuple[float, float]] | None = None
        self._right_down_ts: float = 0.0
        self._down_pos: tuple[float, float] = (0.0, 0.0)
        self._freeze_ref: tuple[float, float] = (0.0, 0.0)
        self._dragging = False
        self._locked_axis: str | None = None
        self._release_since: float | None = None
        self._release_latch: tuple[float, float] | None = None
        # Double click window.
        self._last_up: tuple[float, float, float] | None = None  # ts, x, y
        # Scroll (vertical joystick) + tab switch (horizontal joystick).
        self._scroll_hold = _Debounced(grace)
        self._scroll_y0 = 0.0
        self._scroll_x0 = 0.0
        self._scroll_exit_since: float | None = None
        self._tab_block_until = float("-inf")
        # Palm: ONE persistent debounced hold of the four-finger open-palm
        # pose, updated every frame regardless of engine state — its `.since`
        # value directly gates PALM entry (held >= palm.enter_ms), PALM exit
        # (exits once the debounced signal itself goes false, which already
        # absorbed the jitter grace), and the boolean forwarded into the palm
        # detector (so swipe continuity gets the same jitter tolerance).
        self._open_palm_hold = _Debounced(grace)
        self._last_open_palm: bool = False   # for the palm_debug readout
        # Hands lost.
        self._lost_since: float | None = None
        self._reacquire_hold = _Debounced(grace)

    def reset(self) -> None:
        self.state = EngineState.CLUTCH_WAIT
        self._reset_transients(clutch_required=self.cfg.clutch.engage_ms)
        if self._palm is not None:
            self._palm.reset()

    def notify_suspended(self) -> None:
        """Suspend: forget held buttons WITHOUT emitting (synth.release_all
        posts the UPs) and require a full clutch reacquire to resume."""
        self.state = EngineState.CLUTCH_WAIT
        self._reset_transients(clutch_required=self.cfg.clutch.reacquire_ms)
        if self._palm is not None:
            self._palm.reset()

    @property
    def pinch_values(self) -> dict[str, float]:
        """Latest filtered pinch distances for the preview meters (additive to
        the CONTRACTS.md surface). Keys absent until a valid hand was seen."""
        inf = float("inf")
        out: dict[str, float] = {}
        if self._left_val != inf:
            out["left"] = self._left_val
        if self._right_val != inf:
            out["right"] = self._right_val
        return out

    @property
    def palm_debug(self) -> dict[str, float | bool]:
        """Live palm-gesture tuning readout (additive to the CONTRACTS.md
        surface): whether the four-finger pose is currently held, and the
        continuous five-finger spread metric (Launchpad/Show Desktop) against
        which ``palm.spread_closed`` / ``palm.spread_open`` are compared."""
        out: dict[str, float | bool] = {"open": self._last_open_palm}
        if self._palm is not None and self._palm.last_m is not None:
            out["m"] = self._palm.last_m
        return out

    # -- pose tests ---------------------------------------------------------

    def _ext(self, lm: tuple[Point, ...], tip: int, pip: int, wrist: Point) -> bool:
        if self.cfg.options.extended_test == "y":
            # Mirrored frame is y-down: an extended (upward-pointing) finger
            # has its tip above its PIP. Config alternative for camera-down
            # geometry where the radial test's wrist distances compress.
            return lm[tip].y < lm[pip].y
        return dist(lm[tip], wrist) > _EXTENDED_RATIO * dist(lm[pip], wrist)

    def _index_extended(self, lm: tuple[Point, ...]) -> bool:
        return self._ext(lm, INDEX_TIP, INDEX_PIP, lm[WRIST])

    def _pointer_pose(self, lm: tuple[Point, ...]) -> bool:
        w = lm[WRIST]
        return (
            self._ext(lm, INDEX_TIP, INDEX_PIP, w)
            and not self._ext(lm, MIDDLE_TIP, MIDDLE_PIP, w)
            and not self._ext(lm, RING_TIP, RING_PIP, w)
            and not self._ext(lm, PINKY_TIP, PINKY_PIP, w)
        )

    def _open_palm_pose(self, lm: tuple[Point, ...]) -> bool:
        """Four fingers extended — deliberately EXCLUDES the thumb.

        The trackpad gesture this mirrors ("4-finger swipe") never needed the
        thumb either, and the thumb's radial-from-wrist extension test is the
        least reliable of the five: the thumb's range of motion is mostly
        lateral (across the palm), not radial from the wrist, so on a real
        hand it very often reads as "curled" even when visibly splayed out —
        silently blocking every four/five-finger gesture behind a pose that
        can almost never be satisfied. Five-finger pinch-in/spread-out
        (Launchpad/Show Desktop) don't use this pose test at all — they
        already read the thumb through the continuous spread metric in
        palm.py, which has no such failure mode.
        """
        w = lm[WRIST]
        return (
            self._ext(lm, INDEX_TIP, INDEX_PIP, w)
            and self._ext(lm, MIDDLE_TIP, MIDDLE_PIP, w)
            and self._ext(lm, RING_TIP, RING_PIP, w)
            and self._ext(lm, PINKY_TIP, PINKY_PIP, w)
        )

    def _scroll_pose(self, lm: tuple[Point, ...], scale: float) -> bool:
        w = lm[WRIST]
        if not (
            self._ext(lm, INDEX_TIP, INDEX_PIP, w)
            and self._ext(lm, MIDDLE_TIP, MIDDLE_PIP, w)
            and not self._ext(lm, RING_TIP, RING_PIP, w)
            and not self._ext(lm, PINKY_TIP, PINKY_PIP, w)
        ):
            return False
        return dist(lm[INDEX_TIP], lm[MIDDLE_TIP]) / scale < self.cfg.scroll.together_max

    # -- small helpers ------------------------------------------------------

    def _hand_valid(self, frame: LandmarkFrame, ts: float) -> bool:
        """Present, matching handedness, and not in sustained low confidence.

        cfg.hand "auto" accepts either hand — handedness labels depend on the
        camera's mirror convention (virtual/phone cameras often break it), and
        the anchor-jump continuity check already guards against a second hand
        stealing the cursor mid-track."""
        if not frame.hand_present or frame.scale <= 0.0:
            self._low_conf_since = None
            return False
        want = self.cfg.hand.lower()
        if frame.handedness is None or (
            want != "auto" and frame.handedness.lower() != want
        ):
            return False
        if frame.confidence < _LOW_CONF:
            if self._low_conf_since is None:
                self._low_conf_since = ts
            if ts - self._low_conf_since >= _LOW_CONF_MS:
                return False
        else:
            self._low_conf_since = None
        return True

    def _scale_stable(self, ts: float) -> bool:
        h = self._scale_hist
        while h and ts - h[0][0] > _SCALE_WINDOW_MS:
            h.popleft()
        if len(h) < 2:
            return True
        old_ts, old_s = h[0]
        dt = ts - old_ts
        if dt <= 0.0 or old_s <= 0.0:
            return True
        pct = abs(h[-1][1] - old_s) / old_s * 100.0
        return pct * (100.0 / dt) <= self.cfg.pinch.scale_stability_pct_per_100ms

    def _latch_pos(self, cursor: CursorSample, ts: float) -> tuple[float, float] | None:
        """Conditional position latch at the first below-threshold sample."""
        p = self.cfg.pinch
        if cursor.speed_px_s >= p.latch_max_speed_px_s:
            return None
        target = ts - p.latch_lookback_ms
        best: tuple[float, float] | None = None
        for t, x, y in self._cursor_hist:
            if t <= target:
                best = (x, y)
            else:
                break
        if best is None and self._cursor_hist:
            t, x, y = self._cursor_hist[0]
            best = (x, y)
        return best or (cursor.x, cursor.y)

    def _in_double_window(self, ts: float, cursor: CursorSample) -> bool:
        if self._last_up is None:
            return False
        up_ts, ux, uy = self._last_up
        c = self.cfg.click
        if ts - up_ts > c.double_ms:
            return False
        return ((cursor.x - ux) ** 2 + (cursor.y - uy) ** 2) ** 0.5 <= c.double_max_px

    def _clear_pinch(self) -> None:
        self._left_since = None
        self._left_latch = None
        self._right_since = None
        self._arb = None

    def _enter_hands_lost(self, ts: float) -> None:
        self.state = EngineState.HANDS_LOST
        self._lost_since = ts
        self._reacquire_hold.reset()
        self._prev_anchor = None
        self._clear_pinch()
        self._scroll_hold.reset()
        self._scroll_exit_since = None
        # A real hand-loss invalidates any in-progress open-palm hold outright
        # rather than waiting out the jitter grace — the grace exists for
        # brief tracking blips while the hand is still visible, not this.
        self._open_palm_hold.reset()
        self._release_since = None
        self._release_latch = None

    def _to_pointer(self) -> None:
        self.state = EngineState.POINTER
        self._clutch_hold.reset()
        self._clear_pinch()
        self._scroll_hold.reset()
        self._scroll_exit_since = None
        self._dragging = False
        self._locked_axis = None
        self._release_since = None
        self._release_latch = None
        self._prev_anchor = None

    # -- update -------------------------------------------------------------

    def update(self, frame: LandmarkFrame, cursor: CursorSample) -> EngineOutput:
        ts = frame.ts_ms
        intents: list[Intent] = []
        rebase: tuple[float, float] | None = None
        valid = self._hand_valid(frame, ts)
        open_palm = False
        anchor: Point | None = None

        if valid:
            lm = frame.landmarks
            assert lm is not None
            anchor = lm[INDEX_MCP]
            open_palm = self._open_palm_pose(lm)
            self._left_val = self._pinch_filter(
                "left", dist(lm[THUMB_TIP], lm[INDEX_TIP]) / frame.scale, ts
            )
            self._right_val = self._pinch_filter(
                "right", dist(lm[THUMB_TIP], lm[MIDDLE_TIP]) / frame.scale, ts
            )
            self._scale_hist.append((ts, frame.scale))
            self._cursor_hist.append((ts, cursor.x, cursor.y))
            while self._cursor_hist and ts - self._cursor_hist[0][0] > _HISTORY_MS:
                self._cursor_hist.popleft()

        # Debounced every frame regardless of state: `.since` directly gates
        # PALM entry (held continuously, within jitter grace, for
        # palm.enter_ms) and the debounced boolean below is what both PALM
        # exit and the palm detector see — a single dropped-finger frame mid
        # gesture can no longer reset progress or cut a swipe/pinch short.
        open_palm_since = self._open_palm_hold.update(ts, open_palm)
        open_palm_debounced = open_palm_since is not None
        self._last_open_palm = open_palm_debounced

        # PALM detector sees EVERY frame (it manages its own windows).
        palm_intents: list[Intent] = []
        if self._palm is not None:
            palm_intents = self._palm.update(frame, open_palm_debounced)

        stable = self._scale_stable(ts) if valid else True

        # Hand loss / anchor teleport from any active state.
        if self.state in _ACTIVE_STATES:
            if not valid:
                self._enter_hands_lost(ts)
            elif (
                self._prev_anchor is not None
                and anchor is not None
                and dist(anchor, self._prev_anchor) > _ANCHOR_JUMP_FRAC * frame.img_w
            ):
                self._enter_hands_lost(ts)
        if valid and self.state is not EngineState.HANDS_LOST:
            self._prev_anchor = anchor

        st = self.state
        if st is EngineState.CLUTCH_WAIT:
            rebase = self._tick_clutch(frame, cursor, ts, valid)
        elif st is EngineState.POINTER:
            rebase = self._tick_pointer(
                frame, cursor, ts, stable, open_palm_since, intents
            )
        elif st is EngineState.PINCHED:
            rebase = self._tick_pinched(cursor, ts, stable, intents)
        elif st is EngineState.RIGHT_PINCH:
            rebase = self._tick_right(ts, stable, intents)
        elif st is EngineState.SCROLL:
            rebase = self._tick_scroll(frame, cursor, ts, intents)
        elif st is EngineState.PALM:
            rebase = self._tick_palm(cursor, open_palm_debounced)
        elif st is EngineState.HANDS_LOST:
            rebase = self._tick_hands_lost(frame, cursor, ts, valid, intents)

        # Forward PALM-detector intents: everything while in PALM; also from
        # a POINTER hand, because a real open-palm-then-swipe motion is one
        # fluid gesture that often completes BEFORE the engine has spent
        # palm.enter_ms formally transitioning its own state to PALM (that
        # hold is about pose, not motion, so a fast continuous swipe finishes
        # first) -- gating swipes on "already in PALM" silently dropped
        # exactly the natural case, not just an edge case.
        #
        # Swipes and pinch/spread need different POINTER-side gates:
        # - Swipes already require open_palm + their own displacement/
        #   velocity/direction thresholds inside palm.py, and a genuine swipe
        #   IS fast anchor motion by definition -- also requiring a slow
        #   CURSOR (palm.forward_max_speed_px_s) would fight the very motion
        #   being detected, since cursor speed tracks anchor speed. No extra
        #   gate here.
        # - pinch_in/spread_out don't require fast motion, so the speed gate
        #   still does its original job there: don't fire Launchpad/Show
        #   Desktop just because the cursor is moving normally.
        if palm_intents:
            if self.state is EngineState.PALM:
                intents.extend(palm_intents)
            elif self.state is EngineState.POINTER:
                swipe_names = {
                    self.cfg.bindings.get(k) for k in
                    ("swipe_left", "swipe_right", "swipe_up", "swipe_down")
                } - {None}
                pinch_spread_names = {
                    self.cfg.bindings.get("pinch_in"),
                    self.cfg.bindings.get("spread_out"),
                } - {None}
                slow_enough = cursor.speed_px_s < self.cfg.palm.forward_max_speed_px_s
                for i in palm_intents:
                    if i.name in swipe_names or (
                        i.name in pinch_spread_names and slow_enough
                    ):
                        intents.append(i)

        freeze = self.state in (
            EngineState.CLUTCH_WAIT,
            EngineState.SCROLL,
            EngineState.PALM,
            EngineState.HANDS_LOST,
        )
        drag = self.state is EngineState.PINCHED and self._dragging
        self._last_ts = ts
        return EngineOutput(intents=intents, freeze=freeze, drag=drag, rebase=rebase)

    # -- per-state ticks -----------------------------------------------------

    def _tick_clutch(
        self, frame: LandmarkFrame, cursor: CursorSample, ts: float, valid: bool
    ) -> tuple[float, float] | None:
        pose = valid and frame.landmarks is not None and self._pointer_pose(frame.landmarks)
        since = self._clutch_hold.update(ts, pose)
        if since is not None and ts - since >= self._clutch_required:
            self._to_pointer()
            return (cursor.x, cursor.y)
        return None

    def _tick_pointer(
        self,
        frame: LandmarkFrame,
        cursor: CursorSample,
        ts: float,
        stable: bool,
        open_palm_since: float | None,
        intents: list[Intent],
    ) -> tuple[float, float] | None:
        lm = frame.landmarks
        assert lm is not None
        pinch_active = (
            self._left_since is not None
            or self._right_since is not None
            or self._arb is not None
        )

        # PALM entry (only with a detector wired in, and no pinch in flight).
        # open_palm_since is the debounced hold's start ts (already jitter-
        # tolerant), so entry is a plain duration check against it.
        if (
            self._palm is not None
            and not pinch_active
            and open_palm_since is not None
            and ts - open_palm_since >= self.cfg.palm.enter_ms
        ):
            self.state = EngineState.PALM
            self._clear_pinch()
            self._scroll_hold.reset()
            return None

        # SCROLL entry: pose sustained AND low cursor speed at entry.
        sc = self.cfg.scroll
        pose = self._scroll_pose(lm, frame.scale)
        since = self._scroll_hold.update(ts, pose)
        if (
            since is not None
            and ts - since >= sc.enter_ms
            and cursor.speed_px_s < sc.entry_max_speed_px_s
        ):
            self.state = EngineState.SCROLL
            self._scroll_hold.reset()
            self._scroll_exit_since = None
            self._scroll_y0 = lm[INDEX_MCP].y
            self._scroll_x0 = lm[INDEX_MCP].x
            self._clear_pinch()
            return None

        down = self._tick_pinch_candidates(lm, cursor, ts, stable)
        if down is not None:
            intents.append(down)
            return None
        if self.state is not EngineState.POINTER:
            return None  # right pinch confirmed: DOWN is deferred, no intent yet

        if not cursor.frozen:
            intents.append(
                Intent("move", Phase.MOVE, {"x": cursor.x, "y": cursor.y}, ts)
            )
        return None

    # -- pinch candidacy / arbitration (POINTER) ------------------------------

    def _tick_pinch_candidates(
        self,
        lm: tuple[Point, ...],
        cursor: CursorSample,
        ts: float,
        stable: bool,
    ) -> Intent | None:
        p = self.cfg.pinch
        lv, rv = self._left_val, self._right_val

        # Scale-stability gate: no pinch state transitions during a yaw/depth
        # transient — candidacy restarts once the hand scale settles.
        if not stable:
            self._clear_pinch()
            return None

        in_double = self._in_double_window(ts, cursor)
        left_thresh = self.cfg.click.double_reengage if in_double else p.left_engage

        if lv < left_thresh:
            if self._left_since is None:
                self._left_since = ts
                self._left_cc = 2 if in_double else 1
                self._left_latch = self._latch_pos(cursor, ts)
        else:
            self._left_since = None
            self._left_latch = None

        # Right candidacy REQUIRES a clearly extended index (kills the
        # thumb-transit-past-index misfire class).
        if rv < p.right_engage and self._index_extended(lm):
            if self._right_since is None:
                self._right_since = ts
        else:
            self._right_since = None

        if self._arb is not None:
            a = self._arb
            a.left_min = min(a.left_min, lv)
            a.right_min = min(a.right_min, rv)
            if ts < a.until:
                return None
            self._arb = None
            if a.left_min <= a.right_min:
                winner, w_min, l_min = "left", a.left_min, a.right_min
            else:
                winner, w_min, l_min = "right", a.right_min, a.left_min
            if l_min > p.arbitration_ratio * w_min:
                if winner == "left" and lv < p.left_release:
                    return self._confirm_left(cursor, ts)
                if winner == "right" and rv < p.right_release and self._index_extended(lm):
                    return self._confirm_right(cursor, ts)
            # Ambiguous (or the winner already let go): do nothing, restart.
            self._clear_pinch()
            return None

        left_done = (
            self._left_since is not None and ts - self._left_since >= p.engage_ms
        )
        right_done = (
            self._right_since is not None and ts - self._right_since >= p.engage_ms
        )
        if not (left_done or right_done):
            return None

        # Hold confirmation ~release_ms and pick the smaller distance at its
        # minimum whenever the other pinch is also plausibly in play.
        competing = (left_done and right_done) or (
            left_done and (rv < p.right_release or self._right_since is not None)
        ) or (
            right_done and (lv < p.left_release or self._left_since is not None)
        )
        if competing:
            self._arb = _Arbitration(until=ts + p.release_ms, left_min=lv, right_min=rv)
            return None
        if left_done:
            return self._confirm_left(cursor, ts)
        return self._confirm_right(cursor, ts)

    def _confirm_left(self, cursor: CursorSample, ts: float) -> Intent:
        pos = self._left_latch or (cursor.x, cursor.y)
        cc = self._left_cc
        self.state = EngineState.PINCHED
        self._held = ("left", cc, pos)
        self._down_pos = pos
        self._freeze_ref = (cursor.x, cursor.y)
        self._dragging = False
        self._locked_axis = None
        self._release_since = None
        self._release_latch = None
        self._clear_pinch()
        self._scroll_hold.reset()
        if cc == 2:
            self._last_up = None  # a double consumes the window (no triples)
        return Intent(
            "left", Phase.DOWN, {"x": pos[0], "y": pos[1], "click_count": cc}, ts
        )

    def _confirm_right(self, cursor: CursorSample, ts: float) -> Intent | None:
        """Right click is tap-only and its DOWN is DEFERRED to the release:
        nothing is posted while the pinch is held, so a held right pinch that
        drifts cannot right-drag, and a hold past click.right_tap_max_ms
        aborts cleanly (no orphan DOWN to unwind). ``_held`` stays None —
        the synth has no button down for HANDS_LOST / suspend to release."""
        pos = (cursor.x, cursor.y)
        self.state = EngineState.RIGHT_PINCH
        self._right_down_ts = ts
        self._down_pos = pos
        self._release_since = None
        self._release_latch = None
        self._clear_pinch()
        self._scroll_hold.reset()
        return None

    # -- PINCHED (left held: click pending or dragging) -----------------------

    def _tick_pinched(
        self, cursor: CursorSample, ts: float, stable: bool, intents: list[Intent]
    ) -> tuple[float, float] | None:
        c = self.cfg.click
        p = self.cfg.pinch
        rebase: tuple[float, float] | None = None
        emit_drag = False
        drag_pos = self._down_pos

        if not self._dragging:
            # Distance-only unfreeze: no time clause, so a careful slow click
            # can never micro-drag.
            dx = cursor.x - self._freeze_ref[0]
            dy = cursor.y - self._freeze_ref[1]
            if (dx * dx + dy * dy) ** 0.5 > c.drag_unfreeze_px:
                self._dragging = True
                adx, ady = abs(dx), abs(dy)
                frac = c.drag_axis_deadband_pct / 100.0
                total = adx + ady
                self._locked_axis = None
                if total > 0.0:
                    if adx < frac * total:
                        self._locked_axis = "x"
                    elif ady < frac * total:
                        self._locked_axis = "y"
                # Rebase to the frozen point; drag MOVEs start next frame
                # once the rebased sample comes through.
                rebase = self._down_pos
        else:
            px, py = cursor.x, cursor.y
            if self._locked_axis == "x":
                if abs(cursor.x - self._down_pos[0]) > c.drag_unfreeze_px:
                    self._locked_axis = None
                else:
                    px = self._down_pos[0]
            elif self._locked_axis == "y":
                if abs(cursor.y - self._down_pos[1]) > c.drag_unfreeze_px:
                    self._locked_axis = None
                else:
                    py = self._down_pos[1]
            drag_pos = (px, py)
            emit_drag = True

        held = self._held
        cc = held[1] if held is not None else 1
        if self._dragging:
            rel_thresh, rel_ms = p.drag_release, p.release_ms
        elif cc == 2:
            # Relaxed one-sample release inside the double-click window.
            rel_thresh, rel_ms = c.double_release, 0.0
        else:
            rel_thresh, rel_ms = p.left_release, p.release_ms

        if self._left_val > rel_thresh:
            if self._release_since is None:
                self._release_since = ts
                # UP posts at the position latched at the first
                # above-threshold sample (drag) / the mouse-down point.
                self._release_latch = drag_pos if self._dragging else self._down_pos
            if stable and ts - self._release_since >= rel_ms:
                pos = self._release_latch or self._down_pos
                intents.append(
                    Intent(
                        "left",
                        Phase.UP,
                        {"x": pos[0], "y": pos[1], "click_count": cc},
                        ts,
                    )
                )
                self._last_up = None if cc == 2 else (ts, pos[0], pos[1])
                self._held = None
                self._to_pointer()
                # Continue cursor output from where the visible cursor is (the
                # UP position) — without this, the first POINTER move after a
                # click snaps to the live drifted sample.
                return (pos[0], pos[1])
        else:
            self._release_since = None
            self._release_latch = None

        if emit_drag:
            intents.append(
                Intent("drag", Phase.MOVE, {"x": drag_pos[0], "y": drag_pos[1]}, ts)
            )
            if held is not None:
                self._held = (held[0], held[1], drag_pos)
        return rebase

    # -- RIGHT_PINCH (tap-only, deferred DOWN+UP at release) --------------------

    def _tick_right(
        self, ts: float, stable: bool, intents: list[Intent]
    ) -> tuple[float, float] | None:
        p = self.cfg.pinch
        if self._right_val > p.right_release:
            if self._release_since is None:
                self._release_since = ts
            if stable and ts - self._release_since >= p.release_ms:
                pos = self._down_pos
                held_ms = self._release_since - self._right_down_ts
                if held_ms <= self.cfg.click.right_tap_max_ms:
                    intents.append(
                        Intent("right", Phase.DOWN, {"x": pos[0], "y": pos[1]}, ts)
                    )
                    intents.append(
                        Intent("right", Phase.UP, {"x": pos[0], "y": pos[1]}, ts)
                    )
                # Held too long: abort silently (no right-drag in v1).
                self._to_pointer()
                # The visible cursor sat parked at the tap point all along;
                # continue from there rather than snapping to the live sample.
                return pos
        else:
            self._release_since = None
        return None

    # -- SCROLL (vertical joystick, cursor frozen) -----------------------------

    def _tick_scroll(
        self,
        frame: LandmarkFrame,
        cursor: CursorSample,
        ts: float,
        intents: list[Intent],
    ) -> tuple[float, float] | None:
        sc = self.cfg.scroll
        lm = frame.landmarks
        assert lm is not None
        if self._scroll_pose(lm, frame.scale):
            self._scroll_exit_since = None
            # Horizontal joystick in the same pose = tab switch. One trigger
            # per deflection past the dead zone, then a refractory; the origin
            # re-latches so holding the hand off-center doesn't machine-gun.
            dx_frac = (lm[INDEX_MCP].x - self._scroll_x0) / frame.img_w
            if abs(dx_frac) > sc.tab_deadzone_frac and ts >= self._tab_block_until:
                # Mirrored (selfie) frame: hand moving right = +x = next tab.
                name = "tab_next" if dx_frac > 0 else "tab_prev"
                intents.append(Intent(name, Phase.TRIGGER, {}, ts))
                self._tab_block_until = ts + sc.tab_refractory_ms
                self._scroll_x0 = lm[INDEX_MCP].x
            deflection = (lm[INDEX_MCP].y - self._scroll_y0) / frame.img_h
            mag = abs(deflection) - sc.deadzone_frac
            dt_s = 0.0
            if self._last_ts is not None and ts > self._last_ts:
                dt_s = (ts - self._last_ts) / 1000.0
            if mag > 0.0 and dt_s > 0.0:
                vel = min(sc.gain * mag, sc.max_px_s)
                sign = 1.0 if deflection > 0.0 else -1.0
                if sc.invert:
                    sign = -sign
                dy_px = sign * vel * dt_s
                intents.append(Intent("scroll", Phase.MOVE, {"dy_px": dy_px}, ts))
            return None
        # Velocity zeroes the moment exit-counting starts: no scroll intents
        # while the pose is down.
        if self._scroll_exit_since is None:
            self._scroll_exit_since = ts
        if ts - self._scroll_exit_since >= sc.exit_ms:
            self._to_pointer()
            return (cursor.x, cursor.y)
        return None

    # -- PALM (delegated system gestures, cursor frozen) -----------------------

    def _tick_palm(
        self, cursor: CursorSample, open_palm_debounced: bool
    ) -> tuple[float, float] | None:
        # open_palm_debounced already absorbed the jitter grace (it only goes
        # false after grace_ms of sustained non-open frames), so a single
        # dropped-finger frame mid-gesture can't cut PALM short.
        if open_palm_debounced:
            return None
        self._to_pointer()
        return (cursor.x, cursor.y)

    # -- HANDS_LOST -------------------------------------------------------------

    def _tick_hands_lost(
        self,
        frame: LandmarkFrame,
        cursor: CursorSample,
        ts: float,
        valid: bool,
        intents: list[Intent],
    ) -> tuple[float, float] | None:
        # Auto-mouseUp: no state ever parks with a synthetic button down.
        if (
            self._held is not None
            and self._lost_since is not None
            and ts - self._lost_since >= self.cfg.hands_lost_ms
        ):
            intents.extend(self._release_held(ts))

        pose = valid and frame.landmarks is not None and self._pointer_pose(frame.landmarks)
        since = self._reacquire_hold.update(ts, pose)
        if since is not None and ts - since >= self.cfg.clutch.reacquire_ms:
            # Safety: if the hand came back fast enough that the auto-UP
            # never fired, release before handing the cursor back.
            if self._held is not None:
                intents.extend(self._release_held(ts))
            self._lost_since = None
            self._reacquire_hold.reset()
            self._to_pointer()
            return (cursor.x, cursor.y)
        return None

    def _release_held(self, ts: float) -> list[Intent]:
        if self._held is None:
            return []
        name, cc, pos = self._held
        self._held = None
        self._dragging = False
        payload: dict = {"x": pos[0], "y": pos[1]}
        if name == "left":
            payload["click_count"] = cc
        return [Intent(name, Phase.UP, payload, ts)]
