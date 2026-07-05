"""Golden Intent-sequence tests for gesture_mouse.engine.

Each test replays a synthetic 30fps LandmarkFrame sequence (tests/helpers.py)
through the REAL CursorPipeline + GestureEngine, applying EngineOutput's
freeze/drag/rebase back to the pipeline exactly like __main__'s hot loop, and
asserts the exact Intent name/phase sequence.
"""
from __future__ import annotations

from gesture_mouse.config import Config
from gesture_mouse.engine import GestureEngine
from gesture_mouse.filters import CursorPipeline
from gesture_mouse.types import EngineState, Phase

from helpers import STEP_MS, Seq

SCREEN = (1280.0, 800.0)


def run(frames, cfg=None, palm=None):
    """Replay frames through pipeline + engine; returns (intents, log, engine)."""
    cfg = cfg or Config()
    pipe = CursorPipeline(cfg, *SCREEN)
    eng = GestureEngine(cfg, pipe.pinch, palm)
    log = []
    intents = []
    for f in frames:
        cur = pipe.update(f)
        out = eng.update(f, cur)
        pipe.set_frozen(out.freeze)
        pipe.set_drag(out.drag)
        if out.rebase is not None:
            pipe.rebase(*out.rebase)
        log.append((f, cur, out))
        intents.extend(out.intents)
    return intents, log, eng


def named(intents, *names):
    return [i for i in intents if i.name in names]


def phases(intents):
    return [(i.name, i.phase) for i in intents]


def pos(intent):
    return (intent.payload["x"], intent.payload["y"])


# -- left click -------------------------------------------------------------


def test_careful_stationary_click_exactly_down_up_no_drag():
    s = Seq()
    s.hold("pointer", 400)          # clutch -> POINTER
    s.pinch_to(0.12, 66)
    s.hold(ms=250)                  # careful ~500ms pinch overall
    s.release_pinch(66)
    s.hold(ms=300)
    intents, log, eng = run(s.frames)

    left = named(intents, "left")
    assert phases(left) == [("left", Phase.DOWN), ("left", Phase.UP)]
    assert left[0].payload["click_count"] == 1
    assert left[1].payload["click_count"] == 1
    # Zero micro-drag: no drag intents at all, and UP posts at the DOWN point.
    assert named(intents, "drag") == []
    assert named(intents, "right", "scroll") == []
    assert pos(left[0]) == pos(left[1])
    # Stationary + slow -> conditional latch used the (identical) old position.
    assert eng.state is EngineState.POINTER
    # POINTER emitted cursor moves before the pinch.
    moves = named(intents, "move")
    assert moves and moves[0].ts_ms < left[0].ts_ms


def test_fly_and_pinch_clicks_at_current_position():
    s = Seq()
    s.hold("pointer", 400)
    s.pinch_to(0.12, 33)            # pinch starts...
    s.move_to((450.0, 240.0), 400)  # ...while the hand keeps flying
    s.release_pinch(66)
    s.hold(ms=200)
    intents, log, eng = run(s.frames)

    left = named(intents, "left")
    assert phases(left) == [("left", Phase.DOWN), ("left", Phase.UP)]
    down = left[0]
    by_ts = {c.ts_ms: c for _, c, _ in log}
    cur = by_ts[down.ts_ms]
    # Fast hand at pinch start -> no stale latch: DOWN at the current position.
    assert pos(down) == (cur.x, cur.y)
    earlier = [c for _, c, _ in log if down.ts_ms - 134 <= c.ts_ms <= down.ts_ms - 96]
    assert earlier
    dx = abs(cur.x - earlier[-1].x)
    assert dx > 30.0  # a 100ms-old latch would have been way behind


def test_drag_freeze_unfreeze_deadband_and_release_latch():
    s = Seq()
    s.hold("pointer", 400)
    s.pinch_to(0.12, 66)
    s.hold(ms=200)                  # DOWN lands; cursor stays frozen
    t_move = s.ts
    s.move_to((350.0, 240.0), 200)  # +30 camera px = +100 screen px, pure x
    s.hold(ms=300)
    s.release_pinch(66)
    s.hold(ms=300)
    intents, log, eng = run(s.frames)

    left = named(intents, "left")
    drags = named(intents, "drag")
    assert phases(left) == [("left", Phase.DOWN), ("left", Phase.UP)]
    down, up = left
    assert len(drags) >= 3
    # Frozen until the 12px distance-only unfreeze: no drag before the move.
    assert all(d.ts_ms >= t_move for d in drags)
    assert all(down.ts_ms < d.ts_ms < up.ts_ms for d in drags)
    # Rebase-to-frozen-point: the drag stream starts near the DOWN position.
    d0 = pos(drags[0])
    assert abs(d0[0] - down.payload["x"]) < 30.0
    # Minor-axis dead-band: pure-x drag never wobbles y off the DOWN row.
    assert all(d.payload["y"] == down.payload["y"] for d in drags)
    # UP posts at the position latched at the first above-threshold sample —
    # one of the emitted drag positions, not some later drift.
    assert pos(up) in [pos(d) for d in drags]
    assert eng.state is EngineState.POINTER


# -- double click -------------------------------------------------------------


def _tap(s: Seq, hold_ms: float = 250.0) -> None:
    s.pinch_to(0.12, 66)
    s.hold(ms=hold_ms)
    s.release_pinch(66)


def test_double_click_carries_click_count_2():
    s = Seq()
    s.hold("pointer", 400)
    _tap(s)
    s.hold(ms=100)                  # second tap well inside the 500ms window
    _tap(s)
    s.hold(ms=300)
    intents, _, _ = run(s.frames)

    left = named(intents, "left")
    assert phases(left) == [
        ("left", Phase.DOWN), ("left", Phase.UP),
        ("left", Phase.DOWN), ("left", Phase.UP),
    ]
    assert [i.payload["click_count"] for i in left] == [1, 1, 2, 2]
    assert named(intents, "drag") == []
    # Second tap really landed inside the window.
    assert left[2].ts_ms - left[1].ts_ms < 500.0


def test_two_slow_singles_stay_click_count_1():
    s = Seq()
    s.hold("pointer", 400)
    _tap(s)
    s.hold(ms=700)                  # > click.double_ms after the UP
    _tap(s)
    s.hold(ms=300)
    intents, _, _ = run(s.frames)

    left = named(intents, "left")
    assert phases(left) == [
        ("left", Phase.DOWN), ("left", Phase.UP),
        ("left", Phase.DOWN), ("left", Phase.UP),
    ]
    assert [i.payload["click_count"] for i in left] == [1, 1, 1, 1]
    assert left[2].ts_ms - left[1].ts_ms > 500.0


# -- right click ---------------------------------------------------------------


def test_right_click_thumb_transit_past_index_no_left_intents():
    s = Seq()
    s.hold("pointer", 300)
    s.hold("right_click", 200)       # settle filters with a half-curled middle
    s.thumb_to((0.0, -68.0), 133)    # sweep grazes the extended index tip...
    s.hold(ms=66)
    s.thumb_to((27.0, -18.0), 100)   # ...en route to the middle fingertip
    s.hold(ms=400)                   # thumb parked on the middle tip
    s.release_pinch(66)
    s.hold(ms=300)
    intents, _, eng = run(s.frames)

    assert named(intents, "left") == []
    assert named(intents, "drag") == []
    right = named(intents, "right")
    assert phases(right) == [("right", Phase.DOWN), ("right", Phase.UP)]
    assert pos(right[0]) == pos(right[1])   # tap-only, cursor parked
    assert eng.state is EngineState.POINTER


# -- scroll --------------------------------------------------------------------


def test_scroll_entry_requires_low_cursor_speed():
    s = Seq()
    s.hold("pointer", 400)
    s.hold("scroll", ms=33)
    s.move_to((520.0, 240.0), 300)   # scroll pose held while flying: gated
    t_fast_end = s.ts
    s.hold(ms=300)                   # speed decays -> SCROLL entry
    s.move_to((520.0, 300.0), 200)   # joystick deflection downward
    s.hold("pointer", ms=300)
    intents, _, _ = run(s.frames)

    scrolls = named(intents, "scroll")
    assert scrolls, "scroll never engaged after the hand slowed down"
    assert all(i.ts_ms > t_fast_end for i in scrolls)
    assert all(i.payload["dy_px"] > 0.0 for i in scrolls)
    assert named(intents, "left", "right", "drag") == []


def test_scroll_emits_moves_and_has_no_post_exit_tail():
    cfg = Config()
    s = Seq()
    s.hold("pointer", 400)
    t_pose_start = s.ts
    s.hold("scroll", 200)
    s.move_to((320.0, 300.0), 300)   # anchor moves down 60 camera px
    s.hold(ms=100)
    t_pose_end = s.ts                # scroll pose drops here
    s.hold("pointer", ms=400)
    intents, _, eng = run(s.frames)

    scrolls = named(intents, "scroll")
    assert len(scrolls) >= 3
    assert all(i.phase is Phase.MOVE for i in scrolls)
    assert all(i.payload["dy_px"] > 0.0 for i in scrolls)
    # Velocity zeroes the moment exit-counting starts: zero tail after the
    # last posed frame.
    assert all(i.ts_ms < t_pose_end for i in scrolls)
    # Cursor frozen throughout SCROLL: no move intents from entry until the
    # exit debounce has elapsed.
    t_enter_latest = t_pose_start + cfg.scroll.enter_ms + 1.5 * STEP_MS
    t_exit_earliest = t_pose_end + cfg.scroll.exit_ms - 1.0
    frozen_moves = [
        i for i in named(intents, "move")
        if t_enter_latest <= i.ts_ms < t_exit_earliest
    ]
    assert frozen_moves == []
    # Moves resume after the exit rebase.
    assert any(i.ts_ms > t_exit_earliest for i in named(intents, "move"))
    assert eng.state is EngineState.POINTER


# -- hands lost ------------------------------------------------------------------


def test_hands_lost_mid_drag_auto_mouseup_within_window():
    cfg = Config()
    s = Seq()
    s.hold("pointer", 400)
    s.pinch_to(0.12, 66)
    s.hold(ms=200)
    s.move_to((350.0, 240.0), 200)   # dragging now
    t_lost = s.ts
    s.lose_hand(500)
    intents, _, eng = run(s.frames)

    left = named(intents, "left")
    drags = named(intents, "drag")
    assert phases(left) == [("left", Phase.DOWN), ("left", Phase.UP)]
    up = left[1]
    assert up.ts_ms >= t_lost + cfg.hands_lost_ms
    assert up.ts_ms <= t_lost + cfg.hands_lost_ms + 1.5 * STEP_MS
    # UP posts where the drag last was.
    assert drags and pos(up) == pos(drags[-1])
    # Parked: nothing after the auto-UP.
    assert all(i.ts_ms <= up.ts_ms for i in intents)
    assert eng.state is EngineState.HANDS_LOST


def test_wrong_handedness_is_treated_as_absent():
    cfg = Config()
    cfg.hand = "right"               # explicit pin (default is "auto")
    s = Seq(handedness="Left")
    s.hold("pointer", 400)
    s.pinch_to(0.12, 66)
    s.hold(ms=250)
    s.release_pinch(66)
    s.hold(ms=300)
    intents, _, eng = run(s.frames, cfg=cfg)

    assert intents == []             # never even clutches in
    assert eng.state is EngineState.CLUTCH_WAIT


def test_hand_auto_tracks_either_handedness():
    """Default hand='auto': a 'Left'-labeled hand (mirror-convention-broken
    cameras, left-handed use) is tracked and can click — the regression that
    made a real recording produce zero intents."""
    s = Seq(handedness="Left")
    s.hold("pointer", 400)
    s.pinch_to(0.12, 100)
    s.hold(ms=250)
    s.release_pinch(100)
    s.hold(ms=300)
    intents, _, eng = run(s.frames)  # default Config: hand == "auto"

    left = named(intents, "left")
    assert phases(left) == [("left", Phase.DOWN), ("left", Phase.UP)]
    assert eng.state is EngineState.POINTER


def test_reacquire_after_loss_needs_250ms_of_pointer_pose():
    cfg = Config()
    s = Seq()
    s.hold("pointer", 400)           # POINTER, moves flowing
    s.lose_hand(300)
    t_return = s.ts
    s.hold("pointer", ms=400)
    intents, _, eng = run(s.frames)

    moves = named(intents, "move")
    reacquire_at = t_return + cfg.clutch.reacquire_ms
    # 150ms of pose is not enough — no cursor motion until the full 250ms.
    assert all(not (t_return <= i.ts_ms < reacquire_at) for i in moves)
    assert any(i.ts_ms >= reacquire_at for i in moves)
    assert eng.state is EngineState.POINTER


# -- scale stability ---------------------------------------------------------------


def test_scale_stability_gate_blocks_pinch_during_yaw_transient():
    # Same pinch depth that clicks in the careful-click test, but the hand
    # scale collapses >15%/100ms underneath it (yaw/depth transient): the
    # gate must swallow the whole pinch.
    s = Seq()
    s.hold("pointer", 400)
    s.pinch_to(0.12, 66)
    s.scale_to(0.55, 200)            # ~22%/100ms shrink while "pinched"
    s.release_pinch(33)
    s.hold(ms=400)
    intents, _, eng = run(s.frames)

    assert named(intents, "left", "right", "drag", "scroll") == []
    assert eng.state is EngineState.POINTER


# -- PALM delegation -----------------------------------------------------------


class _StubPalm:
    """Duck-typed PalmDetector: emits one TRIGGER per frame it sees."""

    def __init__(self, name="space_next"):
        self.name = name
        self.active = False
        self.calls = 0

    def update(self, frame, open_palm):
        from gesture_mouse.types import Intent
        self.calls += 1
        self.active = open_palm
        return [Intent(self.name, Phase.TRIGGER, {}, frame.ts_ms)]

    def reset(self):
        self.active = False


def test_palm_mode_freezes_cursor_and_forwards_detector_intents():
    """Swipe-bound intents forward from POINTER too now (see
    test_swipe_bound_intent_forwarded_from_pointer_before_palm_entry for why:
    a real open-palm-then-swipe motion often completes before the engine has
    spent palm.enter_ms formally entering PALM), so this test only checks the
    two invariants that are still PALM-specific: the cursor freezes while in
    PALM, and the detector sees every single frame regardless of state."""
    cfg = Config()
    stub = _StubPalm()
    s = Seq()
    s.hold("pointer", 400)
    t_open = s.ts
    s.hold("open", 300)              # all-5-extended -> PALM after enter_ms
    t_close = s.ts
    s.hold("pointer", ms=300)        # pose ends -> POINTER with rebase
    intents, _, eng = run(s.frames, cfg=cfg, palm=stub)

    # PALM exit is debounced (pose_jitter_grace_ms) so a single dropped-finger
    # frame can't cut a gesture short — PALM lingers up to that long past the
    # pose actually ending, which pushes the window's upper bound out.
    t_exit_max = t_close + cfg.pose_jitter_grace_ms

    trig = named(intents, "space_next")
    assert trig and all(i.phase is Phase.TRIGGER for i in trig)
    # Cursor frozen in PALM: no move intents between entry and (debounced) exit.
    assert all(
        not (t_open + cfg.palm.enter_ms + 1.5 * STEP_MS <= i.ts_ms <= t_exit_max)
        for i in named(intents, "move")
    )
    # Detector saw EVERY frame, not just PALM ones.
    assert stub.calls == len(s.frames)
    assert eng.state is EngineState.POINTER


def test_swipe_bound_intent_forwarded_from_pointer_before_palm_entry():
    """The actual bug behind real hand swipes not registering: a natural
    open-palm-then-swipe motion is one fluid gesture, and palm.py's own
    swipe detector already requires the open-palm pose plus its own
    displacement/velocity/direction thresholds -- but the engine used to
    ALSO require it had separately spent palm.enter_ms transitioning its own
    state to PALM before forwarding any swipe-bound intent. A hand that
    opens and swipes fast never satisfies that redundant gate before the
    swipe itself completes, so a correctly-detected swipe was silently
    dropped. Reproduced here with the hand NEVER holding the open pose long
    enough to formally enter PALM at all (a single 'open' frame) -- the
    swipe-bound stub intent must still be forwarded, purely from POINTER."""
    cfg = Config()
    stub = _StubPalm(name="space_next")  # bound to "swipe_left" by default
    s = Seq()
    s.hold("pointer", 400)      # clutch engage -> POINTER
    s.hold("open", ms=STEP_MS)  # a single frame -- nowhere near enter_ms (80ms)
    intents, _, eng = run(s.frames, cfg=cfg, palm=stub)

    assert named(intents, "space_next")
    assert eng.state is EngineState.POINTER  # never entered PALM at all


def test_pinch_spread_bound_intent_still_gated_by_cursor_speed_from_pointer():
    """Regression guard for the half of the forwarding logic that must NOT
    change: pinch_in/spread_out-bound intents don't require fast motion the
    way swipes do, so the cursor-speed gate still applies to them specifically
    when forwarded from POINTER (not PALM) -- a fast-moving cursor must not
    accidentally trigger Launchpad/Show Desktop."""
    cfg = Config()
    stub = _StubPalm(name="launchpad")  # bound to "pinch_in" by default
    s = Seq()
    s.hold("pointer", 400)                 # clutch engage -> POINTER (stationary)
    t_fast_start = s.ts
    s.move_to((320.0 + 200.0, 240.0), 50)  # fast: well over forward_max_speed_px_s
    intents, _, eng = run(s.frames, cfg=cfg, palm=stub)

    # Stub fires every frame regardless of pose, so it's expected (and fine)
    # to be forwarded during the earlier STATIONARY pointer hold -- the
    # invariant under test is specifically that fast motion blocks it.
    fast_hits = [i for i in named(intents, "launchpad") if i.ts_ms >= t_fast_start]
    assert fast_hits == [], f"launchpad forwarded during fast motion: {fast_hits}"
    assert eng.state is EngineState.POINTER
