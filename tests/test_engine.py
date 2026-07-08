"""Golden Intent-sequence tests for gesture_mouse.engine.

Each test replays a synthetic 30fps LandmarkFrame sequence (tests/helpers.py)
through the REAL CursorPipeline + GestureEngine, applying EngineOutput's
freeze/drag/rebase back to the pipeline exactly like __main__'s hot loop, and
asserts the exact Intent name/phase sequence.
"""
from __future__ import annotations

import random

import pytest

from gesture_mouse.config import Config
from gesture_mouse.engine import GestureEngine, _FingerState, _pip_angle_deg
from gesture_mouse.filters import CursorPipeline, LandmarkSmoother
from gesture_mouse.types import INDEX_MCP, INDEX_PIP, INDEX_TIP, EngineState, Phase, Point

from helpers import STEP_MS, Seq
from pose_fixtures import make_bent_frame

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
        self.armed = False
        self.engaged = False
        self.disarm_calls = 0

    def update(self, frame, open_palm):
        from gesture_mouse.types import Intent
        self.calls += 1
        self.active = open_palm
        return [Intent(self.name, Phase.TRIGGER, {}, frame.ts_ms)]

    def disarm(self):
        self.disarm_calls += 1
        self.armed = False
        self.engaged = False

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


def test_horns_pose_fires_custom_gesture_with_action_payload():
    """Devil-horns 🤘 held ~300ms at a still hand fires the configured custom
    gesture exactly once, carrying its action dict for synth to execute —
    works from CLUTCH_WAIT too (no cursor clutch needed, like swipes)."""
    cfg = Config()
    assert cfg.custom_gestures, "default config ships the dictate example"
    s = Seq(pose="horns")
    s.hold("horns", 600)                 # rest + hold_ms with margin
    intents, _, eng = run(s.frames, cfg=cfg)

    fired = named(intents, "custom:dictate")
    assert len(fired) == 1
    assert fired[0].phase is Phase.TRIGGER
    assert fired[0].payload["action"] == {"type": "key", "key": "option"}
    assert eng.state is EngineState.CLUTCH_WAIT   # horns is not the clutch pose


def test_custom_gesture_cooldown_blocks_continuous_hold():
    cfg = Config()
    s = Seq(pose="horns")
    s.hold("horns", 1100)                # held straight through the cooldown
    intents, _, _ = run(s.frames, cfg=cfg)
    assert len(named(intents, "custom:dictate")) == 1

    # After cooldown + re-hold, it fires again.
    s2 = Seq(pose="horns", start_ms=s.ts + 1300.0)
    s2.hold("horns", 600)
    # continue through the SAME engine by replaying both sequences
    intents_all, _, _ = run(s.frames + s2.frames, cfg=cfg)
    assert len(named(intents_all, "custom:dictate")) == 2


def test_custom_gesture_not_fired_during_pinch():
    cfg = Config()
    s = Seq()
    s.hold("pointer", 400)
    s.pinch_to(0.20, 100)                # left pinch in flight
    s.hold(ms=600)
    intents, _, _ = run(s.frames, cfg=cfg)
    assert named(intents, "custom:dictate") == []


def test_unknown_custom_pose_skipped_and_surfaced():
    cfg = Config()
    cfg.custom_gestures = [{"name": "bad", "pose": "nosuch",
                            "action": {"type": "key", "key": "option"}}]
    from gesture_mouse.filters import CursorPipeline
    pipe = CursorPipeline(cfg, *SCREEN)
    eng = GestureEngine(cfg, pipe.pinch)
    assert eng.custom_skipped == ["bad"]


def test_v2_signature_custom_gesture_fires():
    cfg = Config()
    cfg.custom_gestures = [{
        "name": "rock",
        "signature": {"index": "ext", "middle": "curl", "ring": "curl", "pinky": "ext"},
        "hold_ms": 300.0, "cooldown_ms": 1200.0,
        "action": {"type": "key", "key": "option"},
    }]
    s = Seq(pose="horns")
    s.hold("horns", 600)
    intents, _, _ = run(s.frames, cfg=cfg)
    fired = named(intents, "custom:rock")
    assert len(fired) == 1
    assert fired[0].payload["action"] == {"type": "key", "key": "option"}


def test_reload_customs_picks_up_new_entry_without_reset():
    cfg = Config()
    pipe = CursorPipeline(cfg, *SCREEN)
    eng = GestureEngine(cfg, pipe.pinch)
    assert [g["name"] for g in eng._custom] == ["dictate"]
    latch = eng._latches["index"]
    cfg.custom_gestures.append({
        "name": "added",
        "signature": {"middle": "ext", "ring": "ext", "index": "curl", "pinky": "curl"},
        "action": {"type": "shell", "argv": ["true"]},
    })
    eng.reload_customs()
    assert [g["name"] for g in eng._custom] == ["dictate", "added"]
    assert eng._latches["index"] is latch  # transient state untouched


def test_finger_angles_exact_on_first_frame_and_cleared_on_loss():
    from pose_fixtures import make_bent_frame
    from helpers import lost_frame
    from gesture_mouse.types import CursorSample

    cfg = Config()
    pipe = CursorPipeline(cfg, *SCREEN)
    eng = GestureEngine(cfg, pipe.pinch)
    frame = make_bent_frame(0.0, {"index": 90.0, "middle": 180.0, "ring": 45.0, "pinky": 0.0})
    eng.update(frame, pipe.update(frame))
    # OneEuro first-sample passthrough makes frame-1 angles exact.
    assert eng.finger_angles["index"] == pytest.approx(90.0)
    assert eng.finger_angles["middle"] == pytest.approx(180.0)
    lost = lost_frame(STEP_MS)
    eng.update(lost, pipe.update(lost))
    assert eng.finger_angles == {}


# -- angle-based pose classification (plan doc: gesture reliability) --------
#
# _pip_angle_deg/_FingerState replaced a single hard-cutoff ratio test with
# an angle metric (180 = straight/extended, 0 = folded/curled) plus a real
# hysteresis band, mirroring the engage/release pattern already proven by
# pinch detection. helpers.py's Seq/make_frame fixtures keep every joint
# collinear (see pose_fixtures.py's module docstring for why), so the tests
# below either check the pure geometry directly or use pose_fixtures.py's
# genuinely-bent fixtures.


class TestPipAngle:
    def test_straight_line_is_180(self):
        lm = (Point(0.0, 0.0), Point(0.0, -35.0), Point(0.0, -80.0))
        assert _pip_angle_deg(lm, 0, 1, 2) == pytest.approx(180.0)

    def test_folded_back_is_0(self):
        lm = (Point(0.0, 0.0), Point(0.0, -35.0), Point(0.0, 10.0))
        assert _pip_angle_deg(lm, 0, 1, 2) == pytest.approx(0.0)

    def test_right_angle_bend_is_90(self):
        lm = (Point(0.0, 0.0), Point(0.0, -35.0), Point(-45.0, -35.0))
        assert _pip_angle_deg(lm, 0, 1, 2) == pytest.approx(90.0)

    def test_degenerate_coincident_points_treated_as_straight(self):
        lm = (Point(0.0, 0.0), Point(0.0, -35.0), Point(0.0, -35.0))
        assert _pip_angle_deg(lm, 0, 1, 2) == 180.0


class TestFingerStateHysteresis:
    def test_latches_extended_and_ignores_midband_noise(self):
        fs = _FingerState()
        assert fs.update(170.0, 160.0, 130.0) is True  # clears extend threshold
        # Oscillate inside the dead zone between curl(130) and extend(160):
        # a naive single-threshold test would flicker on every one of these;
        # the latch, once set, must not.
        for angle in (145.0, 132.0, 158.0, 131.0, 150.0):
            assert fs.update(angle, 160.0, 130.0) is True
        assert fs.update(120.0, 160.0, 130.0) is False  # clears curl threshold

    def test_starts_curled_and_stays_so_in_midband(self):
        fs = _FingerState()
        assert fs.update(145.0, 160.0, 130.0) is False

    def test_reset_returns_to_curled(self):
        fs = _FingerState()
        fs.update(170.0, 160.0, 130.0)
        fs.reset()
        assert fs.update(145.0, 160.0, 130.0) is False


class TestBentJointPoseClassification:
    """Regression coverage for the exact gap pose_fixtures.py's docstring
    describes: helpers.py's collinear "half" fixture reads as fully
    extended (180 deg) under the angle metric, so it can never prove a
    genuinely half-curled/relaxed hand "matches no pose" (engine.py module
    docstring: "a relaxed hand keeps POINTER" without tripping any other
    gesture) — these bent-joint fixtures actually flex the PIP joint."""

    def _engine(self):
        cfg = Config()
        return GestureEngine(cfg, lambda name, raw, ts: raw), cfg

    def test_genuinely_half_curled_hand_matches_no_pose(self):
        eng, _ = self._engine()
        frame = make_bent_frame(
            0.0, {"index": 90.0, "middle": 90.0, "ring": 90.0, "pinky": 90.0}
        )
        lm = eng._smoother.smooth(frame)
        assert eng._pointer_pose(lm) is False
        assert eng._open_palm_pose(lm) is False
        assert eng._scroll_pose(lm, frame.scale) is False
        assert eng._horns_pose(lm) is False

    def test_genuinely_extended_hand_matches_open_palm(self):
        eng, _ = self._engine()
        frame = make_bent_frame(
            0.0, {"index": 180.0, "middle": 180.0, "ring": 180.0, "pinky": 180.0}
        )
        lm = eng._smoother.smooth(frame)
        assert eng._open_palm_pose(lm) is True

    def test_genuine_pointer_shape_matches_pointer_only(self):
        eng, _ = self._engine()
        frame = make_bent_frame(
            0.0, {"index": 180.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
        )
        lm = eng._smoother.smooth(frame)
        assert eng._pointer_pose(lm) is True
        assert eng._open_palm_pose(lm) is False


def test_horns_with_thumb_on_middle_fires_custom_not_right_click():
    # The real rock sign 🤘 rests the thumb ON the curled middle fingertip —
    # thumb-middle distance sits well inside right_engage with the index
    # extended, a perfect fake right-pinch. Without the custom-pose gate in
    # _tick_pinch_candidates this confirmed RIGHT_PINCH, blocked the custom
    # loop (state gate), and fired a right CLICK on release instead of the
    # bound action. (The user's exact "rock horns don't work" report.)
    cfg = Config()
    s = Seq()
    s.hold("pointer", 400)          # clutch -> POINTER
    s.hold("horns", 33)
    s.pinch_middle_to(0.2, 66)      # thumb comes to rest on the middle tip
    s.hold(ms=600)                  # hold the sign well past hold_ms=300
    intents, _, eng = run(s.frames, cfg=cfg)
    assert len(named(intents, "custom:dictate")) == 1
    assert named(intents, "right") == []
    assert eng.state is not EngineState.RIGHT_PINCH


def test_slow_horns_formation_cannot_confirm_right_pinch():
    # The formation RACE (distinct from the resting-thumb case above): the
    # thumb reaches the middle fingertip BEFORE the middle finger's latch
    # reads curled, so the custom-pose suppression gate is still inactive —
    # for that window the hand is a perfect fake right-pinch, and >100ms of
    # it used to confirm RIGHT_PINCH, which blocks the custom loop for as
    # long as the sign is held ("horns never fires"). The pinky-curled
    # requirement on right candidacy kills this: in the rock sign the pinky
    # is up from the very first frame, while a real right-click pose always
    # has it folded.
    cfg = Config()
    s = Seq()
    s.hold("pointer", 400)              # clutch -> POINTER
    s.hold("horns_forming", 33)         # pinky up, middle NOT yet curled
    s.pinch_middle_to(0.2, 66)          # thumb lands on the middle tip early
    s.hold(ms=200)                      # >engage_ms in the fake-pinch window
    s.hold("horns", 500)                # sign completes; dictate must fire
    intents, _, eng = run(s.frames, cfg=cfg)
    assert named(intents, "right") == []
    assert len(named(intents, "custom:dictate")) == 1
    assert eng.state is not EngineState.RIGHT_PINCH


def test_scroll_enters_with_ring_stuck_half_extended():
    # Anatomical regression: after an open palm latches every finger
    # extended, the ring physically cannot fully curl while the middle
    # stays extended — it hovers in the hysteresis band and stays latched
    # extended. Scroll's signature must not care (ring is "any").
    from pose_fixtures import make_bent_frame

    cfg = Config()
    pipe = CursorPipeline(cfg, *SCREEN)
    eng = GestureEngine(cfg, pipe.pinch)
    ts = 0.0

    def feed(angles, ms):
        nonlocal ts
        for _ in range(max(1, round(ms / STEP_MS))):
            f = make_bent_frame(ts, angles)
            eng.update(f, pipe.update(f))
            ts += STEP_MS

    feed({"index": 175.0, "middle": 40.0, "ring": 40.0, "pinky": 40.0}, 400)  # clutch
    assert eng.state is EngineState.POINTER
    feed({"index": 175.0, "middle": 175.0, "ring": 175.0, "pinky": 175.0}, 200)  # open palm: all latch ext
    # Two fingers up + pinky curled, ring stuck at 145 deg (between curl 130
    # and extend 160 -> latch HOLDS its extended state from the open palm).
    feed({"index": 175.0, "middle": 175.0, "ring": 145.0, "pinky": 40.0}, 400)
    assert eng.state is EngineState.SCROLL


class TestYModeLatchSync:
    def test_finger_states_truthful_in_y_extended_test_mode(self):
        # The y-mode _ext branch must keep the latches in sync — the panel
        # readout and capture-by-demonstration read finger_states, and a
        # permanently all-curled latch set would make every capture freeze
        # a fist signature regardless of the demonstrated pose.
        cfg = Config()
        cfg.options.extended_test = "y"
        eng = GestureEngine(cfg, lambda name, raw, ts: raw)
        s = Seq(pose="horns")
        s.hold("horns", 100)
        pipe = CursorPipeline(cfg, *SCREEN)
        for f in s.frames:
            eng.update(f, pipe.update(f))
        states = eng.finger_states
        assert states["index"] is True
        assert states["pinky"] is True
        assert states["middle"] is False
        assert states["ring"] is False


class TestRetunePoseSmoothing:
    def test_pushes_cfg_mincutoff_into_the_live_smoother(self):
        cfg = Config()
        eng = GestureEngine(cfg, lambda name, raw, ts: raw)
        eng._smoother.smooth(make_bent_frame(0.0, {"index": 180.0}))  # primes filters
        cfg.pose.smoothing_mincutoff = 8.25
        eng.retune_pose_smoothing()
        for fx, fy in eng._smoother._filters.values():
            assert fx.mincutoff == 8.25
            assert fy.mincutoff == 8.25


class TestLandmarkPreSmoothingReducesFlicker:
    def test_smoothing_cuts_boundary_flicker_beyond_hysteresis_alone(self):
        # A finger held near the extend/curl midpoint with realistic
        # per-frame jitter: raw angles (even through the SAME hysteresis
        # latch) still cross both thresholds often; pre-smoothing the
        # landmarks (as engine.update() now does before every pose test)
        # must cut that further. This isolates smoothing's own contribution
        # on top of the hysteresis fix — the exact failure mode behind
        # "gets the gesture sometimes, mostly misses."
        rng = random.Random(0)
        cfg = Config()
        smoother = LandmarkSmoother(cfg)
        raw_state, smoothed_state = _FingerState(), _FingerState()
        raw_flips = smoothed_flips = 0
        raw_prev = smoothed_prev = None
        base_angle = 150.0  # midpoint of extend(160)/curl(130)
        t = 0.0
        for _ in range(120):
            angle = max(0.0, min(180.0, base_angle + rng.uniform(-25.0, 25.0)))
            frame = make_bent_frame(t, {"index": angle})
            lm_smooth = smoother.smooth(frame)
            angle_raw = _pip_angle_deg(frame.landmarks, INDEX_MCP, INDEX_PIP, INDEX_TIP)
            angle_smooth = _pip_angle_deg(lm_smooth, INDEX_MCP, INDEX_PIP, INDEX_TIP)
            r = raw_state.update(angle_raw, cfg.pose.extend_angle_deg, cfg.pose.curl_angle_deg)
            s = smoothed_state.update(
                angle_smooth, cfg.pose.extend_angle_deg, cfg.pose.curl_angle_deg
            )
            if raw_prev is not None and r != raw_prev:
                raw_flips += 1
            if smoothed_prev is not None and s != smoothed_prev:
                smoothed_flips += 1
            raw_prev, smoothed_prev = r, s
            t += STEP_MS
        assert smoothed_flips < raw_flips
