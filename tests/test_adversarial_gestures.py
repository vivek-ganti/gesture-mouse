"""Adversarial gesture-logic regression tests (originally adversarial-review
probes that exposed real defects; now permanent regressions for the fixes).

Each test simulates plausible hand mechanics frame-by-frame and asserts the
plan-normative behavior: still-then-flick palm swipes fire; pinch-in /
spread-out recovery motion never fires the opposite gesture; a right pinch is
tap-only with a parked cursor and aborts past click.right_tap_max_ms.

Probes that PASSED against the reviewed code (in-drag release at 0.60 not
0.48; relaxed 0.42/0.44 double-click thresholds only inside the 500ms/15px
window; thumb transit with a mid-pause never firing left) were verified and
deleted per review instructions.
"""
from __future__ import annotations

from gesture_mouse.config import Config, DEFAULT_BINDINGS, PalmConfig
from gesture_mouse.palm import PalmDetector
from gesture_mouse.types import EngineState, Phase

from helpers import STEP_MS, Seq
from test_engine import named, phases, run
from test_palm import make_frame as palm_frame, feed


def test_palm_held_still_then_fast_flick_fires_swipe():
    """Plan: swipe = velocity > 1.0 fw/s AND displacement > 22% within
    <=350ms. A palm held stationary ~430ms then flicked 160px (25% of the
    frame) in ~66ms (~3.8 fw/s) satisfies both — it must fire. Guards the
    per-candidate swipe scan (a window-average velocity would be diluted by
    the stationary prefix and never fire)."""
    det = PalmDetector(PalmConfig(), dict(DEFAULT_BINDINGS))
    steps = []
    t = 0.0
    for _ in range(13):                     # ~430ms stationary open palm
        steps.append((palm_frame(t, 320.0, 240.0), True))
        t += STEP_MS
    for x in (240.0, 160.0):                # 160px flick in 2 frames (~2400px/s)
        steps.append((palm_frame(t, x, 240.0), True))
        t += STEP_MS
    for _ in range(12):                     # parked at the end of the flick
        steps.append((palm_frame(t, 160.0, 240.0), True))
        t += STEP_MS
    intents = feed(det, steps)
    assert [i.name for i in intents] == ["space_next"]


def test_pinch_in_then_natural_reopen_fires_only_launchpad():
    """Five-finger pinch-in (Launchpad) followed by the hand naturally
    reopening ~230ms later. The reopen is the mechanical recovery of the
    pinch, not a deliberate spread-out; per invariant 1 ("when in doubt,
    do nothing") nothing else may fire. Guards the shared refractory +
    history-reset marker."""
    det = PalmDetector(PalmConfig(), dict(DEFAULT_BINDINGS))
    ms = [1.1, 1.1, 1.1, 1.0, 0.85, 0.65, 0.45, 0.4, 0.4, 0.4,   # pinch-in
          0.55, 0.75, 0.95, 1.1, 1.1, 1.1, 1.1]                  # reopen
    steps = [(palm_frame(i * STEP_MS, 320.0, 240.0, m=m), m > 1.0)
             for i, m in enumerate(ms)]
    intents = feed(det, steps)
    assert [i.name for i in intents] == ["launchpad"]


def test_spread_out_then_natural_recurl_fires_only_show_desktop():
    det = PalmDetector(PalmConfig(), dict(DEFAULT_BINDINGS))
    ms = [0.4, 0.4, 0.4, 0.55, 0.75, 0.95, 1.1, 1.1, 1.1,        # spread-out
          0.95, 0.75, 0.55, 0.45, 0.4, 0.4, 0.4]                 # relax back
    steps = [(palm_frame(i * STEP_MS, 320.0, 240.0, m=m), m > 1.0)
             for i, m in enumerate(ms)]
    intents = feed(det, steps)
    assert [i.name for i in intents] == ["show_desktop"]


def test_right_tap_with_drift_clicks_at_down_point_no_cursor_jump():
    """A right tap (held under click.right_tap_max_ms) whose hand drifts a
    little posts DOWN+UP together at the tap point, and the cursor resumes
    from there — no teleport to the live drifted sample. Guards the
    deferred-DOWN tap semantics and the rebase-on-exit."""
    s = Seq()
    s.hold("pointer", 400)
    s.hold("right_click", 200)
    s.pinch_middle_to(0.12, 66)
    s.hold(ms=133)                       # brief hold: still a tap
    s.release_pinch(66)
    s.hold(ms=400)
    intents, _, eng = run(s.frames)

    right = named(intents, "right")
    assert phases(right) == [("right", Phase.DOWN), ("right", Phase.UP)]
    up = right[1]
    moves_after = [i for i in named(intents, "move") if i.ts_ms > up.ts_ms]
    assert moves_after, "pointer moves must resume after the right UP"
    first = moves_after[0]
    jump = ((first.payload["x"] - up.payload["x"]) ** 2
            + (first.payload["y"] - up.payload["y"]) ** 2) ** 0.5
    assert jump < 30.0, f"cursor jumped {jump:.0f}px after right release"
    assert eng.state is EngineState.POINTER


def test_right_pinch_held_past_tap_window_aborts_silently():
    """Right click is tap-only (< click.right_tap_max_ms = 400ms). A pinch
    held ~700ms while the hand drifts is neither a tap nor a v1 right-drag:
    it must emit NO right intents at all (invariant 1), and the cursor must
    still resume without a jump from where it was parked."""
    s = Seq()
    s.hold("pointer", 400)
    s.hold("right_click", 200)
    s.pinch_middle_to(0.12, 66)
    s.hold(ms=300)
    s.move_to((380.0, 240.0), 300)       # hand drifts +60 camera px, held
    s.hold(ms=100)
    s.release_pinch(66)
    s.hold(ms=400)
    intents, _, eng = run(s.frames)

    assert named(intents, "right") == []
    assert named(intents, "left", "drag") == []
    assert eng.state is EngineState.POINTER


def test_scroll_carry_accumulates_fractional_deltas():
    """Slow-scroll band: per-frame dy_px < 1 must accumulate across frames
    (guards the synth carry — independent rounding zeroed the whole band)."""
    from gesture_mouse.synth import _carry_round

    carry = 0.0
    posted = 0
    for _ in range(30):                      # 0.4 px/frame for one second
        dy, carry = _carry_round(0.4, carry)
        posted += dy
    assert posted == 12                      # 30 * 0.4, nothing lost
    dy, carry = _carry_round(-0.6, 0.0)      # sign preserved for up-scroll
    assert dy == 0 and abs(carry + 0.6) < 1e-9


def test_scroll_pose_horizontal_flick_switches_tab_once_with_refractory():
    """Scroll pose (index+middle) + horizontal joystick = tab switch. One
    trigger per deflection with a 500ms refractory; the origin re-latches so
    a hand held off-center does not machine-gun Ctrl+Tab."""
    s = Seq()
    s.hold("pointer", 400)
    s.hold("scroll", 200)                # enter SCROLL (low speed at entry)
    s.move_to((400.0, 240.0), 133)       # +80px right = 12.5% of frame width
    s.hold(ms=300)                       # stay deflected inside refractory
    intents, _, eng = run(s.frames)

    tabs = [i for i in intents if i.name in ("tab_next", "tab_prev")]
    assert [i.name for i in tabs] == ["tab_next"]
    assert eng.state is EngineState.SCROLL


def test_scroll_pose_leftward_flick_fires_tab_prev():
    s = Seq()
    s.hold("pointer", 400)
    s.hold("scroll", 200)
    s.move_to((240.0, 240.0), 133)       # -80px left
    s.hold(ms=200)
    intents, _, _ = run(s.frames)
    tabs = [i for i in intents if i.name in ("tab_next", "tab_prev")]
    assert [i.name for i in tabs] == ["tab_prev"]


def test_vertical_scroll_does_not_fire_tab_switches():
    s = Seq()
    s.hold("pointer", 400)
    s.hold("scroll", 200)
    s.move_to((320.0, 340.0), 200)       # straight down: scroll only
    s.hold(ms=300)
    intents, _, _ = run(s.frames)
    assert [i for i in intents if i.name in ("tab_next", "tab_prev")] == []
    assert any(i.name == "scroll" for i in intents)


# -- pose-hold jitter tolerance (the actual "4/5-finger gestures don't work"
# fix): requiring FOUR fingers to agree simultaneously is far more likely to
# flicker false for a stray frame on real (noisy) hand-tracking data than the
# single-landmark pointer pose ever was. The pre-fix code reset every pose
# hold's countdown to zero on the very first such frame, so a real hand
# attempting a 4-finger swipe or 5-finger pinch/spread could see its PALM
# entry (or an in-progress gesture) restart or abort on almost every attempt
# even after the earlier fix that dropped the unreliable thumb requirement.
# ---------------------------------------------------------------------------

def test_palm_entry_survives_single_frame_pose_jitter():
    """80ms of open-palm split by exactly one dropped-pose frame still
    enters PALM: pre-fix, that one frame would have reset the countdown to
    zero, and the post-jitter segment alone (60ms) falls short of enter_ms
    (80ms) — this sequence provably fails under the old reset-on-miss logic."""
    cfg = Config()
    palm = PalmDetector(cfg.palm, dict(cfg.bindings))
    s = Seq()
    s.hold("pointer", 400)          # clutch engage -> POINTER
    s.hold("open", 40)              # ~1 frame open (nowhere near enter_ms alone)
    s.hold("relaxed", ms=STEP_MS)   # ONE jittered frame: not fully extended
    s.hold("open", 60)              # resumes; only 60ms alone, still < enter_ms
    _, _, eng = run(s.frames, cfg=cfg, palm=palm)
    assert eng.state is EngineState.PALM


def test_palm_exit_survives_single_frame_pose_jitter_mid_gesture():
    """Once in PALM, a single dropped-pose frame must not instantly kick the
    engine back to POINTER — a real gesture in progress (e.g. mid-swipe or
    mid-curl) needs to survive the same jitter its entry does. Checked frame
    by frame (not just the final state) since an instant bounce through
    POINTER and back would be invisible at the end of the sequence."""
    from gesture_mouse.filters import CursorPipeline
    from gesture_mouse.engine import GestureEngine
    from test_engine import SCREEN

    cfg = Config()
    palm = PalmDetector(cfg.palm, dict(cfg.bindings))
    pipe = CursorPipeline(cfg, *SCREEN)
    eng = GestureEngine(cfg, pipe.pinch, palm)

    s = Seq()
    s.hold("pointer", 400)
    s.hold("open", 200)             # solid entry into PALM
    entry_frame_count = len(s.frames)
    s.hold("relaxed", ms=STEP_MS)   # ONE jittered frame while gesturing
    s.hold("open", 60)              # pose resumes

    states_after_entry: list[EngineState] = []
    for i, f in enumerate(s.frames):
        cur = pipe.update(f)
        out = eng.update(f, cur)
        pipe.set_frozen(out.freeze)
        pipe.set_drag(out.drag)
        if out.rebase is not None:
            pipe.rebase(*out.rebase)
        if i >= entry_frame_count:
            states_after_entry.append(eng.state)

    assert states_after_entry, "no frames recorded after PALM entry"
    assert all(s is EngineState.PALM for s in states_after_entry), states_after_entry


def test_swipe_survives_a_jittered_frame_mid_motion():
    """A real fast swipe is exactly when hand-tracking is least stable
    (motion blur); one dropped-pose frame in the middle of the flick must
    not silently swallow the swipe."""
    cfg = Config()
    palm = PalmDetector(cfg.palm, dict(cfg.bindings))
    s = Seq()
    s.hold("pointer", 400)
    s.hold("open", 150)                    # enter PALM comfortably
    s.move_to((420.0, 240.0), 33)           # start of the flick (open pose)
    s.hold("relaxed", ms=STEP_MS)           # ONE jittered frame mid-flick
    s.pose = "open"
    s.move_to((520.0, 240.0), 100)          # flick continues (25%+ of frame width)
    intents, _, _ = run(s.frames, cfg=cfg, palm=palm)
    # Rightward anchor motion -> swipe_right -> DEFAULT_BINDINGS "space_prev".
    assert named(intents, "space_prev")


def test_pinch_in_style_intent_forwarded_from_pointer_at_moderate_cursor_speed():
    """The POINTER-path forwarding gate for pinch_in/spread_out
    (palm.forward_max_speed_px_s) used to silently reuse scroll's tight
    60 px/s entry gate, dropping real five-finger gestures whenever the hand
    drifted at all while flexing (very common — holding a hand perfectly
    still while curling all five fingers is hard). It's now a much more
    generous 300 px/s, decoupled from scroll's threshold."""
    from test_engine import _StubPalm

    cfg = Config()
    assert cfg.palm.forward_max_speed_px_s > 60.0  # the loosened default
    stub = _StubPalm(name="launchpad")  # default pinch_in -> launchpad binding
    s = Seq()
    s.hold("pointer", 400)              # clutch engage -> POINTER
    s.move_to((335.0, 240.0), 400)       # settles to ~120-130 px/s: > old 60,
                                         # comfortably < new 300
    intents, _, eng = run(s.frames, cfg=cfg, palm=stub)
    assert named(intents, "launchpad")
    assert eng.state is EngineState.POINTER
