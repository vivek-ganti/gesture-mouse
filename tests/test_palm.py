"""Tests for gesture_mouse.palm (PALM-mode system gestures).

Builds synthetic LandmarkFrame sequences inline (no shared helpers): frames
are constructed so the palm centroid (mean of landmarks 0,5,9,13,17) sits
exactly on the INDEX_MCP anchor and the five fingertips sit at distance
m * scale from it, making the spread metric exact.
"""
from __future__ import annotations

import pytest

from gesture_mouse.config import DEFAULT_BINDINGS, PalmConfig
from gesture_mouse.palm import PalmDetector
from gesture_mouse.types import FINGER_TIPS, LandmarkFrame, Phase, Point

FRAME_W, FRAME_H = 640, 480
SCALE = 100.0
STEP_MS = 33.0  # ~30 fps

# Offsets for the palm-centroid landmarks (0,5,9,13,17); they sum to zero so
# the centroid lands exactly on the anchor, and index 5 IS the anchor.
_BASE_OFFSETS = {0: (0.0, 40.0), 5: (0.0, 0.0), 9: (20.0, 0.0),
                 13: (-20.0, 0.0), 17: (0.0, -40.0)}
# Five unit vectors (pentagon) for fingertip placement around the centroid.
_TIP_DIRS = [(1.0, 0.0), (0.309, -0.951), (-0.809, -0.588),
             (-0.809, 0.588), (0.309, 0.951)]


def make_frame(ts_ms: float, x: float, y: float, m: float = 1.0,
               hand: bool = True) -> LandmarkFrame:
    if not hand:
        return LandmarkFrame(ts_ms=ts_ms, handedness=None, landmarks=None,
                             img_w=FRAME_W, img_h=FRAME_H, confidence=0.0,
                             scale=0.0, source="test")
    pts = [Point(x, y)] * 21
    for idx, (ox, oy) in _BASE_OFFSETS.items():
        pts[idx] = Point(x + ox, y + oy)
    r = m * SCALE
    for tip, (ux, uy) in zip(FINGER_TIPS, _TIP_DIRS):
        pts[tip] = Point(x + ux * r, y + uy * r)
    return LandmarkFrame(ts_ms=ts_ms, handedness="Right", landmarks=tuple(pts),
                         img_w=FRAME_W, img_h=FRAME_H, confidence=0.9,
                         scale=SCALE, source="test")


def feed(det: PalmDetector, steps):
    """steps: iterable of (frame, open_palm); returns all emitted intents."""
    intents = []
    for frame, open_palm in steps:
        intents.extend(det.update(frame, open_palm))
    return intents


# Enough rest frames to arm: the speed estimate needs a >=80ms trajectory
# baseline before it reports "at rest" at all, then arm_hold_ms (150) of
# open-palm-at-rest. 12 frames @30fps = ~400ms, comfortably past both.
N_REST = 12
# 6 motion frames x 45px = 270px net, past 0.25*640=160 (h) / 0.25*480=120 (v).
N_MOVE = 6
STEP_PX = 45.0


def rest_steps(t0: float = 0.0, x0: float = 320.0, y0: float = 240.0,
               n: int = N_REST, open_palm: bool = True):
    """Stationary open-palm frames: the arming prefix."""
    return [(make_frame(t0 + i * STEP_MS, x0, y0), open_palm)
            for i in range(n)]


def swipe_steps(direction: str, t0: float = 0.0, x0: float = 320.0,
                y0: float = 240.0, n_rest: int = N_REST, n_move: int = N_MOVE,
                step_px: float = STEP_PX, motion_pose: bool = False):
    """Arm-at-rest prefix, then a fast flick. The motion frames carry
    open_palm=False by DEFAULT — pose loss during fast motion is the norm on
    real data (motion blur), and the new model must not care."""
    ux, uy = {"left": (-1, 0), "right": (1, 0),
              "up": (0, -1), "down": (0, 1)}[direction]
    steps = rest_steps(t0, x0, y0, n_rest)
    for i in range(1, n_move + 1):
        ts = t0 + (n_rest - 1 + i) * STEP_MS
        steps.append((make_frame(ts, x0 + ux * step_px * i,
                                 y0 + uy * step_px * i), motion_pose))
    return steps


def default_detector() -> PalmDetector:
    return PalmDetector(PalmConfig(), dict(DEFAULT_BINDINGS))


@pytest.mark.parametrize("direction,expected", [
    ("left", "space_next"),
    ("right", "space_prev"),
    ("up", "mission_control"),
    ("down", "app_expose"),
])
def test_swipe_each_direction_fires_bound_intent_once(direction, expected):
    det = default_detector()
    intents = feed(det, swipe_steps(direction))
    assert [i.name for i in intents] == [expected]
    assert intents[0].phase is Phase.TRIGGER
    assert intents[0].ts_ms > 0.0


def test_swipe_cooldown_blocks_immediate_repeat():
    det = default_detector()
    cfg = det.cfg
    intents = feed(det, swipe_steps("left"))
    assert len(intents) == 1
    trigger_ts = intents[0].ts_ms

    # A full arm+swipe attempt entirely inside the cooldown: blocked.
    second = swipe_steps("left", t0=trigger_ts + STEP_MS)
    assert second[-1][0].ts_ms < trigger_ts + cfg.swipe_cooldown_ms
    intents += feed(det, second)
    assert len(intents) == 1  # blocked by cooldown

    # After the cooldown, a fresh arm+swipe fires again.
    t = trigger_ts + cfg.swipe_cooldown_ms + 2 * STEP_MS
    intents += feed(det, swipe_steps("left", t0=t))
    assert [i.name for i in intents] == ["space_next", "space_next"]


def test_swipe_requires_rearm_after_fire():
    det = default_detector()
    cfg = det.cfg
    intents = feed(det, swipe_steps("left"))
    assert len(intents) == 1
    trigger_ts = intents[0].ts_ms
    last_x = 320.0 - N_MOVE * STEP_PX

    # Keep MOVING (never resting) from the fire through well past the
    # cooldown: without a fresh arm (open palm at rest) nothing may fire,
    # no matter how much displacement accumulates.
    t = trigger_ts + STEP_MS
    x = last_x
    end = trigger_ts + cfg.swipe_cooldown_ms + 600.0
    direction = 1.0
    while t < end:
        x += 40.0 * direction
        if not 40.0 <= x <= 600.0:
            direction = -direction
        intents += feed(det, [(make_frame(t, x, 240.0), True)])
        t += STEP_MS
    assert len(intents) == 1

    # Now rest with the palm open (re-arm) and flick: fires.
    intents += feed(det, swipe_steps("right", t0=t + STEP_MS, x0=x))
    assert [i.name for i in intents] == ["space_next", "space_prev"]


def test_slow_palm_drift_does_not_trigger():
    det = default_detector()
    # ~150 px/s open-palm drift for 1.3s: displacement inside any swipe
    # window stays far below the threshold.
    steps = [(make_frame(i * STEP_MS, 400.0 - 5.0 * i, 240.0), True)
             for i in range(40)]
    assert feed(det, steps) == []


def test_wave_reversal_does_not_trigger_swipe():
    det = default_detector()
    # Arm properly, then wave: 99px left, back, left, back — fast legs, but
    # the NET displacement from the arming origin never reaches the 0.25
    # threshold (waving at someone on a call must not switch Spaces).
    steps = rest_steps()
    xs = [287, 254, 221, 254, 287, 320, 287, 254, 221, 254, 287, 320, 320]
    t0 = N_REST * STEP_MS
    steps += [(make_frame(t0 + i * STEP_MS, float(x), 240.0), True)
              for i, x in enumerate(xs)]
    assert feed(det, steps) == []


def test_swipe_survives_hand_loss_within_bridge():
    """THE headline regression: MediaPipe dropping the hand for a few frames
    mid-swipe is documented-normal under fast motion. Arm, start the flick,
    lose the hand for ~130ms (under the 350ms bridge), reappear displaced —
    the swipe must still fire. The old detector cleared everything on a
    single absent frame."""
    det = default_detector()
    steps = swipe_steps("left", n_move=2)          # 90px in, then...
    t0 = steps[-1][0].ts_ms
    for i in range(1, 5):                          # ...4 absent frames (~130ms)
        steps.append((make_frame(t0 + i * STEP_MS, 0, 0, hand=False), False))
    # Reappears well past the displacement threshold.
    steps.append((make_frame(t0 + 5 * STEP_MS, 320.0 - 280.0, 240.0), False))
    intents = feed(det, steps)
    assert [i.name for i in intents] == ["space_next"]


def test_gap_over_bridge_disarms():
    det = default_detector()
    steps = swipe_steps("left", n_move=2)
    t0 = steps[-1][0].ts_ms
    for i in range(1, 15):                         # ~460ms gap > 350ms bridge
        steps.append((make_frame(t0 + i * STEP_MS, 0, 0, hand=False), False))
    steps.append((make_frame(t0 + 15 * STEP_MS, 320.0 - 280.0, 240.0), False))
    assert feed(det, steps) == []


def test_pose_loss_during_motion_is_the_norm():
    """Every motion frame carries open_palm=False (what blur does to finger
    landmarks) — the swipe fires anyway; only ARMING needed the pose."""
    det = default_detector()
    intents = feed(det, swipe_steps("right", motion_pose=False))
    assert [i.name for i in intents] == ["space_prev"]


def test_arming_requires_rest():
    """Open palm that is ALREADY moving fast never arms — big displacement
    with no at-rest arming prefix must not fire."""
    det = default_detector()
    steps = [(make_frame(i * STEP_MS, 600.0 - 45.0 * i, 240.0), True)
             for i in range(12)]                   # 495px of open-palm motion
    assert feed(det, steps) == []


def test_arming_requires_pose():
    """A fist at rest then a flick: never armed (no open palm), no fire."""
    det = default_detector()
    steps = swipe_steps("left")
    steps = [(frame, False) for frame, _ in steps]  # same motion, pose never on
    assert feed(det, steps) == []


def test_window_expiry_requires_rearm():
    """Armed, move partway (below threshold), then stall past
    swipe_max_duration_ms with the pose down: the episode expires, and a
    fast flick AFTER the expiry must not fire without a fresh arm."""
    det = default_detector()
    cfg = det.cfg
    steps = rest_steps()
    t0 = N_REST * STEP_MS
    # 3 fast frames x 33px = 99px: motion started, threshold not reached.
    for i in range(1, 4):
        steps.append((make_frame(t0 + i * STEP_MS, 320.0 - 33.0 * i, 240.0),
                      False))
    # Stall (fist, stationary) until the window is long expired.
    x_stall = 320.0 - 99.0
    t1 = t0 + 4 * STEP_MS
    n_stall = int((cfg.swipe_max_duration_ms + 300.0) / STEP_MS)
    for i in range(n_stall):
        steps.append((make_frame(t1 + i * STEP_MS, x_stall, 240.0), False))
    # Fast flick with no re-arm: dead.
    t2 = t1 + n_stall * STEP_MS
    for i in range(1, 7):
        steps.append((make_frame(t2 + i * STEP_MS, x_stall - 45.0 * i, 240.0),
                      False))
    assert feed(det, steps) == []


def test_diagonal_motion_does_not_fire():
    det = default_detector()
    steps = rest_steps()
    t0 = N_REST * STEP_MS
    # 45-degree-equivalent in FRACTION space: equal fx and fy per frame.
    for i in range(1, 7):
        steps.append((make_frame(t0 + i * STEP_MS,
                                 320.0 - 0.06 * 640 * i,
                                 240.0 - 0.06 * 480 * i), False))
    assert feed(det, steps) == []


def test_five_finger_pinch_in_fires_launchpad_once():
    det = default_detector()
    # Open palm collapses to a fist; open_palm drops as fingers curl, but the
    # gesture must still fire (pinch-in does not require the pose flag).
    ms = [1.0, 1.0, 1.0, 0.9, 0.75, 0.62, 0.5, 0.4, 0.4, 0.4, 0.4]
    flags = [True, True, True, True, False, False, False, False, False,
             False, False]
    steps = [(make_frame(i * STEP_MS, 320, 240, m=m), f)
             for i, (m, f) in enumerate(zip(ms, flags))]
    intents = feed(det, steps)
    assert [i.name for i in intents] == ["launchpad"]
    assert intents[0].phase is Phase.TRIGGER
    # Holding the fist past the refractory still cannot re-fire (no open
    # sample left inside the spread window).
    t0 = len(ms) * STEP_MS
    more = [(make_frame(t0 + i * STEP_MS, 320, 240, m=0.4), False)
            for i in range(30)]
    assert feed(det, more) == []


def test_five_finger_spread_out_fires_show_desktop_once():
    det = default_detector()
    # Curled hand (pose never formed) spreading wide open.
    ms = [0.4, 0.4, 0.4, 0.5, 0.7, 0.9, 1.05, 1.15, 1.15]
    flags = [False, False, False, False, False, False, True, True, True]
    steps = [(make_frame(i * STEP_MS, 320, 240, m=m), f)
             for i, (m, f) in enumerate(zip(ms, flags))]
    intents = feed(det, steps)
    assert [i.name for i in intents] == ["show_desktop"]
    assert intents[0].phase is Phase.TRIGGER


def test_custom_bindings_respected():
    bindings = {"swipe_left": "custom_left", "swipe_right": "custom_right",
                "swipe_up": "custom_up", "swipe_down": "custom_down",
                "pinch_in": "custom_pinch", "spread_out": "custom_spread"}
    det = PalmDetector(PalmConfig(), bindings)
    intents = feed(det, swipe_steps("left"))
    assert [i.name for i in intents] == ["custom_left"]

    det2 = PalmDetector(PalmConfig(), bindings)
    ms = [1.0, 1.0, 0.8, 0.6, 0.45, 0.4]
    steps = [(make_frame(i * STEP_MS, 320, 240, m=m), m > 0.9)
             for i, m in enumerate(ms)]
    assert [i.name for i in feed(det2, steps)] == ["custom_pinch"]


def test_hand_loss_clears_windows():
    det = default_detector()
    # Open palm, then the hand disappears; when it reappears already curled
    # (inside what would have been the spread window) pinch-in must NOT fire
    # because the pre-loss open samples were cleared.
    steps = [(make_frame(i * STEP_MS, 320, 240, m=1.0), True) for i in range(3)]
    steps.append((make_frame(3 * STEP_MS, 0, 0, hand=False), False))
    steps += [(make_frame((4 + i) * STEP_MS, 320, 240, m=0.4), False)
              for i in range(3)]
    assert feed(det, steps) == []
    assert det.active is False


def test_active_flag_tracks_open_palm():
    det = default_detector()
    det.update(make_frame(0.0, 320, 240), open_palm=True)
    assert det.active is True
    det.update(make_frame(STEP_MS, 320, 240), open_palm=False)
    assert det.active is False
    det.reset()
    assert det.active is False
