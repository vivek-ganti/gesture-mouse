"""Tests for gesture_mouse.filters: OneEuro + CursorPipeline.

Pure logic — no camera, no OS. Frames are synthetic; timestamps are explicit
milliseconds. Geometry assumes the default config: control box x=0.20 y=0.45
w=0.60 h=0.35 of a 640x480 frame, mapped to a 1920x1080 screen (box top-left
= camera px (128, 216), box bottom-right = (512, 384), x-gain 5.0).
"""
from __future__ import annotations

import itertools
import math

import pytest

from gesture_mouse.config import Config
from gesture_mouse.filters import QUANTIZE_MAX_SPEED_PX_S, CursorPipeline, OneEuro
from gesture_mouse.types import LandmarkFrame, Point

SCREEN_W, SCREEN_H = 1920.0, 1080.0
IMG_W, IMG_H = 640, 480
FRAME_MS = 33.0  # ~30 fps


def make_frame(ts_ms: float, ax: float = 0.0, ay: float = 0.0,
               present: bool = True) -> LandmarkFrame:
    """A frame whose 21 landmarks all sit at the anchor point (only
    INDEX_MCP matters to the pipeline)."""
    if not present:
        return LandmarkFrame(ts_ms, None, None, IMG_W, IMG_H, 0.0, 0.0, "test")
    pts = tuple(Point(ax, ay) for _ in range(21))
    return LandmarkFrame(ts_ms, "Right", pts, IMG_W, IMG_H, 0.95, 100.0, "test")


def make_pipeline(cfg: Config | None = None) -> CursorPipeline:
    return CursorPipeline(cfg or Config(), SCREEN_W, SCREEN_H)


# ---------------------------------------------------------------- OneEuro --

class TestOneEuro:
    def test_first_sample_passes_through(self):
        f = OneEuro(1.0, 0.007)
        assert f.filter(5.0, 0.0) == 5.0

    def test_converges_on_constant_input(self):
        f = OneEuro(1.0, 0.007)
        f.filter(0.0, 0.0)
        out = 0.0
        for k in range(1, 91):  # 3 s at 30 Hz
            out = f.filter(100.0, k * FRAME_MS)
        assert out == pytest.approx(100.0, abs=0.1)

    def test_output_monotone_toward_constant_target(self):
        f = OneEuro(1.0, 0.007)
        prev = f.filter(0.0, 0.0)
        for k in range(1, 31):
            out = f.filter(100.0, k * FRAME_MS)
            assert prev < out <= 100.0
            prev = out

    def test_jitter_attenuation_at_low_speed(self):
        # 1 px sine jitter at 5 Hz around a stationary value: the filter must
        # attenuate the oscillation well below the raw peak-to-peak of ~2 px.
        f = OneEuro(1.0, 0.007)
        outs = []
        for k in range(151):  # 5 s
            t = k * FRAME_MS
            outs.append(f.filter(50.0 + math.sin(2 * math.pi * 5.0 * t / 1000.0), t))
        tail = outs[-30:]  # last second, transient settled
        assert max(tail) - min(tail) < 0.5  # < 25% of raw ptp

    def test_responsive_on_fast_ramp(self):
        # 1000 px/s ramp: adaptive cutoff (beta) must keep lag small, and
        # clearly smaller than a fixed-cutoff (beta=0) filter's lag.
        adaptive = OneEuro(1.0, 0.007)
        fixed = OneEuro(1.0, 0.0)
        lag_adaptive = lag_fixed = 0.0
        for k in range(61):  # 2 s
            t = k * FRAME_MS
            v = 1000.0 * t / 1000.0
            lag_adaptive = v - adaptive.filter(v, t)
            lag_fixed = v - fixed.filter(v, t)
        assert lag_adaptive < 40.0          # measured ~14 px
        assert lag_fixed > 3 * lag_adaptive  # measured ~159 px

    def test_measured_dt_not_fixed_frame_period(self):
        # Same value step, wildly different dt: a fixed-1/30 implementation
        # would produce identical outputs; measured dt must not.
        small_dt = OneEuro(1.0, 0.007)
        small_dt.filter(0.0, 0.0)
        out_1ms = small_dt.filter(10.0, 1.0)

        large_dt = OneEuro(1.0, 0.007)
        large_dt.filter(0.0, 0.0)
        out_1s = large_dt.filter(10.0, 1000.0)

        assert out_1ms < 0.5   # 1 ms: barely moves
        assert out_1s > 8.0    # 1 s: nearly converged

    def test_irregular_timestamps_track_ramp(self):
        # 500 px/s ramp sampled at jittery intervals: still tracks tightly.
        f = OneEuro(1.0, 0.007)
        t = 0.0
        lag = math.inf
        f.filter(0.0, t)
        for dt in itertools.islice(itertools.cycle([10.0, 45.0, 33.0, 70.0, 20.0]), 60):
            t += dt
            v = 500.0 * t / 1000.0
            lag = v - f.filter(v, t)
        assert abs(lag) < 60.0  # measured ~12 px

    def test_non_increasing_timestamp_returns_previous(self):
        f = OneEuro(1.0, 0.007)
        f.filter(1.0, 0.0)
        settled = f.filter(2.0, FRAME_MS)
        assert f.filter(999.0, FRAME_MS) == settled       # duplicate ts
        assert f.filter(999.0, FRAME_MS - 5.0) == settled  # backwards ts
        # state untouched: next valid sample behaves normally
        assert f.filter(2.0, 2 * FRAME_MS) != settled or settled == 2.0

    def test_reset_reprimes(self):
        f = OneEuro(1.0, 0.007)
        f.filter(0.0, 0.0)
        f.filter(100.0, FRAME_MS)
        f.reset()
        assert f.filter(42.0, 5000.0) == 42.0

    def test_set_mincutoff(self):
        f = OneEuro(1.0, 0.007)
        f.set_mincutoff(0.5)
        assert f.mincutoff == 0.5
        with pytest.raises(ValueError):
            f.set_mincutoff(0.0)

    def test_rejects_non_positive_cutoffs(self):
        with pytest.raises(ValueError):
            OneEuro(0.0, 0.007)
        with pytest.raises(ValueError):
            OneEuro(1.0, 0.007, dcutoff=-1.0)


# --------------------------------------------------------- CursorPipeline --

class TestBoxMapping:
    # First sample primes the One Euro exactly, so a single frame per case
    # yields the exact mapped point (quantized: speed is 0 at rest).
    @pytest.mark.parametrize(
        "anchor,expected",
        [
            ((128.0, 216.0), (0.0, 0.0)),          # box top-left
            ((512.0, 384.0), (1919.0, 1079.0)),    # box bottom-right, clamped
            ((320.0, 300.0), (960.0, 540.0)),      # box center -> screen center
            ((0.0, 0.0), (0.0, 0.0)),              # outside box: clamped to box
            ((640.0, 480.0), (1919.0, 1079.0)),    # outside box, far corner
            ((128.0, 384.0), (0.0, 1079.0)),       # bottom-left
            ((512.0, 216.0), (1919.0, 0.0)),       # top-right
        ],
    )
    def test_corner(self, anchor, expected):
        pipe = make_pipeline()
        s = pipe.update(make_frame(0.0, *anchor))
        assert (s.x, s.y) == expected
        assert s.frozen is False
        assert s.ts_ms == 0.0


class TestRebase:
    def test_rebase_moves_output_to_target(self):
        pipe = make_pipeline()
        t = 0.0
        for k in range(10):
            t = k * FRAME_MS
            s = pipe.update(make_frame(t, 320.0, 300.0))
        assert (s.x, s.y) == (960.0, 540.0)
        pipe.rebase(1200.0, 700.0)
        s = pipe.update(make_frame(t + FRAME_MS, 320.0, 300.0))
        # anchor unchanged -> mapped position unchanged -> output == target
        assert (s.x, s.y) == (1200.0, 700.0)

    def test_rebase_is_continuous_under_motion(self):
        pipe = make_pipeline()
        t = 0.0
        for k in range(10):
            t = k * FRAME_MS
            s = pipe.update(make_frame(t, 320.0, 300.0))
        pipe.rebase(400.0, 800.0)
        # keep moving gently after the rebase: no jump, just small steps
        prev = (400.0, 800.0)
        for k in range(1, 6):
            s = pipe.update(make_frame(t + k * FRAME_MS, 320.0 + k, 300.0))
            step = math.hypot(s.x - prev[0], s.y - prev[1])
            assert step < 15.0  # 1 camera px/frame * 5x gain, filtered
            prev = (s.x, s.y)

    def test_rebase_before_first_frame_is_applied_at_first_frame(self):
        pipe = make_pipeline()
        pipe.rebase(500.0, 500.0)
        s = pipe.update(make_frame(0.0, 200.0, 250.0))
        assert (s.x, s.y) == (500.0, 500.0)

    def test_rebase_target_clamped_to_screen(self):
        pipe = make_pipeline()
        pipe.update(make_frame(0.0, 320.0, 300.0))
        pipe.rebase(99999.0, -50.0)
        s = pipe.update(make_frame(FRAME_MS, 320.0, 300.0))
        assert (s.x, s.y) == (SCREEN_W - 1.0, 0.0)


class TestFrozen:
    def test_frozen_holds_output_and_rebase_on_unfreeze_is_continuous(self):
        pipe = make_pipeline()
        t = 0.0
        for k in range(10):
            t = k * FRAME_MS
            held = pipe.update(make_frame(t, 320.0, 300.0))
        pipe.set_frozen(True)

        # Hand travels far during the freeze, then settles (PALM-exit shape);
        # output must not move at any point.
        ax = 320.0
        for k in range(1, 22):
            t += FRAME_MS
            if k <= 15:
                ax += 10.0  # 150 camera px total = 750 screen px
            s = pipe.update(make_frame(t, ax, 300.0))
            assert s.frozen is True
            assert (s.x, s.y) == (held.x, held.y)

        # Internal filters kept updating: unfreeze + rebase to the held spot
        # must produce continuity despite the accumulated divergence.
        pipe.set_frozen(False)
        pipe.rebase(held.x, held.y)
        t += FRAME_MS
        s = pipe.update(make_frame(t, ax + 1.0, 300.0))
        assert s.frozen is False
        assert math.hypot(s.x - held.x, s.y - held.y) < 15.0

    def test_unfreeze_without_rebase_jumps(self):
        # Sanity check of the mechanism: internal filters really did keep
        # tracking during the freeze (this is why the engine must rebase).
        pipe = make_pipeline()
        t = 0.0
        for k in range(10):
            t = k * FRAME_MS
            held = pipe.update(make_frame(t, 320.0, 300.0))
        pipe.set_frozen(True)
        for k in range(1, 31):
            t += FRAME_MS
            pipe.update(make_frame(t, 320.0 + k * 5.0, 300.0))
        pipe.set_frozen(False)
        t += FRAME_MS
        s = pipe.update(make_frame(t, 470.0, 300.0))
        assert math.hypot(s.x - held.x, s.y - held.y) > 300.0


class TestNoHand:
    def test_no_hand_holds_last_position_with_zero_speed(self):
        pipe = make_pipeline()
        t = 0.0
        for k in range(20):
            t = k * FRAME_MS
            last = pipe.update(make_frame(t, 200.0 + k * 4.0, 300.0))
        assert last.speed_px_s > 0.0
        for k in range(1, 4):
            s = pipe.update(make_frame(t + k * FRAME_MS, present=False))
            assert (s.x, s.y) == (last.x, last.y)
            assert s.speed_px_s == 0.0
            assert s.ts_ms == t + k * FRAME_MS

    def test_no_hand_before_any_hand_parks_at_screen_center(self):
        pipe = make_pipeline()
        s = pipe.update(make_frame(0.0, present=False))
        assert (s.x, s.y) == (SCREEN_W / 2.0, SCREEN_H / 2.0)
        assert s.speed_px_s == 0.0


class TestSpeedAndQuantization:
    def test_speed_estimate_tracks_constant_velocity(self):
        # 60 camera px/s through a 5x x-gain box = 300 screen px/s.
        pipe = make_pipeline()
        s = None
        for k in range(46):  # 1.5 s
            t = k * FRAME_MS
            s = pipe.update(make_frame(t, 130.0 + 60.0 * t / 1000.0, 300.0))
        assert s.speed_px_s == pytest.approx(300.0, rel=0.2)

    def test_quantization_at_rest(self):
        # Sub-pixel camera jitter at rest: outputs must be whole pixels and
        # settle to a single position (no shimmer).
        pipe = make_pipeline()
        outs = []
        for k in range(61):  # 2 s
            t = k * FRAME_MS
            j = 0.4 * math.sin(2 * math.pi * 7.0 * t / 1000.0)
            s = pipe.update(make_frame(t, 320.0 + j, 300.0 + j))
            outs.append(s)
        for s in outs:
            assert s.x == round(s.x) and s.y == round(s.y)
            assert s.speed_px_s < QUANTIZE_MAX_SPEED_PX_S
        assert len({(s.x, s.y) for s in outs[-30:]}) == 1

    def test_fast_motion_not_quantized(self):
        # Sweep the box in ~1 s (~1920 px/s on screen): once the speed
        # estimate settles, outputs keep sub-pixel precision.
        pipe = make_pipeline()
        samples = []
        for k in range(30):
            t = k * FRAME_MS
            samples.append(pipe.update(make_frame(t, 128.0 + 384.0 * t / 1000.0, 300.0)))
        tail = samples[15:]
        assert all(s.speed_px_s > QUANTIZE_MAX_SPEED_PX_S for s in tail)
        assert any(s.x != round(s.x) for s in tail)


class TestDragAndPinch:
    def test_set_drag_switches_mincutoff(self):
        cfg = Config()
        pipe = CursorPipeline(cfg, SCREEN_W, SCREEN_H)
        assert pipe._fx.mincutoff == cfg.one_euro.mincutoff
        pipe.set_drag(True)
        assert pipe._fx.mincutoff == cfg.one_euro.drag_mincutoff
        assert pipe._fy.mincutoff == cfg.one_euro.drag_mincutoff
        pipe.set_drag(False)
        assert pipe._fx.mincutoff == cfg.one_euro.mincutoff
        assert pipe._fy.mincutoff == cfg.one_euro.mincutoff

    def test_pinch_filters_are_lazy_per_name_and_use_pinch_mincutoff(self):
        cfg = Config()
        pipe = CursorPipeline(cfg, SCREEN_W, SCREEN_H)
        assert pipe.pinch("left", 1.0, 0.0) == 1.0  # first sample primes
        second = pipe.pinch("left", 0.0, FRAME_MS)
        assert 0.0 < second < 1.0
        # a different name gets its own fresh filter
        assert pipe.pinch("right", 0.5, 2 * FRAME_MS) == 0.5
        assert pipe._pinch_filters["left"].mincutoff == cfg.one_euro.pinch_mincutoff
        assert pipe._pinch_filters["right"].mincutoff == cfg.one_euro.pinch_mincutoff

    def test_pinch_smoother_than_raw_jitter(self):
        pipe = make_pipeline()
        outs = []
        for k in range(61):
            t = k * FRAME_MS
            raw = 0.4 + 0.1 * math.sin(2 * math.pi * 8.0 * t / 1000.0)
            outs.append(pipe.pinch("left", raw, t))
        tail = outs[-20:]
        assert max(tail) - min(tail) < 0.1  # raw ptp is 0.2


class TestReset:
    def test_reset_clears_state(self):
        cfg = Config()
        pipe = CursorPipeline(cfg, SCREEN_W, SCREEN_H)
        for k in range(10):
            pipe.update(make_frame(k * FRAME_MS, 200.0 + k * 5.0, 300.0))
        pipe.rebase(100.0, 100.0)
        pipe.set_frozen(True)
        pipe.set_drag(True)
        pipe.pinch("left", 0.5, 0.0)
        pipe.reset()

        assert pipe._fx.mincutoff == cfg.one_euro.mincutoff
        assert pipe._pinch_filters == {}
        # frozen flag cleared, offset/filters reprimed: exact mapping again
        s = pipe.update(make_frame(10_000.0, 320.0, 300.0))
        assert (s.x, s.y) == (960.0, 540.0)
        assert s.frozen is False
        assert s.speed_px_s == 0.0
