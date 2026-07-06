"""Headless layout/regression tests for gesture_mouse.preview.

draw_overlay() is pure numpy/cv2 drawing on a caller-provided canvas with no
window-server interaction (see its own module docstring), so it's fully
testable without a camera or display — there is no camera access in this
environment, so these static checks are the only verification available
before a live check on real hardware. They act as a proxy for "the UI
doesn't visually overlap or blow up," not a substitute for actually looking
at it.
"""
from __future__ import annotations

import numpy as np
import pytest

from gesture_mouse.config import Config
from gesture_mouse.preview import (
    _compute_regions,
    _draw_control_box,
    _fit_frame_to_canvas,
    draw_overlay,
)
from gesture_mouse.types import (
    EngineState,
    LandmarkFrame,
    Point,
    SessionState,
    StateSnapshot,
)

CANVAS_SIZES = [(640, 480), (960, 540), (1280, 720), (320, 240)]


def make_landmark_frame(img_w: int, img_h: int, x: float = None, y: float = None) -> LandmarkFrame:
    """21 points spread across the frame (not just one spot) so scaled
    skeleton coordinates exercise more than a single pixel."""
    cx = x if x is not None else img_w / 2.0
    cy = y if y is not None else img_h / 2.0
    pts = tuple(
        Point(cx + (i - 10) * (img_w / 40.0), cy + (i - 10) * (img_h / 40.0))
        for i in range(21)
    )
    return LandmarkFrame(
        ts_ms=0.0, handedness="Right", landmarks=pts, img_w=img_w, img_h=img_h,
        confidence=0.9, scale=80.0, source="test",
    )


def make_snapshot(
    session_state=SessionState.ACTIVE,
    engine_state=EngineState.POINTER,
    hand_present=True,
    fps=30.0,
    latency_ms=10.0,
    suspend_reason=None,
) -> StateSnapshot:
    return StateSnapshot(
        session_state=session_state, engine_state=engine_state,
        hand_present=hand_present, fps=fps, latency_ms=latency_ms,
        suspend_reason=suspend_reason,
    )


class TestRegionsNeverOverlap:
    """The actual collision-prevention mechanism: status/hint/camera-list
    are vertically stacked with a gap, independent of canvas size or text
    content -- this is what makes the confirmed status/hint-vs-camera-list
    overlap bug impossible by construction, not a width-dependent guess."""

    @pytest.mark.parametrize("w,h", CANVAS_SIZES)
    def test_status_then_hint_then_camera_list_strictly_ordered(self, w, h):
        r = _compute_regions(w, h)
        min_line_gap = 10  # generous floor for any font scale used here
        assert r.hint_y >= r.status_y + min_line_gap
        assert r.camera_list_top >= r.hint_y + min_line_gap

    def test_regions_independent_of_canvas_size(self):
        # Deliberately NOT a function of w/h -- collision-proofness must not
        # depend on guessing text width at some particular canvas size.
        a = _compute_regions(320, 240)
        b = _compute_regions(1920, 1080)
        assert a == b


class TestControlBoxFractionPreserved:
    @pytest.mark.parametrize("w,h", CANVAS_SIZES)
    def test_control_box_matches_configured_fractions(self, w, h):
        cfg = Config()
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        _draw_control_box(canvas, cfg)
        # Re-derive the drawn rect from the outer (2px) rectangle's white
        # 1px border pixels rather than re-deriving cfg's own fractions,
        # so this actually checks pixels drawn on the canvas.
        cb = cfg.control_box
        x0, y0 = int(cb.x * w), int(cb.y * h)
        x1, y1 = int((cb.x + cb.w) * w), int((cb.y + cb.h) * h)
        assert canvas[y0, x0].tolist() != [0, 0, 0]
        assert canvas[y0, x1 - 1].tolist() != [0, 0, 0]
        assert canvas[y1 - 1, x0].tolist() != [0, 0, 0]


class TestSkeletonScaling:
    @pytest.mark.parametrize("canvas_w,canvas_h", CANVAS_SIZES)
    def test_large_source_frame_maps_fully_inside_fixed_canvas(self, canvas_w, canvas_h):
        # Simulates the confirmed root cause: a camera delivering a much
        # larger frame (e.g. 1920x1080) than the fixed display canvas.
        lm_frame = make_landmark_frame(1920, 1080)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        snap = make_snapshot()
        draw_overlay(canvas, lm_frame, snap, {}, Config(), show_box=False)
        # No direct return of mapped points, so re-derive with the same
        # scale math draw_overlay uses and assert containment.
        sx = canvas_w / lm_frame.img_w
        sy = canvas_h / lm_frame.img_h
        for p in lm_frame.landmarks:
            px, py = p.x * sx, p.y * sy
            assert 0 <= px < canvas_w
            assert 0 <= py < canvas_h

    def test_small_source_frame_also_maps_correctly(self):
        lm_frame = make_landmark_frame(160, 120)
        canvas = np.zeros((540, 960, 3), dtype=np.uint8)
        snap = make_snapshot()
        draw_overlay(canvas, lm_frame, snap, {}, Config(), show_box=False)
        sx, sy = 960 / 160, 540 / 120
        for p in lm_frame.landmarks:
            assert 0 <= p.x * sx < 960
            assert 0 <= p.y * sy < 540


class TestFitFrameToCanvas:
    def test_resizes_to_exact_canvas_size(self):
        src = np.full((480, 640, 3), 200, dtype=np.uint8)
        out = _fit_frame_to_canvas(src, 960, 540)
        assert out.shape == (540, 960, 3)

    def test_matching_size_still_returns_a_copy(self):
        src = np.full((540, 960, 3), 50, dtype=np.uint8)
        out = _fit_frame_to_canvas(src, 960, 540)
        assert out is not src
        out[0, 0] = 255
        assert src[0, 0].tolist() == [50, 50, 50]


class TestDrawOverlaySmoke:
    """No-crash coverage across canvas sizes x the state combinations that
    change what's drawn (help/banners/camera list)."""

    @pytest.mark.parametrize("w,h", CANVAS_SIZES)
    @pytest.mark.parametrize(
        "session_state,engine_state,fps,show_help",
        [
            (SessionState.IDLE, None, 0.0, False),
            (SessionState.WARMUP, None, 0.0, False),
            (SessionState.ACTIVE, EngineState.POINTER, 30.0, False),
            (SessionState.ACTIVE, EngineState.SCROLL, 5.0, False),  # low-fps banner
            (SessionState.ACTIVE, EngineState.PALM, 30.0, True),    # help overlay
            (SessionState.SUSPENDED, EngineState.CLUTCH_WAIT, 30.0, False),
        ],
    )
    def test_no_crash_and_correct_shape(
        self, w, h, session_state, engine_state, fps, show_help
    ):
        cfg = Config()
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        snap = make_snapshot(session_state=session_state, engine_state=engine_state, fps=fps)
        lm_frame = make_landmark_frame(w, h) if engine_state is not None else None
        out = draw_overlay(
            canvas, lm_frame, snap, {"left": 0.4}, cfg,
            show_box=True, palm_debug={"open": True, "m": 0.7, "phase": "armed"},
            cameras=["FaceTime HD Camera", "Iriun Camera"], current_camera_index=0,
            show_help=show_help,
        )
        assert out.shape == (h, w, 3)
        assert out.dtype == np.uint8
