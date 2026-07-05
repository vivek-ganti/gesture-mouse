"""Tests for gesture_mouse.tracker's camera-selection logic (no real camera
needed — cv2.VideoCapture and list_cameras are mocked).
"""
from __future__ import annotations

from unittest.mock import patch

from gesture_mouse.config import Config
from gesture_mouse.tracker import CameraTracker


def test_candidate_order_immune_to_unstable_enumeration_between_calls():
    """Two list_cameras() calls microseconds apart can return the same
    devices in a different order (confirmed in the wild on real hardware).
    The actual bug this guards: resolving the preferred index via a
    SEPARATE list_cameras() call than the one used for the printed/
    persisted label let a device open correctly while being labeled with
    the OTHER camera's name. _candidate_order() must call list_cameras()
    exactly once and use that single list for both the index and the label.
    """
    cfg = Config()
    cfg.camera.name = "FaceTime HD Camera"
    tracker = CameraTracker(cfg)

    call_count = 0

    def flaky_list_cameras():
        nonlocal call_count
        call_count += 1
        # Order flips between calls -- exactly the observed race: the same
        # two devices, reported in the opposite order the second time.
        if call_count == 1:
            return ["FaceTime HD Camera", "Iriun Camera"]
        return ["Iriun Camera", "FaceTime HD Camera"]

    with patch("gesture_mouse.tracker.list_cameras", side_effect=flaky_list_cameras):
        order, cams = tracker._candidate_order()

    assert call_count == 1, "list_cameras() must be called exactly once"
    preferred_idx = order[0]
    assert cams[preferred_idx] == "FaceTime HD Camera"


def test_candidate_order_prefers_builtin_camera_when_no_name_configured():
    cfg = Config()
    cfg.camera.name = ""
    tracker = CameraTracker(cfg)

    with patch(
        "gesture_mouse.tracker.list_cameras",
        return_value=["Iriun Camera", "FaceTime HD Camera"],
    ):
        order, cams = tracker._candidate_order()

    assert cams[order[0]] == "FaceTime HD Camera"


def test_candidate_order_falls_back_to_probing_all_when_name_not_found():
    cfg = Config()
    cfg.camera.name = "Nonexistent Camera"
    tracker = CameraTracker(cfg)

    with patch(
        "gesture_mouse.tracker.list_cameras",
        return_value=["FaceTime HD Camera", "Iriun Camera"],
    ):
        order, cams = tracker._candidate_order()

    # No crash, no incorrect preference; every index still gets tried.
    assert sorted(order) == [0, 1]
