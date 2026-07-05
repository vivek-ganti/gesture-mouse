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


def test_trigger_commands_cover_every_system_gesture():
    """Every TRIGGER intent name in the vocabulary maps to a runnable argv
    (System Events keystrokes for Spaces/tabs — raw CGEvent chords provably
    don't trigger macOS's Spaces switcher — and the Mission Control binary
    for MC/expose/desktop)."""
    from gesture_mouse.synth import trigger_command

    for name in ("space_prev", "space_next", "mission_control", "app_expose",
                 "show_desktop", "launchpad", "tab_next", "tab_prev"):
        argv = trigger_command(name)
        assert argv, f"no command for {name}"
        assert argv[0] in ("osascript", "open",
                           "/System/Applications/Mission Control.app"
                           "/Contents/MacOS/Mission Control")
    assert trigger_command("space_prev")[-1].endswith("key code 123 using {control down}")
    assert trigger_command("tab_prev")[-1].endswith(
        "key code 48 using {control down, shift down}")
    assert trigger_command("nonsense") is None


def test_spaces_module_imports_and_reads_topology():
    """spaces.py smoke: imports, SkyLight bridges load, and the topology
    reader returns either None or a (spaces, index) pair with the active
    space present. Never switches anything."""
    from gesture_mouse import spaces

    found = spaces._display_spaces()
    if found is not None:
        space_list, idx = found
        assert 0 <= idx < len(space_list)
        assert "ManagedSpaceID" in space_list[idx]


def test_bgra_to_bgr_handles_row_padding():
    """CoreVideo pads rows to bytes_per_row >= width*4; the converter must
    slice padding off and drop alpha without channel reordering."""
    import numpy as np
    from gesture_mouse.tracker import bgra_to_bgr

    w, h, bpr = 3, 2, 20          # 3*4=12 used, 8 pad bytes per row
    rows = []
    val = 0
    for _ in range(h):
        row = []
        for _ in range(w):
            row += [val, val + 1, val + 2, 255]   # B,G,R,A
            val += 10
        row += [0xEE] * (bpr - w * 4)             # padding garbage
        rows += row
    out = bgra_to_bgr(bytes(rows), w, h, bpr)
    assert out.shape == (h, w, 3)
    assert out[0, 0].tolist() == [0, 1, 2]        # B,G,R preserved in order
    assert out[1, 2].tolist() == [50, 51, 52]
    assert 0xEE not in out                        # padding never leaks in


def test_custom_action_argv_mapping():
    """Custom-gesture actions map to runnable argvs; invalid ones -> None."""
    from gesture_mouse.synth import custom_action_argv

    # Bare modifier tap (the Wispr Flow dictation case).
    argv = custom_action_argv({"type": "key", "key": "option"})
    assert argv[0] == "osascript" and argv[-1].endswith("key code 58")
    # Chord with modifiers (aliases normalized).
    argv = custom_action_argv({"type": "key", "key": "d", "modifiers": ["cmd"]})
    assert argv[-1].endswith("key code 2 using {command down}")
    # Shell.
    assert custom_action_argv(
        {"type": "shell", "argv": ["open", "-a", "Snaply"]}
    ) == ["open", "-a", "Snaply"]
    # Reuse of a system trigger.
    assert custom_action_argv({"type": "trigger", "name": "mission_control"})
    # Invalids.
    assert custom_action_argv({"type": "key", "key": "nosuchkey"}) is None
    assert custom_action_argv({"type": "key", "key": "a",
                               "modifiers": ["hyper"]}) is None
    assert custom_action_argv({"type": "shell", "argv": []}) is None
    assert custom_action_argv({"type": "wat"}) is None
