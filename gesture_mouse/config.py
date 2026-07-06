"""Config: dataclass defaults, config.json load/save, mtime hot-reload.

Every tunable threshold in the system lives here. All temporal keys are
milliseconds. Unknown keys in config.json are ignored; missing keys fall back
to defaults, so old configs survive upgrades.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CameraConfig:
    name: str = ""              # empty = first available; set via --camera / config
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True
    rotate: int = 0             # 0 / 90 / 180 / 270, applied before mirror
    orientation: str = "front"  # "front" | "down" (informational; picks docs/defaults)
    # "avf" = native AVFoundation capture, device selected by NAME->uniqueID
    # (label and video can never disagree). "opencv" = cv2.VideoCapture by
    # index — kept as fallback; its index table cannot be reliably correlated
    # with device names (order flips between enumerations on real hardware).
    backend: str = "avf"


@dataclass
class ControlBox:
    """Sub-rectangle of the camera frame mapped to the full screen (fractions).
    Bottom edge kept <= 0.80 so aiming at the Dock keeps the palm in frame."""

    x: float = 0.20
    y: float = 0.45
    w: float = 0.60
    h: float = 0.35


@dataclass
class OneEuroConfig:
    mincutoff: float = 1.0
    beta: float = 0.007
    dcutoff: float = 1.0
    drag_mincutoff: float = 0.5   # while dragging
    pinch_mincutoff: float = 3.0  # pinch-distance filter


@dataclass
class PinchConfig:
    left_engage: float = 0.35
    left_release: float = 0.48
    drag_release: float = 0.60    # in-drag release threshold (must clearly open hand)
    right_engage: float = 0.38
    right_release: float = 0.52
    engage_ms: float = 100.0
    release_ms: float = 66.0
    latch_lookback_ms: float = 100.0
    latch_max_speed_px_s: float = 50.0
    scale_stability_pct_per_100ms: float = 15.0
    arbitration_ratio: float = 1.3   # loser pinch must exceed ratio x winner


@dataclass
class ClickConfig:
    double_ms: float = 500.0
    double_max_px: float = 15.0
    double_reengage: float = 0.42
    double_release: float = 0.44
    drag_unfreeze_px: float = 12.0
    drag_axis_deadband_pct: float = 20.0
    right_tap_max_ms: float = 400.0


@dataclass
class ScrollConfig:
    gain: float = 350.0
    deadzone_frac: float = 0.03
    max_px_s: float = 900.0
    invert: bool = False
    enter_ms: float = 100.0
    exit_ms: float = 130.0
    together_max: float = 0.45          # dist(8,12)/scale for "fingers together"
    entry_max_speed_px_s: float = 60.0
    # Horizontal joystick in the same pose = tab switch (Ctrl+Tab / +Shift).
    tab_deadzone_frac: float = 0.08     # of frame width, from the latched origin
    tab_refractory_ms: float = 500.0


@dataclass
class ClutchConfig:
    engage_ms: float = 150.0
    reacquire_ms: float = 250.0   # from HANDS_LOST / SUSPENDED


@dataclass
class PalmConfig:
    """PALM mode: open-hand system gestures (trackpad parity).

    Swipe model (arm-at-rest -> net displacement; see palm.py): hold an open
    palm STILL for arm_hold_ms to arm, then flick — the motion itself needs
    no pose (motion blur destroys finger landmarks mid-swipe) and tolerates
    the hand disappearing entirely for up to swipe_gap_bridge_ms (normal
    MediaPipe behavior under fast motion)."""

    enter_ms: float = 80.0               # debounced open pose -> PALM (cursor freeze)
    arm_hold_ms: float = 150.0           # open palm at rest this long to arm
    arm_max_speed_fw_s: float = 0.30     # palm-center speed ceiling for "at rest"
    swipe_min_disp_frac: float = 0.25    # net travel, fraction of frame span
    swipe_max_duration_ms: float = 800.0  # fire window after (last) arming origin
    swipe_axis_dominance: float = 1.5    # major axis >= this x minor (fraction space)
    swipe_cooldown_ms: float = 1000.0    # after a fire; full re-arm required
    swipe_gap_bridge_ms: float = 350.0   # hand-absent gap tolerated while armed
    spread_open: float = 0.9             # m above this = open palm start
    spread_closed: float = 0.55          # pinch-in fires when m falls below
    spread_in_start: float = 0.6         # spread-out must start below this
    spread_out: float = 1.0              # spread-out fires when m rises above
    spread_window_ms: float = 500.0
    spread_refractory_ms: float = 800.0
    # Max cursor speed (px/s) for pinch-in/spread-out to be forwarded while
    # still in POINTER (i.e. before/without ever entering PALM — spread-out
    # in particular starts from a fist, so it often completes before the
    # open-palm pose has been held long enough to enter PALM). Deliberately
    # far more generous than scroll.entry_max_speed_px_s: a hand naturally
    # drifts a bit while flexing all five fingers, and this isn't a scroll-
    # style continuous gesture that needs a still start.
    forward_max_speed_px_s: float = 300.0


@dataclass
class TrackingConfig:
    """MediaPipe Hand Landmarker confidence thresholds.

    Presence/tracking are deliberately LOWER than MediaPipe's 0.5 defaults:
    fast lateral motion blurs the hand and the landmark model was trained
    with little blur augmentation, so at default thresholds the hand is
    declared absent for several frames in the middle of every fast swipe.
    Lower presence/tracking keeps the blurred-but-findable hand "present"
    (gross palm position stays usable long after fingertip precision dies,
    which is all swipe detection needs). Detection stays near default so
    re-acquisition after a genuine loss is still fast and non-ghosty."""

    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.4
    min_tracking_confidence: float = 0.3


@dataclass
class PreviewConfig:
    """Fixed internal canvas size for the preview window, independent of the
    camera's actual capture resolution (AVFoundation/OpenCV presets are not
    guaranteed -- cameras silently fall back to whatever they support).
    Every session state (IDLE/WARMUP/SUSPENDED/ACTIVE) renders into this
    same size, so the window never resizes/jumps, and every fixed-pixel UI
    constant in preview.py is tuned against it. The real camera frame is
    stretch-resized into this canvas each draw."""

    canvas_w: int = 960
    canvas_h: int = 540


@dataclass
class PoseConfig:
    """Static pose classification: per-finger extension/curl via the
    interior angle at the PIP joint (angle between the MCP->PIP and
    PIP->TIP vectors; 180 = perfectly straight, smaller = more curled).
    Two thresholds give a real hysteresis band instead of one hard cutoff
    -- mirrors the engage/release pattern already used for pinch detection.
    Live-tunable at runtime (see preview key bindings) since the right
    values depend on individual hand geometry and camera angle.

    smoothing_* configure a One Euro filter applied to landmark coordinates
    BEFORE pose classification runs (separate from CursorPipeline's own
    anchor filter) -- a higher mincutoff than the cursor filter is fine
    since pose tests only need a stable boolean, not px-accurate tracking."""

    extend_angle_deg: float = 160.0
    curl_angle_deg: float = 130.0
    smoothing_mincutoff: float = 1.5
    smoothing_beta: float = 0.02
    smoothing_dcutoff: float = 1.0


@dataclass
class SuspendConfig:
    mouse_divergence_px: float = 8.0
    keyboard_mute_ms: float = 1500.0


@dataclass
class HotkeyConfig:
    toggle: str = "ctrl+alt+g"
    panic: str = "ctrl+alt+escape"


@dataclass
class OptionsConfig:
    dwell_right_click: bool = False  # v1.5 accessibility path; not yet wired
    privacy_preview: bool = True     # skeleton-on-black, no camera image
    inertial_scroll: bool = False    # v1.5 flick-to-scroll; not yet wired
    extended_test: str = "radial"    # "radial" | "y" finger-extension test
    audio_ticks: bool = True         # toggle sound; click ticks stay opt-in
    click_ticks: bool = False
    debug_gestures: bool = False     # print swipe arming/candidate/reject events


# Custom gestures: hold a static pose (at a mostly-still hand) for hold_ms ->
# run an action. Editable/addable in config.json (hot-reloaded). Available
# poses: "horns" (index + pinky extended, middle + ring curled — the rock
# sign; thumb ignored). Action types:
#   {"type": "key", "key": "option"}                     tap a key (incl. bare
#       modifiers: option/command/shift/control — e.g. Wispr Flow's dictation
#       toggle), optional "modifiers": ["command", ...] for chords
#   {"type": "shell", "argv": ["open", "-a", "Snaply"]}  run a command
#   {"type": "trigger", "name": "mission_control"}       reuse a system action
DEFAULT_CUSTOM_GESTURES: list[dict] = [
    {
        "name": "dictate",
        "pose": "horns",
        "hold_ms": 300.0,
        "cooldown_ms": 1200.0,
        "action": {"type": "key", "key": "option"},
    },
]

# Default bindings match macOS trackpad semantics with a mirrored (selfie)
# camera: hand moving left on screen = fingers-left on a trackpad = next Space.
DEFAULT_BINDINGS: dict[str, str] = {
    "swipe_left": "space_next",
    "swipe_right": "space_prev",
    "swipe_up": "mission_control",
    "swipe_down": "app_expose",
    "pinch_in": "launchpad",
    "spread_out": "show_desktop",
}


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand: str = "auto"                     # "auto" | "right" | "left" — "auto"
                                           # tracks either hand; labels depend
                                           # on the camera mirror convention
    control_box: ControlBox = field(default_factory=ControlBox)
    one_euro: OneEuroConfig = field(default_factory=OneEuroConfig)
    pinch: PinchConfig = field(default_factory=PinchConfig)
    click: ClickConfig = field(default_factory=ClickConfig)
    scroll: ScrollConfig = field(default_factory=ScrollConfig)
    clutch: ClutchConfig = field(default_factory=ClutchConfig)
    palm: PalmConfig = field(default_factory=PalmConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    suspend: SuspendConfig = field(default_factory=SuspendConfig)
    hands_lost_ms: float = 265.0
    # Pose holds (clutch engage, scroll entry, PALM entry/exit) tolerate up to
    # this much continuous non-detection before resetting the hold — absorbs
    # single/double-frame hand-tracking jitter, which is far more likely when
    # several landmarks must agree at once (e.g. all four non-thumb fingers
    # extended for a palm swipe) than for a single-landmark pose.
    pose_jitter_grace_ms: float = 120.0
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
    bindings: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BINDINGS))
    custom_gestures: list = field(
        default_factory=lambda: [dict(g) for g in DEFAULT_CUSTOM_GESTURES]
    )


def _merge(dc: Any, data: dict[str, Any]) -> Any:
    """Overlay a dict onto a dataclass instance, recursing into nested ones."""
    for f in dataclasses.fields(dc):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(dc, f.name)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, dict) and isinstance(value, dict):
            current.update(value)
        else:
            setattr(dc, f.name, value)
    return dc


def _assign_in_place(dst: Any, src: Any) -> None:
    """Copy every field value of ``src`` into ``dst``, recursing into nested
    dataclasses and dicts so that ``dst`` and all of its children KEEP their
    object identity. Modules wire themselves to ``store.config`` (and to
    nested pieces like ``cfg.palm`` / ``cfg.suspend`` / ``cfg.bindings``) at
    startup; a hot-reload that swapped in fresh objects would silently
    disconnect them all."""
    for f in dataclasses.fields(dst):
        s_val = getattr(src, f.name)
        d_val = getattr(dst, f.name)
        if dataclasses.is_dataclass(d_val) and dataclasses.is_dataclass(s_val):
            _assign_in_place(d_val, s_val)
        elif isinstance(d_val, dict) and isinstance(s_val, dict):
            d_val.clear()
            d_val.update(s_val)
        else:
            setattr(dst, f.name, s_val)


class ConfigStore:
    """Loads config.json beside the project (or a given path); hot-reloads on
    mtime change. `store.config` is always the current Config."""

    def __init__(self, path: str | os.PathLike[str] = "config.json") -> None:
        self.path = Path(path)
        self._mtime: float | None = None
        self.config = Config()
        self.reload_if_changed(force=True)

    def reload_if_changed(self, force: bool = False) -> bool:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return False
        if not force and mtime == self._mtime:
            return False
        self._mtime = mtime
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return False  # bad half-written file: keep the current config
        # Build defaults+file, then copy IN PLACE so `store.config` (and its
        # nested dataclasses/dicts) keep their identity across hot-reloads.
        _assign_in_place(self.config, _merge(Config(), data))
        return True

    def save(self) -> None:
        self.path.write_text(json.dumps(dataclasses.asdict(self.config), indent=2) + "\n")
        self._mtime = self.path.stat().st_mtime
