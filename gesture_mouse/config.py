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
    """PALM mode: open-hand system gestures (trackpad parity)."""

    enter_ms: float = 80.0
    swipe_min_vel_fw_s: float = 1.0      # frame-widths per second
    swipe_min_disp_frac: float = 0.22    # of frame width (or height for vertical)
    swipe_window_ms: float = 350.0
    swipe_refractory_ms: float = 600.0
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
