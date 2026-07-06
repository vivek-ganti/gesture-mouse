"""OpenCV tuning window: skeleton, control box, pinch meters, status banners.

Privacy mode (``cfg.options.privacy_preview``, default ON) draws the skeleton
on a black canvas — the camera image never reaches the screen. The camera
image is only shown when privacy is explicitly disabled AND a BGR frame is
available (``bgr_frame`` is None in replay).

The preview renders onto a FIXED internal canvas (``cfg.preview.canvas_w``
x ``canvas_h``) in every session state, never the camera's actual capture
size. AVFoundation/OpenCV capture presets are not guaranteed — cameras
silently fall back to whatever they support (plan doc: UI scaling/overlap
fix) — so the real frame can be a different size than expected, in either
direction. Every fixed-pixel UI constant in this module (meters, fonts, the
help panel) is tuned against that one fixed size; the real camera frame is
stretch-resized into it every draw via :func:`_fit_frame_to_canvas`, and
landmark points (in the ORIGINAL frame's pixel space) are rescaled to match
in :func:`_draw_skeleton`. Stretch, not letterbox: privacy mode (the
default) draws a synthetic skeleton on black, not a real photo, so minor
aspect distortion is low-cost, and stretch needs no padding-offset math
threaded through every draw call.

Layout collisions (status/hint text vs. the camera list) are prevented by
giving each element its own reserved, non-overlapping region
(:func:`_compute_regions`) instead of every draw function picking a fixed
y-offset independently.

Note: this deliberately does NOT attempt to correct ``cv2.imshow``'s known
Retina/HiDPI window-scaling behavior on macOS (renders in raw pixels, so a
2x-scale display shows the window ~2x larger than the image's pixel size
warrants) — that's an open, unfixed OpenCV limitation (upstream issue
#20403), not something fixable from here. A fixed, correctly-laid-out
internal canvas stays legible regardless of what scale the OS ultimately
draws it at; don't re-attempt a WINDOW_AUTOSIZE/DPI workaround.

All drawing is factored into :func:`draw_overlay` — pure numpy/cv2 primitive
calls on a caller-provided canvas, no window-server interaction — so tests run
headless. ``Preview.show()`` only adds canvas selection, ``imshow`` and
``waitKey``; the window itself is created lazily on the first ``show()``.
Total draw cost is a handful of cv2 primitives (well under 5 ms).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .types import INDEX_MCP, EngineState, LandmarkFrame, SessionState, StateSnapshot

WINDOW_NAME = "gesture-mouse"

# Standard MediaPipe 21-landmark hand skeleton (bone index pairs).
HAND_BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm edge
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
LOW_FPS_THRESHOLD: float = 20.0
# Normalized pinch distance rendered at full meter width; thresholds
# (engage/release <= 0.6) land comfortably inside the bar.
_METER_FULL_SCALE: float = 0.9

# BGR colors, matched to the indicator dot palette.
_WHITE = (255, 255, 255)
_GRAY = (140, 140, 140)
_DARK = (60, 60, 60)
_GREEN = (94, 214, 51)
_BLUE = (255, 140, 64)
_PURPLE = (245, 107, 184)
_ORANGE = (38, 158, 255)
_YELLOW = (51, 217, 255)
_RED = (59, 59, 242)

_ENGINE_COLORS: dict[EngineState, tuple[int, int, int]] = {
    EngineState.CLUTCH_WAIT: _WHITE,
    EngineState.POINTER: _GREEN,
    EngineState.PINCHED: _BLUE,
    EngineState.RIGHT_PINCH: _BLUE,
    EngineState.SCROLL: _PURPLE,
    EngineState.PALM: _ORANGE,
    EngineState.HANDS_LOST: _YELLOW,
}


def _state_color(snap: StateSnapshot | None) -> tuple[int, int, int]:
    if snap is None or snap.session_state is SessionState.IDLE:
        return _GRAY
    if snap.session_state is SessionState.WARMUP:
        return _WHITE
    if snap.session_state is SessionState.SUSPENDED:
        return _RED
    if snap.engine_state is None:
        return _WHITE
    return _ENGINE_COLORS.get(snap.engine_state, _WHITE)


def _put_text(
    canvas: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.45,
) -> None:
    # Drop shadow keeps text readable over the camera image.
    cv2.putText(canvas, text, (org[0] + 1, org[1] + 1), _FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, _FONT, scale, color, 1, cv2.LINE_AA)


def _draw_control_box(canvas: np.ndarray, cfg: Config) -> None:
    h, w = canvas.shape[:2]
    cb = cfg.control_box
    x0, y0 = int(cb.x * w), int(cb.y * h)
    x1, y1 = int((cb.x + cb.w) * w), int((cb.y + cb.h) * h)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), _DARK, 2)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), _WHITE, 1)


def _fit_frame_to_canvas(src: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    """Stretch-resize ``src`` (whatever size it actually is) to exactly
    ``canvas_w`` x ``canvas_h``. Always returns a fresh array — even when
    the sizes already match — so callers never draw on the caller's frame."""
    if src.shape[1] == canvas_w and src.shape[0] == canvas_h:
        return src.copy()
    return cv2.resize(src, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)


def _draw_skeleton(
    canvas: np.ndarray,
    lm_frame: LandmarkFrame,
    color: tuple[int, int, int],
    sx: float = 1.0,
    sy: float = 1.0,
) -> None:
    """``sx``/``sy`` rescale landmark points from the frame's ORIGINAL pixel
    space (``lm_frame.img_w/img_h``) into the fixed display canvas's space —
    they differ whenever the camera's actual capture size isn't the
    canvas's fixed size, which per the module docstring is the common case."""
    pts = [(int(p.x * sx), int(p.y * sy)) for p in lm_frame.landmarks]
    for a, b in HAND_BONES:
        cv2.line(canvas, pts[a], pts[b], color, 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(canvas, pt, 2, _WHITE, -1, cv2.LINE_AA)
    # Anchor (INDEX_MCP) highlighted — this point IS the cursor.
    cv2.circle(canvas, pts[INDEX_MCP], 6, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, pts[INDEX_MCP], 7, _WHITE, 1, cv2.LINE_AA)


def _draw_meter(
    canvas: np.ndarray,
    x: int,
    y: int,
    bar_w: int,
    bar_h: int,
    label: str,
    value: float | None,
    engage: float,
    release: float,
    full_scale: float = _METER_FULL_SCALE,
    invert: bool = False,
) -> None:
    def to_x(v: float) -> int:
        frac = min(max(v / full_scale, 0.0), 1.0)
        return x + int(frac * bar_w)

    cv2.rectangle(canvas, (x, y), (x + bar_w, y + bar_h), _GRAY, 1)
    if value is not None:
        armed = value >= engage if invert else value <= engage
        fill = _GREEN if armed else (110, 110, 110)
        cv2.rectangle(
            canvas, (x + 1, y + 1), (max(to_x(value) - 1, x + 1), y + bar_h - 1), fill, -1
        )
    # Threshold ticks: engage (green) and release (white).
    for tv, tc in ((engage, _GREEN), (release, _WHITE)):
        tx = to_x(tv)
        cv2.line(canvas, (tx, y - 3), (tx, y + bar_h + 3), tc, 1)
    text = f"{label} {value:.2f}" if value is not None else f"{label} -"
    _put_text(canvas, text, (x + bar_w + 8, y + bar_h - 1), _WHITE, 0.4)


def _draw_pinch_meters(
    canvas: np.ndarray, pinch_values: dict[str, float], cfg: Config
) -> None:
    h = canvas.shape[0]
    bar_w, bar_h, gap = 160, 10, 8
    specs = (
        ("L", "left", cfg.pinch.left_engage, cfg.pinch.left_release),
        ("R", "right", cfg.pinch.right_engage, cfg.pinch.right_release),
    )
    y = h - gap - bar_h - (len(specs) - 1) * (bar_h + gap)
    for label, key, engage, release in specs:
        _draw_meter(
            canvas, gap, y, bar_w, bar_h, label, pinch_values.get(key), engage, release
        )
        y += bar_h + gap


def _draw_palm_meter(
    canvas: np.ndarray, palm_debug: dict[str, float | bool | str], cfg: Config
) -> None:
    """Two rows above the pinch meters: the five-finger spread metric
    (Launchpad = falls below spread_closed; Show Desktop = rises above
    spread_out), and the swipe detector's state — ARMING/ARMED/COOLDOWN tag
    plus a displacement-progress bar that fills toward the fire threshold
    while a swipe is in flight."""
    h = canvas.shape[0]
    bar_w, bar_h, gap = 160, 10, 8
    y = h - gap - bar_h - 2 * (bar_h + gap)   # one row above the 2 pinch meters
    m = palm_debug.get("m")
    m_val = m if isinstance(m, float) else None
    _draw_meter(
        canvas, gap, y, bar_w, bar_h, "P", m_val,
        cfg.palm.spread_closed, cfg.palm.spread_open, full_scale=1.4,
    )
    open_now = bool(palm_debug.get("open"))
    if open_now:
        _put_text(canvas, "4-FINGER OPEN", (gap + bar_w + 90, y + bar_h - 1),
                  _GREEN, 0.4)

    # Swipe row: displacement progress toward swipe_min_disp_frac + phase tag.
    y_sw = y - bar_h - gap
    disp = palm_debug.get("disp_frac")
    disp_val = disp if isinstance(disp, float) else None
    _draw_meter(
        canvas, gap, y_sw, bar_w, bar_h, "S", disp_val,
        cfg.palm.swipe_min_disp_frac, cfg.palm.swipe_min_disp_frac,
        full_scale=max(0.5, cfg.palm.swipe_min_disp_frac * 2.0), invert=True,
    )
    phase = palm_debug.get("phase")
    if isinstance(phase, str) and phase != "idle":
        tag = phase.upper()
        color = _GREEN if phase == "armed" else (
            _GRAY if phase == "cooldown" else _WHITE)
        _put_text(canvas, tag, (gap + bar_w + 90, y_sw + bar_h - 1), color, 0.4)


def _draw_banner(canvas: np.ndarray, text: str, y: int, bg: tuple[int, int, int]) -> int:
    """Full-width band with centered text; returns the y below the band."""
    w = canvas.shape[1]
    band_h = 24
    cv2.rectangle(canvas, (0, y), (w, y + band_h), bg, -1)
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.5, 1)
    cv2.putText(
        canvas, text, ((w - tw) // 2, y + (band_h + th) // 2), _FONT, 0.5, _WHITE, 1,
        cv2.LINE_AA,
    )
    return y + band_h + 4


# State -> what-to-do-next, shown as a second status line. Usability: the
# user should never have to remember the state machine — the UI says what
# the current state expects.
_HINTS: dict[EngineState, str] = {
    EngineState.CLUTCH_WAIT: "hold INDEX FINGER up ~150ms for cursor, or open hand still + flick to swipe",
    EngineState.POINTER: "pinch=click  thumb+middle=right  2 fingers=scroll  open hand still=swipe  horns=custom",
    EngineState.PINCHED: "move to drag, open fingers to release",
    EngineState.RIGHT_PINCH: "release quickly for right-click",
    EngineState.SCROLL: "move up/down to scroll, flick sideways for tabs, relax to exit",
    EngineState.PALM: "flick now to swipe (or relax to cancel)",
    EngineState.HANDS_LOST: "show your hand + hold index finger up 250ms",
}


@dataclass(frozen=True)
class _Regions:
    """Reserved, non-overlapping screen y-positions for one draw_overlay()
    call. Every element reads its position from here instead of picking a
    fixed offset independently — THIS is what actually prevents collisions
    (plan doc: UI scaling/overlap fix), not the specific numbers below."""

    status_y: int
    hint_y: int
    camera_list_top: int


def _compute_regions(canvas_w: int, canvas_h: int) -> _Regions:
    status_y = 18
    hint_y = 36
    # Stacked BELOW the status+hint block (not beside it, right-aligned as
    # it is) so it can never collide with them regardless of how wide
    # either line's text ends up being at any canvas size.
    camera_list_top = hint_y + 20
    return _Regions(status_y=status_y, hint_y=hint_y, camera_list_top=camera_list_top)


def _draw_status(canvas: np.ndarray, snap: StateSnapshot, regions: _Regions) -> None:
    engine = snap.engine_state.value if snap.engine_state else "-"
    text = (
        f"{snap.session_state.value}:{engine}"
        f"  {snap.fps:4.1f} fps  {snap.latency_ms:5.1f} ms"
    )
    if snap.suspend_reason:
        text += f"  [{snap.suspend_reason}]"
    if not snap.hand_present:
        text += "  (no hand)"
    _put_text(canvas, text, (8, regions.status_y), _state_color(snap))

    if snap.session_state is SessionState.SUSPENDED:
        hint = "you used mouse/keyboard - hold index finger up 250ms (or ctrl+alt+G) to resume"
    elif snap.session_state is SessionState.IDLE:
        hint = "press ctrl+alt+G to start"
    elif snap.engine_state is not None:
        hint = _HINTS.get(snap.engine_state, "")
    else:
        hint = ""
    if hint:
        _put_text(canvas, hint, (8, regions.hint_y), _GRAY, 0.4)


_HELP_LINES: tuple[tuple[str, str], ...] = (
    ("GESTURES", ""),
    ("index finger up 150ms", "take the cursor (clutch)"),
    ("thumb+index pinch", "left click (two taps = double)"),
    ("thumb+middle pinch", "right click (index stays up)"),
    ("pinch + move", "drag"),
    ("index+middle up, move", "scroll / flick sideways = switch tabs"),
    ("open hand STILL, then flick", "left/right = Spaces, up = Mission Control, down = App Expose"),
    ("open palm -> fist", "Launchpad"),
    ("fist -> open palm", "Show Desktop"),
    ("KEYS", ""),
    ("h", "toggle this help"),
    ("1-9", "switch camera (list top-right)"),
    ("[ ] and ; '", "tune cursor smoothing / responsiveness"),
    ("- = and , .", "tune gesture forgiveness / smoothing"),
    ("p / b / q", "privacy view / control box / quit"),
    ("SAFETY", ""),
    ("touch mouse or type", "gestures suspend instantly"),
    ("ctrl+alt+G / ctrl+alt+esc", "toggle on-off / panic stop"),
)


def _draw_help(canvas: np.ndarray, cfg: Config) -> None:
    """Semi-transparent guide panel: gestures, keys, safety, plus whatever
    custom gestures the config defines (so the guide is never stale)."""
    lines: list[tuple[str, str]] = list(_HELP_LINES)
    customs = [g for g in (getattr(cfg, "custom_gestures", None) or [])
               if isinstance(g, dict) and g.get("pose")]
    if customs:
        lines.append(("CUSTOM (config.json)", ""))
        for g in customs:
            act = g.get("action") or {}
            if act.get("type") == "key":
                desc = "press " + "+".join(
                    list(act.get("modifiers") or []) + [str(act.get("key", "?"))])
            elif act.get("type") == "shell":
                desc = "run " + " ".join(act.get("argv") or ["?"])
            else:
                desc = str(act.get("name", act.get("type", "?")))
            lines.append((f"{g.get('pose')} pose ({g.get('name')})", desc))

    h, w = canvas.shape[:2]
    pad, line_h = 14, 17
    panel_h = pad * 2 + line_h * len(lines)
    panel_w = min(w - 20, 560)
    x0, y0 = (w - panel_w) // 2, max(44, (h - panel_h) // 2)
    sub = canvas[y0:y0 + panel_h, x0:x0 + panel_w]
    sub[:] = (sub * 0.25).astype(sub.dtype)          # darken, keep context
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), _GRAY, 1)
    y = y0 + pad + 4
    for left_txt, right_txt in lines:
        if not right_txt:                            # section header
            _put_text(canvas, left_txt, (x0 + pad, y), _GREEN, 0.45)
        else:
            _put_text(canvas, left_txt, (x0 + pad, y), _WHITE, 0.42)
            _put_text(canvas, right_txt, (x0 + pad + 210, y), _GRAY, 0.42)
        y += line_h


def _draw_camera_list(
    canvas: np.ndarray, cameras: list[str], current_index: int | None, y: int
) -> int:
    """Top-right numbered camera list; the active one is highlighted. Press
    the matching digit key (1-9) to switch. Returns the y below the list."""
    w = canvas.shape[1]
    for i, name in enumerate(cameras[:9]):
        current = i == current_index
        label = f"[{i + 1}] {name}" + ("  *" if current else "")
        color = _GREEN if current else _GRAY
        (tw, _), _ = cv2.getTextSize(label, _FONT, 0.42, 1)
        _put_text(canvas, label, (w - tw - 8, y), color, 0.42)
        y += 16
    return y


def draw_overlay(
    canvas: np.ndarray,
    lm_frame: LandmarkFrame | None,
    snap: StateSnapshot,
    pinch_values: dict[str, float],
    cfg: Config,
    show_box: bool = True,
    palm_debug: dict[str, float | bool] | None = None,
    cameras: list[str] | None = None,
    current_camera_index: int | None = None,
    show_help: bool = False,
) -> np.ndarray:
    """Draw all preview widgets onto ``canvas`` (in place) and return it.

    Pure cv2-primitive drawing on the given numpy image — no windows, no
    waitKey — so it is testable headless. ``show_box`` gates the control-box
    rectangle (the 'b' live-tune key). ``canvas`` is assumed to be the fixed
    display size (``cfg.preview.canvas_w/h``); ``lm_frame``'s landmark
    points are in its OWN original capture size and are rescaled here.
    """
    h, w = canvas.shape[:2]
    regions = _compute_regions(w, h)
    if show_box:
        _draw_control_box(canvas, cfg)
    if lm_frame is not None and lm_frame.hand_present:
        sx = w / lm_frame.img_w if lm_frame.img_w else 1.0
        sy = h / lm_frame.img_h if lm_frame.img_h else 1.0
        _draw_skeleton(canvas, lm_frame, _state_color(snap), sx, sy)
    if palm_debug is not None:
        _draw_palm_meter(canvas, palm_debug, cfg)
    _draw_pinch_meters(canvas, pinch_values, cfg)
    _draw_status(canvas, snap, regions)
    if cameras:
        _draw_camera_list(canvas, cameras, current_camera_index, regions.camera_list_top)
    # The guide doubles as the IDLE screen (nothing else to show there), and
    # is toggleable any time with 'h'.
    if show_help or snap.session_state is SessionState.IDLE:
        _draw_help(canvas, cfg)

    banner_y = 28
    if snap.session_state is SessionState.WARMUP:
        banner_y = _draw_banner(canvas, "WARMING UP...", banner_y, (90, 60, 20))
    if 0.0 < snap.fps < LOW_FPS_THRESHOLD:
        _draw_banner(canvas, f"LOW FPS: {snap.fps:.0f}", banner_y, (30, 30, 150))
    return canvas


class Preview:
    """cv2 tuning window. Window creation is lazy (first ``show()``)."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._opened = False
        self.show_box = True   # control-box overlay toggle ('b' key)
        self.show_help = False  # guide overlay toggle ('h' key; auto in IDLE)

    def show(
        self,
        bgr_frame: np.ndarray | None,
        lm_frame: LandmarkFrame | None,
        snap: StateSnapshot,
        pinch_values: dict[str, float],
        palm_debug: dict[str, float | bool] | None = None,
        cameras: list[str] | None = None,
        current_camera_index: int | None = None,
    ) -> int:
        """Render one frame; returns the ``cv2.waitKey(1)`` key code (-1 when
        no key was pressed). ``waitKey`` also pumps the Cocoa event loop.

        ``cameras`` (all enumerated camera names) + ``current_camera_index``
        draw the on-screen camera picker (press 1-9 to switch); both default
        to None so callers that don't offer switching (replay) need no change.
        """
        cw, ch = self._cfg.preview.canvas_w, self._cfg.preview.canvas_h

        if self._cfg.options.privacy_preview or bgr_frame is None:
            canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
        else:
            canvas = _fit_frame_to_canvas(bgr_frame, cw, ch)

        draw_overlay(canvas, lm_frame, snap, pinch_values, self._cfg,
                     show_box=self.show_box, palm_debug=palm_debug,
                     cameras=cameras, current_camera_index=current_camera_index,
                     show_help=self.show_help)

        if not self._opened:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            self._opened = True
        cv2.imshow(WINDOW_NAME, canvas)
        return cv2.waitKey(1)

    def close(self) -> None:
        if self._opened:
            self._opened = False
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except cv2.error:
                pass  # window already gone (user closed it / shutdown race)
