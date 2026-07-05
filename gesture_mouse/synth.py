"""Input synthesis: raw Quartz CGEvents posted to the HID event tap.

The only module that touches CGEventPost. Every event is tagged twice with
0x6D6F7573 ("mous"): once on the CGEventSource userData and once per event via
kCGEventSourceUserData, so an external observer (or a future event-tap guard)
can tell our synthetic input from real input. Mouse positions are always
clamped to the main display bounds before posting.

Never pyautogui / pynput — plain CGEvents need only the Accessibility grant.
"""
from __future__ import annotations

import logging
import subprocess
import time
from collections import deque

import Quartz

from .types import Intent, Phase

log = logging.getLogger(__name__)

# "mous" — tag identifying every event we synthesize.
EVENT_TAG = 0x6D6F7573

# TRIGGER name -> (virtual keycode, CGEventFlags) chords.
_CHORDS: dict[str, tuple[int, int]] = {
    "space_prev": (123, Quartz.kCGEventFlagMaskControl),        # Ctrl+Left
    "space_next": (124, Quartz.kCGEventFlagMaskControl),        # Ctrl+Right
    "mission_control": (126, Quartz.kCGEventFlagMaskControl),   # Ctrl+Up
    "app_expose": (125, Quartz.kCGEventFlagMaskControl),        # Ctrl+Down
    "show_desktop": (103, Quartz.kCGEventFlagMaskSecondaryFn),  # fn+F11
    "tab_next": (48, Quartz.kCGEventFlagMaskControl),           # Ctrl+Tab
    "tab_prev": (48, Quartz.kCGEventFlagMaskControl             # Ctrl+Shift+Tab
                 | Quartz.kCGEventFlagMaskShift),
}

# CGEventFlags mask -> the (left-side) virtual keycode CGEventCreateKeyboardEvent
# turns into a real flagsChanged event for that modifier (Carbon HIToolbox
# Events.h kVK_* constants). Order matters only for readability; posted
# down in this order, up in reverse.
_MODIFIER_KEYCODES: tuple[tuple[int, int], ...] = (
    (Quartz.kCGEventFlagMaskCommand, 55),       # kVK_Command
    (Quartz.kCGEventFlagMaskShift, 56),         # kVK_Shift
    (Quartz.kCGEventFlagMaskAlternate, 58),     # kVK_Option
    (Quartz.kCGEventFlagMaskControl, 59),       # kVK_Control
    (Quartz.kCGEventFlagMaskSecondaryFn, 63),   # kVK_Function
)


def real_cursor_pos() -> tuple[float, float]:
    """Current *actual* cursor position (needs no TCC permission)."""
    loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return (float(loc.x), float(loc.y))


def screen_size() -> tuple[float, float]:
    """Main display size in points (CG global coordinates, y-down)."""
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return (float(bounds.size.width), float(bounds.size.height))


def _carry_round(value: float, carry: float) -> tuple[int, float]:
    """Integer part of value+carry plus the new fractional remainder."""
    total = value + carry
    whole = int(total)  # truncate toward zero so the remainder keeps its sign
    return whole, total - whole


def _clamp_to_main_display(x: float, y: float) -> tuple[float, float]:
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    x0, y0 = float(bounds.origin.x), float(bounds.origin.y)
    # -1 keeps the post strictly on-screen; exactly width/height is off-display.
    x1 = x0 + float(bounds.size.width) - 1.0
    y1 = y0 + float(bounds.size.height) - 1.0
    return (min(max(x, x0), x1), min(max(y, y0), y1))


class Synth:
    """Executes engine Intents as CGEvents. See types.Intent for the vocabulary."""

    def __init__(self) -> None:
        self._source = Quartz.CGEventSourceCreate(
            Quartz.kCGEventSourceStateHIDSystemState)
        Quartz.CGEventSourceSetUserData(self._source, EVENT_TAG)
        self.left_down: bool = False
        self.right_down: bool = False
        # Seeded with the real cursor so release_all() before any post is sane;
        # guards must gate on has_posted, not on last_pos itself.
        self.last_pos: tuple[float, float] = real_cursor_pos()
        self.has_posted: bool = False          # True once any mouse event posted
        self.last_chord_ts: float = float("-inf")  # time.monotonic() of last chord
        self._scroll_carry: float = 0.0        # sub-pixel scroll remainder
        # Recent posted positions (monotonic s, x, y). Posted events take a
        # frame or two to apply to the real cursor, so the mouse guard must
        # compare against this window, not only the newest post — otherwise
        # our own in-flight moves read as "physical mouse" divergence.
        self.recent_posts: deque[tuple[float, float, float]] = deque(maxlen=8)

    # -- low-level posting ----------------------------------------------------

    def _post(self, event: object) -> None:
        Quartz.CGEventSetIntegerValueField(
            event, Quartz.kCGEventSourceUserData, EVENT_TAG)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def _post_mouse(self, event_type: int, x: float, y: float, button: int,
                    click_state: int | None = None) -> None:
        x, y = _clamp_to_main_display(x, y)
        event = Quartz.CGEventCreateMouseEvent(
            self._source, event_type, (x, y), button)
        if click_state is not None:
            Quartz.CGEventSetIntegerValueField(
                event, Quartz.kCGMouseEventClickState, click_state)
        self._post(event)
        self.last_pos = (x, y)
        self.has_posted = True
        self.recent_posts.append((time.monotonic(), x, y))

    def _post_chord(self, keycode: int, flags: int) -> None:
        """Post modifier(s) as REAL flagsChanged key presses bracketing the
        base key, not just a flags bit set on the base key's own event.

        Setting CGEventSetFlags alone is not reliably enough: some apps (and
        even some system shortcut dispatch) read actual modifier-key state
        rather than trusting the flags field on an unrelated key event, so a
        flags-only chord can arrive at the frontmost app as a bare,
        unmodified key — observed in practice as Ctrl+Left/Right performing
        a plain arrow-key action (e.g. video seek) instead of switching
        Spaces. Each active modifier gets its own keyDown before the base
        key and keyUp after; the base key's own down/up still carry the
        flags too (redundant, but that's the documented-safe pattern).
        """
        mods = [kc for mask, kc in _MODIFIER_KEYCODES if flags & mask]
        for mod_kc in mods:
            ev = Quartz.CGEventCreateKeyboardEvent(self._source, mod_kc, True)
            Quartz.CGEventSetFlags(ev, flags)
            self._post(ev)
        for key_down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(
                self._source, keycode, key_down)
            Quartz.CGEventSetFlags(event, flags)
            self._post(event)
        for mod_kc in reversed(mods):
            ev = Quartz.CGEventCreateKeyboardEvent(self._source, mod_kc, False)
            Quartz.CGEventSetFlags(ev, 0)
            self._post(ev)
        self.last_chord_ts = time.monotonic()

    # -- intent handlers ------------------------------------------------------

    def _xy(self, payload: dict) -> tuple[float, float]:
        return (float(payload.get("x", self.last_pos[0])),
                float(payload.get("y", self.last_pos[1])))

    def _move(self, x: float, y: float) -> None:
        # While the left button is held every position update must be a drag
        # event or apps see the button released.
        if self.left_down:
            self._post_mouse(Quartz.kCGEventLeftMouseDragged, x, y,
                             Quartz.kCGMouseButtonLeft)
        else:
            self._post_mouse(Quartz.kCGEventMouseMoved, x, y,
                             Quartz.kCGMouseButtonLeft)

    def _scroll(self, dy_px: float) -> None:
        # Per-frame deltas in the slow-scroll band are fractional (< 1 px);
        # rounding each frame independently would zero the whole band, so the
        # remainder carries into the next frame.
        dy, self._scroll_carry = _carry_round(dy_px, self._scroll_carry)
        if dy == 0:
            return
        # Intent convention is +down / -up; CG scroll wheel is positive-up.
        event = Quartz.CGEventCreateScrollWheelEvent2(
            self._source, Quartz.kCGScrollEventUnitPixel, 1, -dy, 0, 0)
        self._post(event)

    def execute(self, intent: Intent) -> None:
        """Dispatch one Intent. Unknown name/phase combos are logged, ignored."""
        name, phase, payload = intent.name, intent.phase, intent.payload

        if name == "move" and phase is Phase.MOVE:
            self._move(*self._xy(payload))
        elif name == "drag" and phase is Phase.MOVE:
            x, y = self._xy(payload)
            self._post_mouse(Quartz.kCGEventLeftMouseDragged, x, y,
                             Quartz.kCGMouseButtonLeft)
        elif name == "left" and phase is Phase.DOWN:
            x, y = self._xy(payload)
            clicks = int(payload.get("click_count", 1))
            self._post_mouse(Quartz.kCGEventLeftMouseDown, x, y,
                             Quartz.kCGMouseButtonLeft, click_state=clicks)
            self.left_down = True
        elif name == "left" and phase is Phase.UP:
            x, y = self._xy(payload)
            clicks = int(payload.get("click_count", 1))
            self._post_mouse(Quartz.kCGEventLeftMouseUp, x, y,
                             Quartz.kCGMouseButtonLeft, click_state=clicks)
            self.left_down = False
        elif name == "right" and phase is Phase.DOWN:
            x, y = self._xy(payload)
            self._post_mouse(Quartz.kCGEventRightMouseDown, x, y,
                             Quartz.kCGMouseButtonRight, click_state=1)
            self.right_down = True
        elif name == "right" and phase is Phase.UP:
            x, y = self._xy(payload)
            self._post_mouse(Quartz.kCGEventRightMouseUp, x, y,
                             Quartz.kCGMouseButtonRight, click_state=1)
            self.right_down = False
        elif name == "scroll" and phase is Phase.MOVE:
            self._scroll(float(payload.get("dy_px", 0.0)))
        elif name in _CHORDS and phase is Phase.TRIGGER:
            keycode, flags = _CHORDS[name]
            self._post_chord(keycode, flags)
        elif name == "launchpad" and phase is Phase.TRIGGER:
            # No default keyboard shortcut exists for Launchpad.
            subprocess.Popen(["open", "-a", "Launchpad"])
        else:
            log.warning("ignoring unknown intent %s/%s", name, phase.value)

    def release_all(self) -> None:
        """Post UP for any held synthetic button (invariant: never exit with a
        button down). Safe to call repeatedly; exceptions swallowed so atexit
        and finally blocks never mask the original error."""
        x, y = self.last_pos
        if self.left_down:
            try:
                self._post_mouse(Quartz.kCGEventLeftMouseUp, x, y,
                                 Quartz.kCGMouseButtonLeft, click_state=1)
            except Exception:
                log.exception("release_all: left UP failed")
            self.left_down = False
        if self.right_down:
            try:
                self._post_mouse(Quartz.kCGEventRightMouseUp, x, y,
                                 Quartz.kCGMouseButtonRight, click_state=1)
            except Exception:
                log.exception("release_all: right UP failed")
            self.right_down = False
