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

# TRIGGER name -> subprocess argv. System actions are delivered through
# external mechanisms, NOT raw CGEvent keyboard chords, after live debugging
# on real hardware proved that macOS's Spaces switcher ("Move left/right a
# space") never reacts to synthetic Ctrl+Arrow chords posted via CGEventPost
# from this process — regardless of flags-on-event, real bracketing modifier
# key presses, inter-event delays, cursor display, or direction validity —
# while the exact same chord sent through AppleScript System Events switches
# the Space within ~1s (verified via SkyLight CGSCopyManagedDisplaySpaces
# before/after). Mission Control / App Expose / Show Desktop don't need
# keystrokes at all: the Mission Control binary takes an argument (none =
# toggle Mission Control, 1 = show desktop, 2 = app windows).
#
# All of these run via subprocess.Popen — fire-and-forget, never blocking
# the 30Hz hot loop. First use of the osascript path triggers a ONE-TIME
# macOS Automation permission prompt ("... wants to control System Events");
# permissions.preflight() surfaces it at startup instead of mid-gesture.
_MC_BIN = "/System/Applications/Mission Control.app/Contents/MacOS/Mission Control"


def _osascript_keystroke(keycode: int, *modifiers: str) -> list[str]:
    using = ""
    if modifiers:
        using = " using {" + ", ".join(f"{m} down" for m in modifiers) + "}"
    return ["osascript", "-e",
            f'tell application "System Events" to key code {keycode}{using}']


def trigger_command(name: str) -> list[str] | None:
    """argv for a TRIGGER intent, or None for unknown names (pure; tested)."""
    commands: dict[str, list[str]] = {
        "space_prev": _osascript_keystroke(123, "control"),      # Ctrl+Left
        "space_next": _osascript_keystroke(124, "control"),      # Ctrl+Right
        "mission_control": [_MC_BIN],
        "app_expose": [_MC_BIN, "2"],
        "show_desktop": [_MC_BIN, "1"],
        "launchpad": ["open", "-a", "Launchpad"],
        "tab_next": _osascript_keystroke(48, "control"),         # Ctrl+Tab
        "tab_prev": _osascript_keystroke(48, "control", "shift"),
    }
    return commands.get(name)


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

    def _run_trigger(self, name: str) -> None:
        """Launch the external command for a TRIGGER intent, fire-and-forget.

        space_prev/space_next go through spaces.switch_space() FIRST: it
        activates an app living on the target Space (a public, supported
        call), which is instant, posts no keystrokes at all (nothing for the
        keyboard guard to attribute), and needs no Automation permission.
        Only when it reports no target/candidate (e.g. an empty desktop) do
        we fall back to the osascript keystroke.

        Popen (never run/wait): the 30Hz hot loop must not stall on process
        spawn + osascript execution (~100-400ms). last_chord_ts marks the
        LAUNCH time; the actual keystroke (for the osascript paths) lands up
        to a few hundred ms later, which is why the keyboard guard attributes
        keyDowns to us for a generous window after this timestamp."""
        if name in ("space_prev", "space_next"):
            from . import spaces  # lazy: AppKit/SkyLight only when first used

            if spaces.switch_space("prev" if name == "space_prev" else "next"):
                return
            log.info("%s: no activatable app on target space; osascript fallback",
                     name)
        argv = trigger_command(name)
        if argv is None:
            log.warning("no trigger command for %s", name)
            return
        subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
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
        elif phase is Phase.TRIGGER and trigger_command(name) is not None:
            self._run_trigger(name)
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
