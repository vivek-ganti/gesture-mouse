"""Corner-dot status indicator: a tiny always-on-top NSPanel dot.

A ~16x16 borderless, non-activating, click-through panel pinned to the
top-right corner of the main screen (8px margin, below the menu bar). The dot
color is the state readout (plan doc "Feedback UI"):

    gray idle - pulsing warmup - white clutch-wait - green pointer -
    blue pinched - purple scroll - orange palm - yellow hands-lost -
    red suspended (with a tiny M/K/C glyph for the mouse/keyboard/meeting
    suspend reason).

Constraints this implementation is built around:

- **Never steals focus.** The panel has ``NSWindowStyleMaskNonactivatingPanel``,
  ignores all mouse events, and is shown with ``orderFrontRegardless()`` which
  orders the window without activating the app. A borderless panel also refuses
  key/main status by default.
- **Visible everywhere.** ``NSStatusWindowLevel`` plus collection behavior
  CanJoinAllSpaces | FullScreenAuxiliary | Stationary keeps it on every Space,
  over fullscreen apps, and out of Expose shuffles.
- **Main thread only.** AppKit is not thread-safe; the hot loop *is* the main
  thread, so ``set_state()`` / ``close()`` must be called from it and
  ``set_state()`` raises if not.
- **Lazy window creation.** ``Indicator()`` touches no window-server state, so
  the module imports and the class constructs in headless test environments.
  The panel is created on the first ``set_state()`` call; if creation fails
  (no display / no window server) the indicator disables itself with one
  stderr warning instead of crashing the pipeline.
- **NSApplication coexistence with cv2.** cv2's HighGUI also runs Cocoa on the
  main thread. If this process has no UI application identity yet (activation
  policy Prohibited — a plain python process), we create the shared app and set
  policy Accessory so no Dock icon or menu bar appears. If something else
  (cv2) already promoted the process, the policy is left alone.
- **Redraw without a running NSRunLoop.** With ``--no-preview`` nothing pumps
  Cocoa events, so after flagging the view dirty we force a synchronous draw
  and spin the runloop for one non-blocking pass so the CoreAnimation
  transaction actually commits to the screen.
"""
from __future__ import annotations

import math
import sys
import time

import objc
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSApplicationActivationPolicyProhibited,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSPanel,
    NSRunningApplication,
    NSScreen,
    NSStatusWindowLevel,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import (
    NSAttributedString,
    NSDate,
    NSDefaultRunLoopMode,
    NSInsetRect,
    NSMakeRect,
    NSRunLoop,
    NSThread,
)

from .types import EngineState, SessionState, StateSnapshot

DOT_SIZE: float = 16.0   # panel content size, points
MARGIN: float = 8.0      # gap from the visibleFrame's top-right corner
_PULSE_PERIOD_S: float = 1.2

_REASON_GLYPHS: dict[str, str] = {"mouse": "M", "keyboard": "K", "meeting": "C"}

# sRGB (r, g, b, a) in 0..1 — pure data so tests can assert without Cocoa.
_GRAY = (0.55, 0.55, 0.55, 1.0)
_WHITE = (1.0, 1.0, 1.0, 1.0)
_GREEN = (0.20, 0.84, 0.37, 1.0)
_BLUE = (0.25, 0.55, 1.0, 1.0)
_PURPLE = (0.72, 0.42, 0.96, 1.0)
_ORANGE = (1.0, 0.62, 0.15, 1.0)
_YELLOW = (1.0, 0.85, 0.20, 1.0)
_RED = (0.95, 0.23, 0.23, 1.0)

_ENGINE_COLORS: dict[EngineState, tuple[float, float, float, float]] = {
    EngineState.CLUTCH_WAIT: _WHITE,
    EngineState.POINTER: _GREEN,
    EngineState.PINCHED: _BLUE,
    EngineState.RIGHT_PINCH: _BLUE,   # both pinches read as "pinched"
    EngineState.SCROLL: _PURPLE,
    EngineState.PALM: _ORANGE,
    EngineState.HANDS_LOST: _YELLOW,
}


def dot_style(
    snap: StateSnapshot | None, now_s: float | None = None
) -> tuple[tuple[float, float, float, float], str | None]:
    """Map a snapshot to ((r, g, b, a) sRGB, optional glyph letter).

    Pure and headless-testable — all Cocoa color/text construction stays in
    the view. ``now_s`` (monotonic seconds) drives the warmup pulse and is
    injectable for deterministic tests.
    """
    if snap is None:
        return _GRAY, None
    if snap.session_state is SessionState.IDLE:
        return _GRAY, None
    if snap.session_state is SessionState.WARMUP:
        t = time.monotonic() if now_s is None else now_s
        pulse = 0.5 + 0.5 * math.sin(t * 2.0 * math.pi / _PULSE_PERIOD_S)
        r, g, b, _ = _WHITE
        return (r, g, b, 0.25 + 0.6 * pulse), None
    if snap.session_state is SessionState.SUSPENDED:
        reason = snap.suspend_reason or ""
        glyph = _REASON_GLYPHS.get(reason) or (reason[:1].upper() or None)
        return _RED, glyph
    if snap.engine_state is None:
        return _WHITE, None
    return _ENGINE_COLORS.get(snap.engine_state, _WHITE), None


class _IndicatorDotView(NSView):
    """The 16x16 dot. Reads ``self.snap`` (set by Indicator) on each draw."""

    def initWithFrame_(self, frame):  # noqa: N802 - ObjC selector
        self = objc.super(_IndicatorDotView, self).initWithFrame_(frame)
        if self is not None:
            self.snap = None
        return self

    def drawRect_(self, rect):  # noqa: N802 - ObjC selector
        (r, g, b, a), glyph = dot_style(self.snap)
        bounds = self.bounds()
        oval = NSBezierPath.bezierPathWithOvalInRect_(NSInsetRect(bounds, 1.5, 1.5))
        NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, a).setFill()
        oval.fill()
        # Thin dark rim so the dot stays visible on white backgrounds.
        NSColor.colorWithSRGBRed_green_blue_alpha_(0.0, 0.0, 0.0, 0.35 * a).setStroke()
        oval.setLineWidth_(1.0)
        oval.stroke()
        if glyph:
            attrs = {
                NSFontAttributeName: NSFont.boldSystemFontOfSize_(9.0),
                NSForegroundColorAttributeName: NSColor.whiteColor(),
            }
            text = NSAttributedString.alloc().initWithString_attributes_(glyph, attrs)
            size = text.size()
            text.drawAtPoint_(
                (
                    bounds.origin.x + (bounds.size.width - size.width) / 2.0,
                    bounds.origin.y + (bounds.size.height - size.height) / 2.0,
                )
            )


class Indicator:
    """Corner-dot state indicator. Main-thread only; lazy window creation."""

    def __init__(self) -> None:
        self._panel = None
        self._view = None
        self._failed = False

    def set_state(self, snap: StateSnapshot) -> None:
        """Store the snapshot and redraw. Cheap (~16x16 bitmap) so it is
        called every frame; the per-frame redraw is what animates the warmup
        pulse."""
        if self._failed:
            return
        if not NSThread.isMainThread():
            raise RuntimeError("Indicator.set_state must be called from the main thread")
        if self._panel is None:
            try:
                self._create_panel()
            except Exception as exc:  # no window server / no display
                self._failed = True
                print(f"gesture-mouse indicator disabled: {exc}", file=sys.stderr)
                return
        self._view.snap = snap
        self._view.setNeedsDisplay_(True)
        self._panel.displayIfNeeded()
        # One non-blocking runloop pass so the CoreAnimation transaction
        # commits even when nothing else (e.g. cv2.waitKey) pumps events.
        NSRunLoop.currentRunLoop().runMode_beforeDate_(
            NSDefaultRunLoopMode, NSDate.distantPast()
        )

    def close(self) -> None:
        if self._panel is not None:
            self._panel.orderOut_(None)
            self._panel = None
            self._view = None

    # -- internal ----------------------------------------------------------

    def _create_panel(self) -> None:
        app = NSApplication.sharedApplication()
        # Plain python processes start Prohibited; become an Accessory app so
        # the panel can appear with no Dock icon and no menu bar. If cv2's
        # HighGUI already gave the process a UI identity, leave it alone.
        current = NSRunningApplication.currentApplication().activationPolicy()
        if current == NSApplicationActivationPolicyProhibited:
            app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        screen = NSScreen.mainScreen()
        if screen is None:
            screens = NSScreen.screens()
            screen = screens[0] if screens else None
        if screen is None:
            raise RuntimeError("no display attached")

        # visibleFrame excludes the menu bar (and Dock); Cocoa is y-up, so the
        # top-right corner is (maxX, maxY) of the visible frame.
        vf = screen.visibleFrame()
        x = vf.origin.x + vf.size.width - DOT_SIZE - MARGIN
        y = vf.origin.y + vf.size.height - DOT_SIZE - MARGIN

        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, DOT_SIZE, DOT_SIZE),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setOpaque_(False)
        panel.setIgnoresMouseEvents_(True)
        panel.setHasShadow_(False)
        panel.setHidesOnDeactivate_(False)
        panel.setBecomesKeyOnlyIfNeeded_(True)
        panel.setLevel_(NSStatusWindowLevel)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary
        )
        # Python owns the panel lifetime; don't let close() dealloc behind us.
        panel.setReleasedWhenClosed_(False)

        view = _IndicatorDotView.alloc().initWithFrame_(
            NSMakeRect(0, 0, DOT_SIZE, DOT_SIZE)
        )
        panel.setContentView_(view)
        panel.orderFrontRegardless()  # shows without activating anything

        self._panel = panel
        self._view = view
