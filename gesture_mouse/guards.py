"""Suspend guards: physical input always beats synthesis.

Both checks are cheap polls of Quartz global state — no event tap, no NSEvent
monitor, no run loop, and no Input Monitoring TCC grant.
CGEventSourceSecondsSinceLastEventType and CGEventGetLocation work without any
permission at all; only the synthesizer itself needs Accessibility.

Keyboard attribution: the HID keyDown timestamp cannot distinguish real typing
from our own synthetic chords (palm gestures) or from the toggle hotkey press
that started the session. Instead of a blanket grace window (which either
self-suspends after every chord or goes deaf to real typing), we reconstruct
the *time of* the last keyDown and ignore it only when it is attributable to
a known synthetic/pre-activation moment; any keyDown strictly after those
moments mutes immediately.
"""
from __future__ import annotations

import math
import time

import Quartz

from .config import SuspendConfig
from .synth import Synth, real_cursor_pos

# A keyDown within this window after a synthetic chord / activation keystroke
# is attributed to that event, not to the user typing.
_ATTRIB_SLOP_S = 0.10
# Per-frame divergence below this is measurement noise, not a human hand.
_DIVERGENCE_NOISE_PX = 1.5
# Posts younger than this can still be "the" applied cursor position.
_POST_WINDOW_S = 0.20


class Guards:
    """Polled each frame by the hot loop; any True triggers SUSPENDED."""

    def __init__(self, cfg: SuspendConfig, synth: Synth) -> None:
        self._cfg = cfg
        self._synth = synth
        self._keyboard_baseline: float = float("-inf")
        self._div_accum: float = 0.0

    def rearm(self) -> None:
        """Call on every ACTIVE entry (activation and resume): keyDowns from
        before this moment — the toggle hotkey, the typing that caused a
        keyboard suspend — never count against the new session."""
        self._keyboard_baseline = time.monotonic()
        self._div_accum = 0.0

    def mouse_moved_physically(self) -> bool:
        """True when the real cursor diverges from OUR recent posts — someone
        grabbed the physical mouse.

        Two subtleties, both learned the hard way:

        - Posted events take a frame or two to apply, so while the hand moves
          the cursor, ``real pos`` lags the newest post by one frame's travel.
          Comparing against the newest post alone suspends the app the moment
          it starts working. Divergence is therefore the MINIMUM distance to
          any recent post (a short window) — our own in-flight stream always
          matches one of them.
        - A slow physical mouse move under the synthetic stream never exceeds
          the threshold in one frame (the stream keeps snapping the cursor
          back), so above-noise divergence *accumulates* across consecutive
          frames and a clean frame resets it.
        """
        if not self._synth.has_posted:
            return False  # nothing posted yet: any divergence is not ours to judge
        rx, ry = real_cursor_pos()
        now = time.monotonic()
        candidates = [
            (x, y) for (t, x, y) in self._synth.recent_posts
            if now - t <= _POST_WINDOW_S
        ]
        if not candidates:
            candidates = [self._synth.last_pos]
        div = min(math.hypot(rx - x, ry - y) for (x, y) in candidates)
        if div > self._cfg.mouse_divergence_px:
            return True
        if div > _DIVERGENCE_NOISE_PX:
            self._div_accum += div
            return self._div_accum > self._cfg.mouse_divergence_px
        self._div_accum = 0.0
        return False

    def keyboard_active(self) -> bool:
        """True when a *real* key went down within keyboard_mute_ms."""
        seconds = Quartz.CGEventSourceSecondsSinceLastEventType(
            Quartz.kCGEventSourceStateHIDSystemState, Quartz.kCGEventKeyDown)
        if seconds * 1000.0 >= self._cfg.keyboard_mute_ms:
            return False
        last_keydown = time.monotonic() - seconds
        if last_keydown <= self._synth.last_chord_ts + _ATTRIB_SLOP_S:
            return False  # our own synthetic chord
        if last_keydown <= self._keyboard_baseline + _ATTRIB_SLOP_S:
            return False  # pre-activation keystroke (e.g. the toggle hotkey)
        return True
