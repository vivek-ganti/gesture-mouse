"""Space (virtual desktop) switching WITHOUT raw keyboard chords.

Synthetic Ctrl+Left/Right chords posted via CGEventPost do not switch Spaces
on all machines: on the development Mac, every direct-post variant —
flags-only, real bracketing modifier presses, inter-event delays, direction
and cursor-display controlled — was measured (via the window server's
active-space ID) to have zero effect. Two delivery paths DID verifiably
switch the Space on that same machine:

1. Activating an application whose windows live on the target Space — macOS
   itself then switches Spaces (normal animation). Public, supported
   (``NSRunningApplication.activate``), instant, posts no keystrokes (nothing
   for the keyboard guard to attribute), needs no Automation permission.
   This module implements that path; it is the PRIMARY mechanism.
2. The same chord sent through AppleScript System Events (osascript) —
   measured switching within ~1s. That is the caller's FALLBACK for the
   cases this module reports False (e.g. the target is an empty desktop
   with nothing to activate); it needs the one-time Automation consent.

Mechanism:
1. Enumerate the current display's Spaces in order + the active one, via the
   read-only private SkyLight call ``CGSCopyManagedDisplaySpaces`` (the same
   API used by WhichSpace and friends; stable for a decade).
2. Pick the neighbor in the requested direction.
3. Find something to activate there: fullscreen Spaces carry their owning
   ``pid`` directly; for desktop Spaces, map candidate windows (public
   ``CGWindowListCopyWindowInfo``) onto Spaces with ``CGSCopySpacesForWindows``
   and take the frontmost-most candidate's owner.
4. ``NSRunningApplication.activate()`` — a public, supported call.

Returns False when there is no neighbor or nothing activatable on it (e.g.
an empty desktop) — the caller falls back to posting the chord, which is
also what happens on machines where the chord path works fine anyway.
"""
from __future__ import annotations

import ctypes
import logging

import objc
from AppKit import NSRunningApplication
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGNullWindowID,
    kCGWindowLayer,
    kCGWindowListOptionAll,
    kCGWindowNumber,
    kCGWindowOwnerPID,
)

log = logging.getLogger(__name__)

_sky = ctypes.CDLL(
    "/System/Library/PrivateFrameworks/SkyLight.framework/SkyLight"
)
_sky.CGSMainConnectionID.restype = ctypes.c_int
_sky.CGSMainConnectionID.argtypes = []
_sky.CGSCopyManagedDisplaySpaces.restype = ctypes.c_void_p
_sky.CGSCopyManagedDisplaySpaces.argtypes = [ctypes.c_int]
_sky.CGSCopySpacesForWindows.restype = ctypes.c_void_p
_sky.CGSCopySpacesForWindows.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

_cf = ctypes.CDLL(
    "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
)
_cf.CFRelease.argtypes = [ctypes.c_void_p]
_cf.CFRelease.restype = None

_ALL_SPACES_MASK = 7  # kCGSSpaceIncludesCurrent | ...Others | ...User


def _bridge_and_copy(ptr: int | None):
    """Toll-free-bridge a CF Copy-rule pointer to Python data, then release.

    objc.objc_object(c_void_p=...) RETAINS (does not consume the +1 from the
    Copy call), so after deep-copying to plain Python and dropping the
    wrapper, one CFRelease balances the Copy.
    """
    if not ptr:
        return None
    wrapper = objc.objc_object(c_void_p=ptr)
    try:
        import Foundation  # noqa: F401  (ensures containers are bridged)

        def plain(obj):
            if hasattr(obj, "items"):
                return {str(k): plain(v) for k, v in obj.items()}
            if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
                return [plain(x) for x in obj]
            if hasattr(obj, "doubleValue"):
                d = obj.doubleValue()
                return int(d) if d == int(d) else d
            return obj
        return plain(wrapper)
    finally:
        del wrapper
        _cf.CFRelease(ptr)


def _display_spaces() -> tuple[list[dict], int | None] | None:
    """(ordered space dicts, index of the active one) for the display that
    currently has the active space; None when unavailable."""
    cid = _sky.CGSMainConnectionID()
    data = _bridge_and_copy(_sky.CGSCopyManagedDisplaySpaces(cid))
    if not data:
        return None
    for display in data:
        spaces = display.get("Spaces") or []
        current = (display.get("Current Space") or {}).get("ManagedSpaceID")
        if not spaces or current is None:
            continue
        for i, sp in enumerate(spaces):
            if sp.get("ManagedSpaceID") == current:
                return spaces, i
    return None


def _spaces_of_window(cid: int, window_id: int) -> list[int]:
    from Foundation import NSArray

    arr = NSArray.arrayWithObject_(window_id)
    ptr = _sky.CGSCopySpacesForWindows(
        cid, _ALL_SPACES_MASK, ctypes.c_void_p(objc.pyobjc_id(arr))
    )
    out = _bridge_and_copy(ptr)
    return [int(x) for x in out] if out else []


def _pid_on_space(space_id: int) -> int | None:
    """PID of an app with a normal window on the given space, preferring the
    front of the window list (roughly z-ordered)."""
    cid = _sky.CGSMainConnectionID()
    windows = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)
    if not windows:
        return None
    checked = 0
    for w in windows:
        if w.get(kCGWindowLayer, 1) != 0:
            continue  # not a normal app window
        wid = w.get(kCGWindowNumber)
        pid = w.get(kCGWindowOwnerPID)
        if wid is None or pid is None:
            continue
        checked += 1
        if checked > 60:  # bound the SkyLight round-trips
            break
        if space_id in _spaces_of_window(cid, int(wid)):
            return int(pid)
    return None


def switch_space(direction: str) -> bool:
    """direction: 'prev' (left) or 'next' (right). True if an activation was
    issued for the adjacent Space; False -> caller should fall back to the
    keyboard chord."""
    try:
        found = _display_spaces()
        if found is None:
            return False
        spaces, idx = found
        target_idx = idx - 1 if direction == "prev" else idx + 1
        if not 0 <= target_idx < len(spaces):
            log.debug("no space to the %s", direction)
            return False
        target = spaces[target_idx]

        pid = target.get("pid")  # fullscreen spaces carry their owner
        if pid is None:
            pid = _pid_on_space(int(target.get("ManagedSpaceID")))
        if pid is None:
            log.debug("nothing activatable on target space")
            return False
        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(
            int(pid)
        )
        if app is None:
            return False
        return bool(app.activateWithOptions_(0))
    except Exception:
        log.exception("switch_space failed; falling back to chord")
        return False
