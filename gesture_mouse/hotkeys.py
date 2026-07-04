"""Global hotkeys via Carbon ``RegisterEventHotKey`` (raw ctypes).

Implementation choice: ctypes against
``/System/Library/Frameworks/Carbon.framework/Carbon`` rather than the
``quickmachotkey`` pip package. Registration returns noErr on macOS 15 /
Apple Silicon (verified by ``tools/hotkey_test.py``), needs **no TCC grant at
all**, and keeps the dependency surface at zero — so the panic hotkey can
never be silently dead behind a revoked permission.

Threading model — MAIN THREAD ONLY
----------------------------------
``register_hotkeys()`` registers on the CALLING thread (the main thread) and
spawns NO thread. Event delivery happens two ways, both on the main thread:

1. Anything that runs the normal Cocoa event machinery (``cv2.waitKey``,
   ``NSApp`` event fetching) dispatches queued Carbon events — including our
   hotkey events — to the installed handler as a side effect.
2. ``pump_events()`` drains the queue explicitly (non-blocking) and must be
   called regularly from the hot loop so hotkeys stay live even with
   ``--no-preview`` (nothing else pumping).

An earlier revision pumped ``ReceiveNextEvent`` on a dedicated background
thread. That works only while the process has no windows: the Carbon event
queue is process-wide, so once AppKit windows exist (indicator panel, cv2
preview) window events land in the same queue, get dispatched on the
background thread, trip AppKit's main-thread assertions, and the process
dies with SIGTRAP. Do not reintroduce the thread.

CALLBACK CONTRACT: ``toggle_cb`` / ``panic_cb`` run on the main thread from
whatever call pumped the queue, at an arbitrary point in the frame. They MUST
only set ``threading.Event`` flags — the loop polls the flags at a safe point
and does the real work. Never touch the camera, cv2, AppKit, or the synth
from a callback.

Hotkey spec strings: ``"ctrl+alt+g"``, ``"cmd+shift+escape"`` — one or more
modifiers (ctrl/control, alt/opt/option, cmd/command, shift) joined by ``+``
with a final key (letters, digits, escape/space/tab/return/delete, f1-f12).
At least one modifier is required: a bare key would swallow that key
system-wide.

Failure is loud: any nonzero OSStatus raises ``HotkeyError``. OSStatus
-9878 (``eventHotKeyExistsErr``) means the combination is reserved/conflicting
— pick a different spec in config.json. Note macOS lets ordinary apps
double-register the same combo without error (verified: two processes both
got noErr), so absence of -9878 is not proof of exclusivity.
"""
from __future__ import annotations

import ctypes
import threading
from ctypes import POINTER, byref, c_double, c_int32, c_ubyte, c_uint32, c_ulong, c_void_p
from typing import Any, Callable

from .config import HotkeyConfig

__all__ = ["HotkeyError", "parse_hotkey", "pump_events", "register_hotkeys"]


class HotkeyError(RuntimeError):
    """Hotkey registration or event-loop setup failed (never silent)."""


# --- Carbon constants -------------------------------------------------------

_kEventClassKeyboard = 0x6B657962   # 'keyb'
_kEventHotKeyPressed = 5
_kEventParamDirectObject = 0x2D2D2D2D  # '----'
_typeEventHotKeyID = 0x686B6964     # 'hkid'
_SIGNATURE = 0x474D484B             # 'GMHK' — tags our EventHotKeyIDs

_noErr = 0
_eventLoopTimedOutErr = -9875
_STATUS_HINTS = {
    -9878: "eventHotKeyExistsErr: combination already registered by another app",
    -9877: "eventHotKeyInvalidErr: invalid key/modifier combination",
    -50: "paramErr: bad parameter",
}

# Carbon modifier masks (Events.h).
_cmdKey = 0x0100
_shiftKey = 0x0200
_optionKey = 0x0800
_controlKey = 0x1000

_MODIFIERS = {
    "cmd": _cmdKey, "command": _cmdKey, "meta": _cmdKey,
    "shift": _shiftKey,
    "alt": _optionKey, "opt": _optionKey, "option": _optionKey,
    "ctrl": _controlKey, "control": _controlKey,
}

# Virtual keycodes (kVK_ANSI_* / kVK_* from HIToolbox Events.h). ANSI layout;
# hotkeys match by physical key position, which is what users expect.
_KEYCODES = {
    "a": 0x00, "s": 0x01, "d": 0x02, "f": 0x03, "h": 0x04, "g": 0x05,
    "z": 0x06, "x": 0x07, "c": 0x08, "v": 0x09, "b": 0x0B, "q": 0x0C,
    "w": 0x0D, "e": 0x0E, "r": 0x0F, "y": 0x10, "t": 0x11,
    "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "6": 0x16, "5": 0x17,
    "9": 0x19, "7": 0x1A, "8": 0x1C, "0": 0x1D,
    "o": 0x1F, "u": 0x20, "i": 0x22, "p": 0x23, "l": 0x25, "j": 0x26,
    "k": 0x28, "n": 0x2D, "m": 0x2E,
    "return": 0x24, "enter": 0x24, "tab": 0x30, "space": 0x31,
    "delete": 0x33, "backspace": 0x33, "escape": 0x35, "esc": 0x35,
    "f1": 0x7A, "f2": 0x78, "f3": 0x63, "f4": 0x76, "f5": 0x60, "f6": 0x61,
    "f7": 0x62, "f8": 0x64, "f9": 0x65, "f10": 0x6D, "f11": 0x67, "f12": 0x6F,
}


class _EventHotKeyID(ctypes.Structure):
    _fields_ = [("signature", c_uint32), ("id", c_uint32)]


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("eventClass", c_uint32), ("eventKind", c_uint32)]


# OSStatus (*)(EventHandlerCallRef, EventRef, void *userData)
_EventHandlerProc = ctypes.CFUNCTYPE(c_int32, c_void_p, c_void_p, c_void_p)

_carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")

_carbon.GetEventDispatcherTarget.restype = c_void_p
_carbon.GetEventDispatcherTarget.argtypes = []
_carbon.InstallEventHandler.restype = c_int32
_carbon.InstallEventHandler.argtypes = [
    c_void_p, _EventHandlerProc, c_ulong, POINTER(_EventTypeSpec),
    c_void_p, POINTER(c_void_p),
]
_carbon.RegisterEventHotKey.restype = c_int32
_carbon.RegisterEventHotKey.argtypes = [
    c_uint32, c_uint32, _EventHotKeyID, c_void_p, c_uint32, POINTER(c_void_p),
]
_carbon.GetEventParameter.restype = c_int32
_carbon.GetEventParameter.argtypes = [
    c_void_p, c_uint32, c_uint32, POINTER(c_uint32),
    c_ulong, POINTER(c_ulong), c_void_p,
]
_carbon.ReceiveNextEvent.restype = c_int32
_carbon.ReceiveNextEvent.argtypes = [
    c_ulong, POINTER(_EventTypeSpec), c_double, c_ubyte, POINTER(c_void_p),
]
_carbon.SendEventToEventTarget.restype = c_int32
_carbon.SendEventToEventTarget.argtypes = [c_void_p, c_void_p]
_carbon.ReleaseEvent.restype = None
_carbon.ReleaseEvent.argtypes = [c_void_p]


def parse_hotkey(spec: str) -> tuple[int, int]:
    """Parse ``"ctrl+alt+g"`` -> ``(keycode, carbon_modifier_mask)``."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if len(parts) < 2:
        raise ValueError(
            f"hotkey {spec!r}: need at least one modifier plus a key, e.g. 'ctrl+alt+g'"
        )
    *mods, key = parts
    modifiers = 0
    for mod in mods:
        if mod not in _MODIFIERS:
            raise ValueError(
                f"hotkey {spec!r}: unknown modifier {mod!r} "
                f"(use {sorted(set(_MODIFIERS))})"
            )
        modifiers |= _MODIFIERS[mod]
    if key not in _KEYCODES:
        raise ValueError(f"hotkey {spec!r}: unknown key {key!r}")
    return _KEYCODES[key], modifiers


# Module-level keep-alives: the handler UPP and EventHotKeyRefs must outlive
# the C references Carbon holds, or the process crashes on the next event.
_lock = threading.Lock()
_callbacks: dict[int, Callable[[], None]] = {}
_hotkey_refs: list[c_void_p] = []
_handler_upp: Any = None
_registered = False
_dispatch_target: c_void_p | None = None

# kEventDurationNoWait: return immediately if the queue is empty.
_kEventDurationNoWait = 0.0


def _dispatch(_call_ref: int, event_ref: int, _user_data: int) -> int:
    """Installed Carbon handler: route the fired hotkey ID to its callback."""
    hkid = _EventHotKeyID()
    err = _carbon.GetEventParameter(
        event_ref, _kEventParamDirectObject, _typeEventHotKeyID,
        None, ctypes.sizeof(hkid), None, byref(hkid),
    )
    if err == _noErr and hkid.signature == _SIGNATURE:
        callback = _callbacks.get(hkid.id)
        if callback is not None:
            try:
                callback()  # contract: flag-setter only, must not block/raise
            except Exception:
                pass  # a broken callback must never kill the event pump
    return _noErr


def register_hotkeys(
    toggle_cb: Callable[[], None],
    panic_cb: Callable[[], None],
    cfg: HotkeyConfig,
) -> None:
    """Register cfg.toggle / cfg.panic globally; raise HotkeyError on failure.

    Main thread only, no thread spawned (see module docstring). Registration
    failures (e.g. OSStatus -9878 conflicts) raise here at startup, never as
    a silently dead panic key. Call ``pump_events()`` from the hot loop so the
    callbacks (main-thread flag-setters) actually fire.
    """
    global _handler_upp, _registered, _dispatch_target
    with _lock:
        if _registered:
            raise HotkeyError("hotkeys already registered in this process")
        entries: list[tuple[int, int, int, str]] = []
        for hotkey_id, (spec_str, callback) in enumerate(
            ((cfg.toggle, toggle_cb), (cfg.panic, panic_cb)), start=1
        ):
            keycode, modifiers = parse_hotkey(spec_str)
            _callbacks[hotkey_id] = callback
            entries.append((hotkey_id, keycode, modifiers, spec_str))
        if _handler_upp is None:
            _handler_upp = _EventHandlerProc(_dispatch)

        target = c_void_p(_carbon.GetEventDispatcherTarget())
        handler_ref = c_void_p()
        spec = _EventTypeSpec(_kEventClassKeyboard, _kEventHotKeyPressed)
        err = _carbon.InstallEventHandler(
            target, _handler_upp, 1, byref(spec), None, byref(handler_ref),
        )
        if err != _noErr:
            raise HotkeyError(f"InstallEventHandler failed: OSStatus {err}")
        for hotkey_id, keycode, modifiers, spec_str in entries:
            ref = c_void_p()
            err = _carbon.RegisterEventHotKey(
                keycode, modifiers, _EventHotKeyID(_SIGNATURE, hotkey_id),
                target, 0, byref(ref),
            )
            if err != _noErr:
                hint = _STATUS_HINTS.get(err, "")
                raise HotkeyError(
                    f"RegisterEventHotKey({spec_str!r}) failed: OSStatus {err}"
                    + (f" ({hint})" if hint else "")
                )
            _hotkey_refs.append(ref)
        _dispatch_target = target
        _registered = True


def pump_events() -> None:
    """Drain pending Carbon events on the calling (main) thread, dispatching
    each to the event dispatcher target — this is what delivers hotkey
    presses to the registered callbacks. Non-blocking; call once per loop
    tick. Safe no-op before registration.

    Dispatching on the main thread is the same thing RunApplicationEventLoop
    (or Cocoa's own event fetching) does; the SIGTRAP class this replaces came
    from doing it on a background thread once windows existed.
    """
    if _dispatch_target is None:
        return
    while True:
        event = c_void_p()
        err = _carbon.ReceiveNextEvent(
            0, None, _kEventDurationNoWait, 1, byref(event)
        )
        if err != _noErr or not event:
            break
        _carbon.SendEventToEventTarget(event, _dispatch_target)
        _carbon.ReleaseEvent(event)
