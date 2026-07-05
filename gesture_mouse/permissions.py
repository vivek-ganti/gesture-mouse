"""TCC permission preflight: Camera + Accessibility (the full v1 surface).

IMPORTANT — grants attach to the HOST binary. During development that is the
app hosting the Python process (Terminal, iTerm2, VS Code, ...), not this
project: macOS attributes the unbundled interpreter to its "responsible
process". Switching terminal apps means granting again for the new host, and
a revoked grant hits every Python script that host runs.

Two grants are required:
- Camera (AVFoundation): prompted automatically on first capture; we prompt
  explicitly in ``preflight`` so the checklist can refuse a half-permissioned
  start before the tracking loop opens the device.
- Accessibility (ApplicationServices AXIsProcessTrusted*): required for
  CGEvent synthesis to actually land. There is no programmatic "request" —
  the prompt merely adds the host app (unchecked) to System Settings ->
  Privacy & Security -> Accessibility; the user must flip the switch.

Global hotkeys (Carbon) need no TCC grant at all — the panic key works even
half-permissioned. Hotkey conflicts are surfaced by ``hotkeys.register_hotkeys``
raising, not checked here.

Read-only by design: ``camera_status()`` and ``accessibility_status(False)``
never trigger prompts, so headless checks are safe.
"""
from __future__ import annotations

import subprocess
import threading

from ApplicationServices import (  # type: ignore[import-untyped]
    AXIsProcessTrusted,
    AXIsProcessTrustedWithOptions,
    kAXTrustedCheckOptionPrompt,
)
from AVFoundation import AVCaptureDevice, AVMediaTypeVideo  # type: ignore[import-untyped]

__all__ = [
    "camera_status",
    "request_camera",
    "accessibility_status",
    "deep_link",
    "preflight",
]

# AVAuthorizationStatus raw values.
_AV_STATUS = {0: "undetermined", 1: "restricted", 2: "denied", 3: "granted"}

_PANE_URLS = {
    "camera": "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera",
    "accessibility": (
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    ),
}


def camera_status() -> str:
    """Camera TCC state: 'granted' | 'denied' | 'restricted' | 'undetermined'.

    Never prompts.
    """
    raw = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
    return _AV_STATUS.get(int(raw), f"unknown({int(raw)})")


def request_camera(timeout_s: float = 120.0) -> bool:
    """Trigger the OS camera prompt (first run only); True if granted.

    The completion handler fires on a private dispatch queue, so this works
    headless with no runloop. If access was already determined the handler
    fires immediately with the existing verdict.
    """
    done = threading.Event()
    verdict = {"granted": False}

    def _handler(granted: bool) -> None:
        verdict["granted"] = bool(granted)
        done.set()

    AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVMediaTypeVideo, _handler
    )
    done.wait(timeout_s)  # user may ignore the dialog; treat timeout as denied
    return verdict["granted"]


def accessibility_status(prompt: bool = False) -> bool:
    """True if the host process is trusted for Accessibility (CGEvent posting).

    prompt=True shows the one-time system dialog and adds the host app
    (unchecked) to the Accessibility pane; it still returns the CURRENT
    verdict immediately — the user flips the switch out-of-band.
    """
    if prompt:
        return bool(
            AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
        )
    return bool(AXIsProcessTrusted())


def deep_link(pane: str) -> None:
    """Open System Settings on the given pane: 'camera' or 'accessibility'."""
    try:
        url = _PANE_URLS[pane]
    except KeyError:
        raise ValueError(f"unknown pane {pane!r}; expected one of {sorted(_PANE_URLS)}")
    subprocess.run(["open", url], check=False)


def automation_status(prompt: bool = False) -> str:
    """System Events automation consent: 'granted' | 'denied' | 'unknown'.

    Space switching and tab switching are delivered via AppleScript System
    Events keystrokes (raw CGEvent chords provably don't trigger macOS's
    Spaces switcher — see synth.py). That path needs the one-time Automation
    consent ("... wants to control System Events"). prompt=True runs a
    harmless System Events call so the dialog appears NOW, at startup,
    rather than the first time a swipe fires mid-session.
    """
    import subprocess

    if not prompt:
        return "unknown"  # no read-only query exists that never prompts
    try:
        r = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to count processes'],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "unknown"  # dialog left unanswered
    if r.returncode == 0:
        return "granted"
    return "denied"  # typically AppleEvent error -1743


def preflight(require: bool = True) -> bool:
    """Print the permission checklist; True only if the REQUIRED items
    (camera + accessibility) are granted.

    require=True additionally triggers the OS prompts for whatever is missing
    (camera consent dialog; Accessibility "add to list" dialog; the System
    Events Automation consent). require=False is strictly read-only — safe
    for headless/CI runs.

    Automation is reported but NOT gating: without it the cursor, clicks,
    scroll and Launchpad/Mission Control still work — only the keystroke-
    based gestures (Spaces switch, tab switch) would silently no-op, which
    the checklist calls out instead.
    """
    cam = camera_status()
    if require and cam == "undetermined":
        print("Requesting camera access (watch for the macOS dialog)...")
        request_camera()
        cam = camera_status()

    ax_ok = accessibility_status(prompt=False)
    if require and not ax_ok:
        # Shows the one-time dialog / adds the host app to the pane list.
        accessibility_status(prompt=True)

    auto = automation_status(prompt=require)

    cam_ok = cam == "granted"
    print("\ngesture-mouse permission preflight")
    print("(grants attach to the app hosting Python: Terminal/iTerm/IDE)")
    print(f"  [{'ok' if cam_ok else '!!'}] Camera         {cam}")
    if not cam_ok:
        print("       grant: System Settings -> Privacy & Security -> Camera")
        print(f"       open:  {_PANE_URLS['camera']}")
    print(f"  [{'ok' if ax_ok else '!!'}] Accessibility  {'granted' if ax_ok else 'not granted'}")
    if not ax_ok:
        print("       enable the host app: System Settings -> Privacy & Security -> Accessibility")
        print(f"       open:  {_PANE_URLS['accessibility']}")
    auto_tag = "ok" if auto == "granted" else ("!!" if auto == "denied" else "??")
    print(f"  [{auto_tag}] Automation     {auto}   (Spaces/tab switching via System Events)")
    if auto == "denied":
        print("       enable: System Settings -> Privacy & Security -> Automation")
        print("       -> your terminal app -> System Events")

    if cam_ok and ax_ok:
        print("  all required permissions granted.\n")
        return True
    print("  missing permissions — grant the items above and start again.\n")
    return False
