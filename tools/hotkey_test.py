#!/usr/bin/env python
"""Self-test for gesture_mouse.hotkeys: register the configured hotkeys and
print when they fire.

Usage:
    .venv/bin/python tools/hotkey_test.py [seconds]

Registers the toggle/panic hotkeys from config.json (defaults ctrl+alt+g /
ctrl+alt+escape) and polls for presses for [seconds] (default: until Ctrl+C).
Successful registration prints "registration OK"; a nonzero OSStatus raises
HotkeyError (-9878 = eventHotKeyExistsErr for reserved/conflicting combos;
note macOS lets ordinary apps double-register the same combo without error).

Models the real wiring: callbacks only set threading.Event flags; this main
thread polls the flags and prints. No TCC permission is required.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from gesture_mouse.config import ConfigStore  # noqa: E402
from gesture_mouse.hotkeys import pump_events, register_hotkeys  # noqa: E402


def main() -> int:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else None
    cfg = ConfigStore(_ROOT / "config.json").config.hotkeys

    toggle_flag = threading.Event()
    panic_flag = threading.Event()
    register_hotkeys(toggle_flag.set, panic_flag.set, cfg)
    print(f"registration OK: toggle={cfg.toggle!r} panic={cfg.panic!r}")
    if duration is None:
        print("press the hotkeys to test; Ctrl+C to quit")

    deadline = None if duration is None else time.monotonic() + duration
    try:
        while deadline is None or time.monotonic() < deadline:
            pump_events()  # deliveries happen here, on the main thread
            if toggle_flag.is_set():
                toggle_flag.clear()
                print(f"TOGGLE pressed ({cfg.toggle})")
            if panic_flag.is_set():
                panic_flag.clear()
                print(f"PANIC pressed ({cfg.panic})")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
