#!/usr/bin/env python
"""Generate synthetic JSONL fixtures from the tests/helpers.py builders.

Usage:
    .venv/bin/python tools/make_fixture.py click --out fixtures/click.jsonl
    .venv/bin/python tools/make_fixture.py all --out-dir fixtures

Kinds: pointer-move, click, double-click, drag, right-click, scroll,
palm-swipe-left/right/up/down, palm-pinch-in, palm-spread-out, all.

The sequences mirror the golden engine tests (tests/test_engine.py), so a
generated fixture replayed with `python -m gesture_mouse --replay FILE`
prints the same intent classes the tests assert. Frames use the canonical
640x480 mirrored-frame template hand at 30 fps; headers follow the frozen
JSONL convention (tracker.py Recorder).
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tests"))

from gesture_mouse.config import Config  # noqa: E402
from gesture_mouse.tracker import Recorder  # noqa: E402
from helpers import IMG_H, IMG_W, Seq  # noqa: E402

_CLUTCH_MS = 400.0  # pointer pose long past clutch.engage_ms (150) to settle


def seq_pointer_move() -> Seq:
    """Clutch in, then tour the control box (moves only, no pinches)."""
    s = Seq()
    s.hold("pointer", _CLUTCH_MS)
    s.move_to((450.0, 240.0), 400)
    s.move_to((450.0, 330.0), 300)
    s.move_to((250.0, 330.0), 400)
    s.move_to((320.0, 240.0), 300)
    s.hold(ms=200)
    return s


def seq_click() -> Seq:
    """Careful stationary click: exactly DOWN,UP with click_count=1."""
    s = Seq()
    s.hold("pointer", _CLUTCH_MS)
    s.pinch_to(0.12, 66)
    s.hold(ms=250)
    s.release_pinch(66)
    s.hold(ms=300)
    return s


def seq_double_click() -> Seq:
    """Two taps inside the 500 ms / 15 px window: second pair click_count=2."""
    s = Seq()
    s.hold("pointer", _CLUTCH_MS)
    for pause in (100.0, 300.0):
        s.pinch_to(0.12, 66)
        s.hold(ms=250)
        s.release_pinch(66)
        s.hold(ms=pause)
    return s


def seq_drag() -> Seq:
    """Pinch, hold (frozen), pull +30 camera px in x: DOWN, drag MOVEs, UP."""
    s = Seq()
    s.hold("pointer", _CLUTCH_MS)
    s.pinch_to(0.12, 66)
    s.hold(ms=200)
    s.move_to((350.0, 240.0), 200)
    s.hold(ms=300)
    s.release_pinch(66)
    s.hold(ms=300)
    return s


def seq_right_click() -> Seq:
    """Thumb transits past the extended index to the middle tip: right
    DOWN/UP only — the transit must NOT produce a left click."""
    s = Seq()
    s.hold("pointer", 300)
    s.hold("right_click", 200)
    s.thumb_to((0.0, -68.0), 133)
    s.hold(ms=66)
    s.thumb_to((27.0, -18.0), 100)
    s.hold(ms=400)
    s.release_pinch(66)
    s.hold(ms=300)
    return s


def seq_scroll() -> Seq:
    """Scroll pose settles, joystick deflects down, pose drops: scroll MOVEs
    with dy_px > 0 and no post-exit tail."""
    s = Seq()
    s.hold("pointer", _CLUTCH_MS)
    s.hold("scroll", 200)
    s.move_to((320.0, 300.0), 300)
    s.hold(ms=100)
    s.hold("pointer", ms=400)
    return s


def _seq_palm_swipe(dx: float, dy: float) -> Seq:
    """Arm-at-rest, then one fast flick (the new swipe model).

    Open palm held STILL ~250 ms arms the detector (needs the >=80 ms speed
    baseline plus arm_hold_ms=150 of open-palm-at-rest); the flick then only
    needs net displacement >= 0.25 of the frame span within 800 ms — the
    motion frames deliberately need no pose (blur kills it on real data).
    """
    s = Seq()
    s.hold("pointer", _CLUTCH_MS)
    s.hold("open", 250)                      # arming: open palm at rest
    s.move_to((320.0 + dx, 240.0 + dy), 150)  # the flick
    s.hold(ms=100)
    s.hold("pointer", ms=300)
    return s


def seq_palm_pinch_in() -> Seq:
    """Open palm (spread m~1.25) collapsing to a fist (m~0.38) inside the
    500 ms window -> the pinch_in binding (default: launchpad)."""
    s = Seq()
    s.hold("pointer", _CLUTCH_MS)
    s.hold("open", 200)
    s.hold("fist", 200)
    s.hold("pointer", ms=300)
    return s


def seq_palm_spread_out() -> Seq:
    """Fist (m~0.38 < spread_in_start) bursting open (m~1.25 > spread_out)
    -> the spread_out binding (default: show_desktop)."""
    s = Seq()
    s.hold("pointer", _CLUTCH_MS)
    s.hold("fist", 200)
    s.hold("open", 200)
    s.hold("pointer", ms=300)
    return s


KINDS: dict[str, object] = {
    "pointer-move": seq_pointer_move,
    "click": seq_click,
    "double-click": seq_double_click,
    "drag": seq_drag,
    "right-click": seq_right_click,
    "scroll": seq_scroll,
    "palm-swipe-left": lambda: _seq_palm_swipe(-200.0, 0.0),
    "palm-swipe-right": lambda: _seq_palm_swipe(200.0, 0.0),
    "palm-swipe-up": lambda: _seq_palm_swipe(0.0, -150.0),
    "palm-swipe-down": lambda: _seq_palm_swipe(0.0, 150.0),
    "palm-pinch-in": seq_palm_pinch_in,
    "palm-spread-out": seq_palm_spread_out,
}


def write_fixture(kind: str, path: Path) -> int:
    frames = KINDS[kind]().frames
    path.parent.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(str(path), {
        "source": f"synthetic:{kind}",
        "mirror": True,
        "rotate": 0,
        "img_w": IMG_W,
        "img_h": IMG_H,
        "config": dataclasses.asdict(Config()),
    })
    try:
        for frame in frames:
            recorder.write(frame)
    finally:
        recorder.close()
    return len(frames)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate synthetic gesture fixtures (JSONL).")
    ap.add_argument("kind", choices=[*KINDS, "all"])
    ap.add_argument("--out", metavar="FILE",
                    help="output path (single kind only; default <out-dir>/<kind>.jsonl)")
    ap.add_argument("--out-dir", metavar="DIR", default="fixtures",
                    help="output directory (default: fixtures/)")
    args = ap.parse_args()

    kinds = list(KINDS) if args.kind == "all" else [args.kind]
    if args.out is not None and len(kinds) > 1:
        ap.error("--out is only valid with a single kind")
    for kind in kinds:
        path = Path(args.out) if args.out else Path(args.out_dir) / f"{kind}.jsonl"
        n = write_fixture(kind, path)
        print(f"wrote {n:4d} frames: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
