#!/usr/bin/env python
"""Replay a recorded JSONL fixture headless and print per-frame summaries.

Usage:
    .venv/bin/python tools/replay.py wave.jsonl

No camera, no MediaPipe: this only exercises ReplayTracker, so it runs
anywhere (CI, headless) and verifies fixtures byte-for-byte.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from gesture_mouse.tracker import ReplayTracker  # noqa: E402
from gesture_mouse.types import INDEX_MCP  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a JSONL landmark recording.")
    parser.add_argument("file", help="JSONL recording produced by tools/record.py")
    args = parser.parse_args()

    tracker = ReplayTracker(args.file)
    h = tracker.header
    print(f"header: source={h.get('source')!r} mirror={h.get('mirror')} "
          f"rotate={h.get('rotate')} dims={h.get('img_w')}x{h.get('img_h')}")

    frames = hand_frames = 0
    first_ts: float | None = None
    last_ts: float | None = None
    try:
        while (frame := tracker.read()) is not None:
            if frame.hand_present:
                anchor = frame.landmarks[INDEX_MCP]
                detail = (f"hand={frame.handedness:<5} conf={frame.confidence:.2f} "
                          f"scale={frame.scale:6.1f} anchor=({anchor.x:6.1f},{anchor.y:6.1f})")
                hand_frames += 1
            else:
                detail = "hand=absent"
            print(f"[{frames:4d}] ts={frame.ts_ms:9.1f}ms {frame.img_w}x{frame.img_h} {detail}")
            if first_ts is None:
                first_ts = frame.ts_ms
            last_ts = frame.ts_ms
            frames += 1
    finally:
        tracker.close()

    if frames == 0:
        print("totals: 0 frames")
        return 1
    span_s = (last_ts - first_ts) / 1000.0 if frames > 1 else 0.0
    fps = (frames - 1) / span_s if span_s > 0 else 0.0
    print(f"totals: {frames} frames ({hand_frames} with hand, "
          f"{frames - hand_frames} absent) over {span_s:.2f}s -> {fps:.1f} fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
