#!/usr/bin/env python
"""Record live LandmarkFrames from a camera to a JSONL fixture.

Usage:
    .venv/bin/python tools/record.py --camera "FaceTime HD Camera" --seconds 5 --out wave.jsonl

Opens the camera (this WILL trigger the macOS camera permission prompt on
first run), streams frames through CameraTracker, tees every frame to a
Recorder, and prints an fps summary. Replay with tools/replay.py — no camera
needed there.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from gesture_mouse.config import ConfigStore  # noqa: E402
from gesture_mouse.tracker import CameraTracker, Recorder, list_cameras  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Record hand-landmark frames to JSONL.")
    parser.add_argument("--camera", default=None,
                        help="camera name (default: config.json / first available)")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="recording duration in seconds (default: 5)")
    parser.add_argument("--out", default="recording.jsonl",
                        help="output JSONL path (default: recording.jsonl)")
    parser.add_argument("--show", action="store_true",
                        help="show the live camera image + detection overlay while recording")
    args = parser.parse_args()

    cfg = ConfigStore(_REPO_ROOT / "config.json").config
    if args.camera is not None:
        cfg.camera.name = args.camera

    cams = list_cameras()
    print(f"cameras: {cams}")
    print(f"recording {args.seconds:.1f}s from "
          f"{cfg.camera.name or (cams[0] if cams else 'index 0')} -> {args.out}")

    tracker = CameraTracker(cfg, model_path=str(_REPO_ROOT / "hand_landmarker.task"))
    tracker.open()

    recorder: Recorder | None = None
    frames = hand_frames = grab_failures = 0
    first_ts: float | None = None
    last_ts: float | None = None
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            frame = tracker.read()
            if frame is None:
                grab_failures += 1
                continue
            if recorder is None:
                # Header is written once real dims are known (first good grab).
                recorder = Recorder(args.out, {
                    "source": frame.source,
                    "mirror": cfg.camera.mirror,
                    "rotate": cfg.camera.rotate,
                    "img_w": frame.img_w,
                    "img_h": frame.img_h,
                    "config": dataclasses.asdict(cfg),
                })
            recorder.write(frame)
            frames += 1
            if frame.hand_present:
                hand_frames += 1
            if first_ts is None:
                first_ts = frame.ts_ms
            last_ts = frame.ts_ms
            if args.show and tracker.last_bgr is not None:
                import cv2

                img = tracker.last_bgr.copy()
                if frame.hand_present:
                    for p in frame.landmarks:
                        cv2.circle(img, (int(p.x), int(p.y)), 3, (0, 255, 0), -1)
                    banner = f"HAND: {frame.handedness} conf {frame.confidence:.2f}"
                    color = (0, 255, 0)
                else:
                    banner = "NO HAND"
                    color = (0, 0, 255)
                remaining = deadline - time.monotonic()
                cv2.putText(img, f"{banner}   rec {remaining:.0f}s left", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow("gesture-mouse recording", img)
                cv2.waitKey(1)
    finally:
        tracker.close()
        if recorder is not None:
            recorder.close()
        if args.show:
            import cv2

            cv2.destroyAllWindows()
            cv2.waitKey(1)

    if frames == 0:
        print(f"no frames captured ({grab_failures} grab failures) — nothing written")
        return 1
    span_s = (last_ts - first_ts) / 1000.0 if frames > 1 else 0.0
    fps = (frames - 1) / span_s if span_s > 0 else 0.0
    print(f"wrote {frames} frames ({hand_frames} with hand, {grab_failures} grab "
          f"failures) over {span_s:.2f}s -> {fps:.1f} fps: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
