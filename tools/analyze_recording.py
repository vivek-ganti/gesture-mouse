#!/usr/bin/env python
"""Analyze a recorded JSONL clip end-to-end: detection quality + gesture output.

This is how the pipeline gets verified without a live camera on the dev
machine — a user records a clip on a camera-authorized host, and this replays
it through the real filters + engine, reporting:

  - detection: frame count, fps, % frames with a hand, mean/median confidence,
    handedness split, mean hand scale (px)
  - poses seen: how many frames matched pointer / open-palm / scroll poses
  - pinch: min normalized thumb-index and thumb-middle distances reached
  - intents: the full Intent stream the engine produced

If "hand present" is high and poses/pinches show up, detection works; the
intent stream shows whether the gesture logic fired correctly.

Usage:
    .venv/bin/python tools/analyze_recording.py clip.jsonl
"""
from __future__ import annotations

import statistics
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from gesture_mouse.config import ConfigStore  # noqa: E402
from gesture_mouse.engine import GestureEngine  # noqa: E402
from gesture_mouse.filters import CursorPipeline  # noqa: E402
from gesture_mouse.palm import PalmDetector  # noqa: E402
from gesture_mouse.tracker import ReplayTracker  # noqa: E402
from gesture_mouse.types import (  # noqa: E402
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    dist,
)


def _ext(lm, tip, pip) -> bool:
    w = lm[WRIST]
    return dist(lm[tip], w) > 1.15 * dist(lm[pip], w)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    tracker = ReplayTracker(path)
    print(f"header: {tracker.header}")

    store = ConfigStore(str(_ROOT / "config.json"))
    cfg = store.config
    pipeline = CursorPipeline(cfg, 1440.0, 900.0)
    palm = PalmDetector(cfg.palm, cfg.bindings)
    engine = GestureEngine(cfg, pipeline.pinch, palm)

    n = present = 0
    confs: list[float] = []
    scales: list[float] = []
    hands: Counter[str] = Counter()
    poses = Counter()
    min_left = min_right = float("inf")
    intents: list = []
    first_ts = last_ts = None

    while True:
        frame = tracker.read()
        if frame is None:
            break
        n += 1
        if first_ts is None:
            first_ts = frame.ts_ms
        last_ts = frame.ts_ms
        cursor = pipeline.update(frame)
        out = engine.update(frame, cursor)
        intents.extend(out.intents)
        if frame.hand_present:
            present += 1
            confs.append(frame.confidence)
            scales.append(frame.scale)
            hands[frame.handedness or "?"] += 1
            lm = frame.landmarks
            idx = _ext(lm, INDEX_TIP, INDEX_PIP)
            mid = _ext(lm, MIDDLE_TIP, MIDDLE_PIP)
            rng = _ext(lm, RING_TIP, RING_PIP)
            pky = _ext(lm, PINKY_TIP, PINKY_PIP)
            if idx and not mid and not rng and not pky:
                poses["pointer"] += 1
            if idx and mid and rng and pky:
                poses["open_palm"] += 1
            if idx and mid and not rng and not pky:
                poses["scroll(2-finger)"] += 1
            if frame.scale > 0:
                min_left = min(min_left, dist(lm[THUMB_TIP], lm[INDEX_TIP]) / frame.scale)
                min_right = min(min_right, dist(lm[THUMB_TIP], lm[MIDDLE_TIP]) / frame.scale)

    dur_s = ((last_ts - first_ts) / 1000.0) if (first_ts is not None and last_ts) else 0.0
    fps = (n - 1) / dur_s if dur_s > 0 else 0.0

    print("\n=== DETECTION ===")
    print(f"frames: {n}   duration: {dur_s:.1f}s   fps: {fps:.1f}")
    pct = 100.0 * present / n if n else 0.0
    print(f"hand present: {present}/{n} ({pct:.0f}%)   handedness: {dict(hands)}")
    if confs:
        print(f"confidence: mean {statistics.mean(confs):.2f}  "
              f"median {statistics.median(confs):.2f}  min {min(confs):.2f}")
        print(f"hand scale px: mean {statistics.mean(scales):.0f}  "
              f"range {min(scales):.0f}-{max(scales):.0f}")
    print("\n=== POSES (frames matched) ===")
    print(f"pointer(index only): {poses['pointer']}   "
          f"open_palm(all 5): {poses['open_palm']}   "
          f"scroll(index+middle): {poses['scroll(2-finger)']}")
    print(f"pinch reach: min thumb-index {min_left:.2f} "
          f"(engage<{cfg.pinch.left_engage})   "
          f"min thumb-middle {min_right:.2f} (engage<{cfg.pinch.right_engage})")

    print("\n=== INTENTS ===")
    if not intents:
        print("(none — no gesture fired)")
    else:
        counts = Counter(f"{i.name}/{i.phase.value}" for i in intents)
        for k, c in counts.most_common():
            print(f"  {k}: {c}")

    print("\n=== VERDICT ===")
    want = cfg.hand.strip().lower()
    matched = (
        present if want == "auto"
        else sum(c for h, c in hands.items() if h.lower() == want)
    )
    if pct < 50:
        print("LOW hand-detection — hand often out of frame, too dark, or wrong camera.")
    elif present and matched < present * 0.5:
        print(
            f"Hand detected but labeled {dict(hands)} while config expects "
            f"{cfg.hand!r} — the engine will ignore it. Either set "
            f'"hand" in config.json to the reported label, or camera.mirror '
            f"is wrong for this camera (phone/rear cameras are unmirrored)."
        )
    elif confs and statistics.mean(confs) < 0.6:
        print("Hand detected but LOW confidence — improve lighting / hand distance.")
    else:
        print("Detection looks HEALTHY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
