#!/usr/bin/env python
"""Sweep One Euro mincutoff/beta over a recorded fixture; print lag/jitter.

Usage:
    .venv/bin/python tools/tune.py recording.jsonl
    .venv/bin/python tools/tune.py recording.jsonl \
        --mincutoff 0.5 1.0 1.5 2.0 --beta 0.0 0.004 0.007 0.02
    .venv/bin/python tools/tune.py fixtures/pointer-move.jsonl --noise-px 1.5

For every (mincutoff, beta) combination the fixture is replayed through the
REAL CursorPipeline and compared against the unfiltered control-box mapping
of the same anchor:

  jitter_px  RMS frame-to-frame output motion while the raw anchor is nearly
             still (< 20 px/s in screen space) — what you see as shimmer.
  lag_px     mean |output - raw| while the raw anchor moves fast
             (> 200 px/s) — how far the cursor trails the hand.
  lag_ms     the same lag expressed in time (lag_px / raw speed).

Casiez tuning procedure (One Euro, CHI 2012): (1) set beta = 0 and pick the
smallest mincutoff that kills jitter at rest; (2) raise beta until fast
motion stops lagging. Synthetic fixtures are noiseless — add --noise-px ~1.5
(deterministic, seeded) to make the jitter column meaningful, or record a
real fixture with tools/record.py.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from random import Random

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from gesture_mouse.config import Config  # noqa: E402
from gesture_mouse.filters import CursorPipeline  # noqa: E402
from gesture_mouse.tracker import ReplayTracker  # noqa: E402
from gesture_mouse.types import INDEX_MCP, LandmarkFrame, Point  # noqa: E402

SCREEN_W, SCREEN_H = 1440.0, 900.0
STILL_SPEED_PX_S = 20.0   # raw speed below this counts as "at rest"
FAST_SPEED_PX_S = 200.0   # raw speed above this counts as "moving"


def load_frames(
    path: str, noise_px: float
) -> list[tuple[LandmarkFrame, LandmarkFrame]]:
    """(clean, noisy) pairs. The clean frame is the motion ground truth
    (stillness classification, lag reference); the noisy one feeds the
    filter. Noise is seeded so every parameter combo sees identical input."""
    tracker = ReplayTracker(path)
    rng = Random(42)
    frames: list[tuple[LandmarkFrame, LandmarkFrame]] = []
    try:
        while (frame := tracker.read()) is not None:
            noisy = frame
            if noise_px > 0.0 and frame.hand_present:
                lm = list(frame.landmarks)
                a = lm[INDEX_MCP]
                lm[INDEX_MCP] = Point(a.x + rng.gauss(0.0, noise_px),
                                      a.y + rng.gauss(0.0, noise_px))
                noisy = LandmarkFrame(
                    ts_ms=frame.ts_ms, handedness=frame.handedness,
                    landmarks=tuple(lm), img_w=frame.img_w, img_h=frame.img_h,
                    confidence=frame.confidence, scale=frame.scale,
                    source=frame.source)
            frames.append((frame, noisy))
    finally:
        tracker.close()
    return frames


def raw_map(frame: LandmarkFrame, cfg: Config) -> tuple[float, float]:
    """Unfiltered control-box mapping of the anchor (mirrors CursorPipeline)."""
    box = cfg.control_box
    bx, by = box.x * frame.img_w, box.y * frame.img_h
    bw = max(box.w * frame.img_w, 1e-6)
    bh = max(box.h * frame.img_h, 1e-6)
    a = frame.landmarks[INDEX_MCP]
    cx = min(max(a.x, bx), bx + bw)
    cy = min(max(a.y, by), by + bh)
    return ((cx - bx) / bw * SCREEN_W, (cy - by) / bh * SCREEN_H)


def evaluate(frames: list[tuple[LandmarkFrame, LandmarkFrame]],
             mincutoff: float, beta: float,
             base: Config) -> tuple[float, float, float]:
    cfg = Config()
    cfg.control_box = base.control_box
    cfg.one_euro.mincutoff = mincutoff
    cfg.one_euro.beta = beta
    pipe = CursorPipeline(cfg, SCREEN_W, SCREEN_H)

    prev_raw: tuple[float, float, float] | None = None  # ts, x, y (clean)
    prev_out: tuple[float, float] | None = None
    still_deltas: list[float] = []
    lag_px: list[float] = []
    lag_ms: list[float] = []
    for clean, noisy in frames:
        if not clean.hand_present:
            prev_raw = None
            prev_out = None
            pipe.update(noisy)
            continue
        rx, ry = raw_map(clean, cfg)
        out = pipe.update(noisy)
        raw_speed = 0.0
        if prev_raw is not None:
            dt_s = (clean.ts_ms - prev_raw[0]) / 1000.0
            if dt_s > 0.0:
                raw_speed = math.hypot(rx - prev_raw[1], ry - prev_raw[2]) / dt_s
        if prev_raw is not None and prev_out is not None:
            if raw_speed < STILL_SPEED_PX_S:
                still_deltas.append(math.hypot(out.x - prev_out[0],
                                               out.y - prev_out[1]))
            elif raw_speed > FAST_SPEED_PX_S:
                err = math.hypot(out.x - rx, out.y - ry)
                lag_px.append(err)
                lag_ms.append(err / raw_speed * 1000.0)
        prev_raw = (clean.ts_ms, rx, ry)
        prev_out = (out.x, out.y)

    jitter = (math.fsum(d * d for d in still_deltas) / len(still_deltas)) ** 0.5 \
        if still_deltas else float("nan")
    mean_lag_px = math.fsum(lag_px) / len(lag_px) if lag_px else float("nan")
    mean_lag_ms = math.fsum(lag_ms) / len(lag_ms) if lag_ms else float("nan")
    return jitter, mean_lag_px, mean_lag_ms


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep One Euro parameters over a JSONL fixture.")
    ap.add_argument("file", help="JSONL fixture (tools/record.py or tools/make_fixture.py)")
    ap.add_argument("--mincutoff", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--beta", type=float, nargs="+",
                    default=[0.0, 0.002, 0.007, 0.02])
    ap.add_argument("--noise-px", type=float, default=0.0,
                    help="add seeded Gaussian anchor noise (camera px) for synthetic fixtures")
    args = ap.parse_args()

    frames = load_frames(args.file, args.noise_px)
    hand = sum(1 for clean, _ in frames if clean.hand_present)
    print(f"{args.file}: {len(frames)} frames ({hand} with hand), "
          f"noise={args.noise_px:g}px")
    base = Config()

    print(f"{'mincutoff':>9}  {'beta':>7}  {'jitter_px':>9}  {'lag_px':>7}  {'lag_ms':>7}")
    for mc in args.mincutoff:
        for b in args.beta:
            jitter, lpx, lms = evaluate(frames, mc, b, base)
            print(f"{mc:9.3f}  {b:7.4f}  {jitter:9.3f}  {lpx:7.1f}  {lms:7.1f}")
    print("\nCasiez: beta=0, smallest mincutoff with acceptable jitter_px;")
    print("then raise beta until lag_ms is acceptable at speed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
