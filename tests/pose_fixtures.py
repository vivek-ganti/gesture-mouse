"""Bent-joint hand fixtures for validating the angle-based pose classifier.

tests/helpers.py's ``_JOINTS`` keeps every finger's PIP/DIP/TIP perfectly
collinear with its MCP (by construction — see its own docstring: pip/dip/tip
offsets are all along the same ray from the MCP). That's sufficient for the
old tip-to-wrist RATIO test, but it can never exercise a genuine PIP bend:
MCP, PIP and TIP always sit on one straight line, so
``gesture_mouse.engine._pip_angle_deg`` only ever sees exactly 180 (deg,
extended) or 0 (deg, curled) on those fixtures — even ``helpers.py``'s
"half" state, meant to model an ambiguous half-curled finger, reads as a
full 180 (deg) under the angle metric, since its TIP still sits strictly
further from the MCP than its PIP, along the very same ray as "ext".

These builders place TIP at an explicit PIP-vertex angle instead, so tests
can exercise engine.py's angle-based ``_ext()``/``_FingerState`` machinery
against genuinely bent geometry that a real half-curled hand would produce.

Kept as its own module rather than merged into ``helpers.py``, per that
module's own "test_palm.py keeps its own inline builders — do not merge
them here" convention for fixture families with a different shape.
"""
from __future__ import annotations

import math

from gesture_mouse.types import LandmarkFrame, Point

IMG_W, IMG_H = 640, 480
HAND_SPAN = 80.0

_MCP_OFF = {"index": (0.0, 0.0), "middle": (27.0, 0.0),
            "ring": (54.0, 0.0), "pinky": (80.0, 0.0)}
_FINGER_IDS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
               "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
_WRIST_OFF = (30.0, 95.0)
_THUMB_OFF = {1: (5.0, 75.0), 2: (-5.0, 50.0), 3: (-10.0, 30.0), 4: (-20.0, 30.0)}
_PIP_LEN = 35.0
_TIP_LEN = 45.0


def bent_finger_offsets(
    angle_deg: float, bend_dir: float = 1.0
) -> tuple[tuple[float, float], tuple[float, float]]:
    """PIP/TIP offsets (relative to the finger's own MCP) such that
    ``engine._pip_angle_deg`` returns exactly ``angle_deg`` at the PIP
    joint: 180 = straight/extended, 0 = folded fully back/curled. PIP sits
    straight "up" from MCP by a fixed length; TIP is placed by rotating the
    PIP->MCP direction by ``angle_deg`` (``bend_dir`` only flips which way
    the finger bends sideways — cosmetic, doesn't affect the angle)."""
    pip = (0.0, -_PIP_LEN)
    ux, uy = 0.0, 1.0  # PIP -> MCP direction, already unit length
    theta = math.radians(angle_deg) * bend_dir
    vx = ux * math.cos(theta) - uy * math.sin(theta)
    vy = ux * math.sin(theta) + uy * math.cos(theta)
    tip = (pip[0] + vx * _TIP_LEN, pip[1] + vy * _TIP_LEN)
    return pip, tip


def make_bent_frame(
    ts_ms: float,
    angles: dict[str, float],
    x: float = 320.0,
    y: float = 240.0,
    confidence: float = 0.9,
) -> LandmarkFrame:
    """A full 21-point frame where each finger named in ``angles`` (e.g.
    ``{"index": 90.0}``) is placed with a genuine PIP bend of that many
    degrees via :func:`bent_finger_offsets`. Fingers not named default to
    fully curled (0 deg). DIP is placed at the PIP/TIP midpoint — no pose
    test in engine.py reads DIP, so its exact position is cosmetic."""
    off: dict[int, tuple[float, float]] = {0: _WRIST_OFF}
    off.update(_THUMB_OFF)
    for finger, (mcp_i, pip_i, dip_i, tip_i) in _FINGER_IDS.items():
        mx, my = _MCP_OFF[finger]
        pip_off, tip_off = bent_finger_offsets(angles.get(finger, 0.0))
        pip = (mx + pip_off[0], my + pip_off[1])
        tip = (mx + tip_off[0], my + tip_off[1])
        dip = ((pip[0] + tip[0]) / 2.0, (pip[1] + tip[1]) / 2.0)
        off[mcp_i] = (mx, my)
        off[pip_i] = pip
        off[dip_i] = dip
        off[tip_i] = tip
    pts = tuple(Point(x + off[i][0], y + off[i][1]) for i in range(21))
    return LandmarkFrame(
        ts_ms=ts_ms, handedness="Right", landmarks=pts, img_w=IMG_W, img_h=IMG_H,
        confidence=confidence, scale=HAND_SPAN, source="fixture",
    )
