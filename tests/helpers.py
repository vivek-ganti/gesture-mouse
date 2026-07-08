"""Ergonomic LandmarkFrame fixture builders for engine tests (no numpy/camera).

A canonical right-hand 21-point template in PIXEL coordinates of a 640x480
mirrored frame. ``(x, y)`` positions the INDEX_MCP anchor; all other points
are template offsets (optionally scaled by ``hand_scale`` to fake depth/yaw
changes — the anchor stays put so only ``LandmarkFrame.scale`` moves).

Poses: ``pointer`` (index extended, others curled — the clutch pose),
``open`` (all five extended), ``fist``, ``relaxed`` (half-curled: maintains
POINTER but matches no pose), ``scroll`` (index+middle extended & together,
ring+pinky folded), ``right_click`` (pointer with a half-extended middle so
a thumb-middle pinch is geometrically reachable).

Pinches place the thumb tip at an exact normalized distance ``d`` from the
index (``pinch_index``) or middle (``pinch_middle``) fingertip, so
``dist(4,8)/scale`` (resp. ``dist(4,12)/scale``) equals ``d`` exactly. An
explicit ``thumb=(ox, oy)`` offset overrides both (used for transit paths).

``Seq`` builds 30fps sequences with correct ``ts_ms``: ``hold(pose, ms)``,
``move_to(xy, ms)``, ``pinch_to(d, ms)``, ``lose_hand(ms)`` etc.

test_palm.py keeps its own inline builders — do not merge them here.
"""
from __future__ import annotations

from gesture_mouse.types import LandmarkFrame, Point

IMG_W, IMG_H = 640, 480
FPS = 30.0
STEP_MS = 1000.0 / FPS
HAND_SPAN = 80.0          # dist(INDEX_MCP, PINKY_MCP) == LandmarkFrame.scale
PINCH_OPEN = 1.3          # normalized thumb-index distance when released

# Template offsets relative to the INDEX_MCP anchor (pixel units, y-down).
_WRIST_OFF = (30.0, 95.0)
_THUMB_BASE = {1: (5.0, 75.0), 2: (-5.0, 50.0), 3: (-10.0, 30.0)}
_MCP_OFF = {"index": (0.0, 0.0), "middle": (27.0, 0.0),
            "ring": (54.0, 0.0), "pinky": (80.0, 0.0)}
_FINGER_IDS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
               "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
# (pip, dip, tip) offsets relative to the finger's own MCP.
_JOINTS = {
    "ext": ((0.0, -35.0), (0.0, -60.0), (0.0, -80.0)),
    "curl": ((0.0, -18.0), (0.0, -8.0), (0.0, 10.0)),
    "half": ((0.0, -25.0), (0.0, -30.0), (0.0, -30.0)),
}
# pose -> (finger states, default thumb-tip offset)
_POSES = {
    "pointer": ({"index": "ext", "middle": "curl", "ring": "curl",
                 "pinky": "curl"}, (-20.0, 30.0)),
    "right_click": ({"index": "ext", "middle": "half", "ring": "curl",
                     "pinky": "curl"}, (-20.0, 30.0)),
    "scroll": ({"index": "ext", "middle": "ext", "ring": "curl",
                "pinky": "curl"}, (-20.0, 30.0)),
    "open": ({"index": "ext", "middle": "ext", "ring": "ext",
              "pinky": "ext"}, (-45.0, -5.0)),
    "fist": ({"index": "curl", "middle": "curl", "ring": "curl",
              "pinky": "curl"}, (15.0, 45.0)),
    "relaxed": ({"index": "half", "middle": "half", "ring": "half",
                 "pinky": "half"}, (-20.0, 30.0)),
    "horns": ({"index": "ext", "middle": "curl", "ring": "curl",
               "pinky": "ext"}, (-20.0, 30.0)),
    # Mid-formation rock sign: pinky already up, middle still descending
    # ("half" reads as extended under the angle metric — collinear joints).
    # Used to prove a fake thumb-middle pinch during the transition cannot
    # confirm a right click (the pinky-curled requirement).
    "horns_forming": ({"index": "ext", "middle": "half", "ring": "curl",
                       "pinky": "ext"}, (-20.0, 30.0)),
}


def _tip_off(pose: str, finger: str) -> tuple[float, float]:
    states, _ = _POSES[pose]
    mx, my = _MCP_OFF[finger]
    tx, ty = _JOINTS[states[finger]][2]
    return (mx + tx, my + ty)


def thumb_offset(pose: str, pinch_index: float | None = None,
                 pinch_middle: float | None = None,
                 thumb: tuple[float, float] | None = None) -> tuple[float, float]:
    """Resolve the effective thumb-tip offset for the given parameters."""
    if thumb is not None:
        return thumb
    if pinch_index is not None:
        ix, iy = _tip_off(pose, "index")
        return (ix, iy + pinch_index * HAND_SPAN)
    if pinch_middle is not None:
        mx, my = _tip_off(pose, "middle")
        return (mx, my + pinch_middle * HAND_SPAN)
    return _POSES[pose][1]


def make_frame(ts_ms: float, x: float = 320.0, y: float = 240.0, *,
               pose: str = "pointer", pinch_index: float | None = None,
               pinch_middle: float | None = None,
               thumb: tuple[float, float] | None = None,
               hand_scale: float = 1.0, handedness: str | None = "Right",
               confidence: float = 0.9) -> LandmarkFrame:
    states, _ = _POSES[pose]
    off: dict[int, tuple[float, float]] = {0: _WRIST_OFF}
    off.update(_THUMB_BASE)
    off[4] = thumb_offset(pose, pinch_index, pinch_middle, thumb)
    for finger, (mcp, pip, dip, tip) in _FINGER_IDS.items():
        mx, my = _MCP_OFF[finger]
        joints = _JOINTS[states[finger]]
        off[mcp] = (mx, my)
        for idx, (jx, jy) in zip((pip, dip, tip), joints):
            off[idx] = (mx + jx, my + jy)
    pts = tuple(
        Point(x + off[i][0] * hand_scale, y + off[i][1] * hand_scale)
        for i in range(21)
    )
    return LandmarkFrame(ts_ms=ts_ms, handedness=handedness, landmarks=pts,
                         img_w=IMG_W, img_h=IMG_H, confidence=confidence,
                         scale=HAND_SPAN * hand_scale, source="fixture")


def lost_frame(ts_ms: float) -> LandmarkFrame:
    return LandmarkFrame(ts_ms=ts_ms, handedness=None, landmarks=None,
                         img_w=IMG_W, img_h=IMG_H, confidence=0.0,
                         scale=0.0, source="fixture")


class Seq:
    """30fps LandmarkFrame sequence builder with correct ts_ms."""

    def __init__(self, x: float = 320.0, y: float = 240.0,
                 pose: str = "pointer", start_ms: float = 0.0,
                 handedness: str | None = "Right",
                 confidence: float = 0.9) -> None:
        self.frames: list[LandmarkFrame] = []
        self.ts = start_ms
        self.x, self.y = x, y
        self.pose = pose
        self.handedness = handedness
        self.confidence = confidence
        self.pinch_index: float | None = None
        self.pinch_middle: float | None = None
        self.thumb: tuple[float, float] | None = None
        self.hand_scale = 1.0

    @staticmethod
    def _n(ms: float) -> int:
        return max(1, round(ms / STEP_MS))

    def _emit(self) -> None:
        self.frames.append(make_frame(
            self.ts, self.x, self.y, pose=self.pose,
            pinch_index=self.pinch_index, pinch_middle=self.pinch_middle,
            thumb=self.thumb, hand_scale=self.hand_scale,
            handedness=self.handedness, confidence=self.confidence))
        self.ts += STEP_MS

    def hold(self, pose: str | None = None, ms: float = 100.0) -> "Seq":
        if pose is not None:
            self.pose = pose
        for _ in range(self._n(ms)):
            self._emit()
        return self

    def move_to(self, xy: tuple[float, float], ms: float = 100.0) -> "Seq":
        n = self._n(ms)
        x0, y0 = self.x, self.y
        for i in range(1, n + 1):
            self.x = x0 + (xy[0] - x0) * i / n
            self.y = y0 + (xy[1] - y0) * i / n
            self._emit()
        return self

    def pinch_to(self, d: float, ms: float = 66.0) -> "Seq":
        """Thumb-index pinch to normalized distance d (linear approach)."""
        start = self.pinch_index if self.pinch_index is not None else PINCH_OPEN
        self.pinch_middle = None
        self.thumb = None
        n = self._n(ms)
        for i in range(1, n + 1):
            self.pinch_index = start + (d - start) * i / n
            self._emit()
        return self

    def pinch_middle_to(self, d: float, ms: float = 66.0) -> "Seq":
        start = self.pinch_middle if self.pinch_middle is not None else 1.0
        self.pinch_index = None
        self.thumb = None
        n = self._n(ms)
        for i in range(1, n + 1):
            self.pinch_middle = start + (d - start) * i / n
            self._emit()
        return self

    def release_pinch(self, ms: float = 66.0) -> "Seq":
        """Open the hand back to the pose's neutral thumb."""
        if self.pinch_index is not None:
            self.pinch_to(PINCH_OPEN, ms)
            self.pinch_index = None
        elif self.pinch_middle is not None:
            self.pinch_middle_to(1.0, ms)
            self.pinch_middle = None
        elif self.thumb is not None:
            self.thumb_to(_POSES[self.pose][1], ms)
            self.thumb = None
        return self

    def thumb_to(self, offset: tuple[float, float], ms: float = 66.0) -> "Seq":
        """Drive the thumb tip along a straight template-space path."""
        sx, sy = thumb_offset(self.pose, self.pinch_index,
                              self.pinch_middle, self.thumb)
        self.pinch_index = None
        self.pinch_middle = None
        n = self._n(ms)
        for i in range(1, n + 1):
            self.thumb = (sx + (offset[0] - sx) * i / n,
                          sy + (offset[1] - sy) * i / n)
            self._emit()
        return self

    def scale_to(self, hand_scale: float, ms: float = 100.0) -> "Seq":
        """Fake a yaw/depth transient: hand scale ramps, anchor stays put."""
        s0 = self.hand_scale
        n = self._n(ms)
        for i in range(1, n + 1):
            self.hand_scale = s0 + (hand_scale - s0) * i / n
            self._emit()
        return self

    def lose_hand(self, ms: float = 100.0) -> "Seq":
        for _ in range(self._n(ms)):
            self.frames.append(lost_frame(self.ts))
            self.ts += STEP_MS
        return self
