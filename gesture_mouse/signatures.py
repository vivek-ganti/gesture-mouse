"""Pose signatures: the single source of truth for "what a pose is".

Pure logic (CONTRACTS.md global rules): no macOS / cv2 / mediapipe / numpy
imports, no wall-clock reads.

A ``Signature`` maps finger name -> ``"ext"`` | ``"curl"`` | ``"any"``; a
missing finger means ``"any"``. Only the four non-thumb fingers GATE
matching (``FINGERS``); the thumb is sampled and displayed (``ALL_FINGERS``)
but never gates in v1 — its hinge geometry is the least reliable of the
five (see engine.py ``_open_palm_pose`` docstring), so thumb entries in a
signature are normalized to ``"any"``.

Every built-in pose is a signature too (``BUILTINS``), evaluated by the
same engine matcher as user-defined gestures — one code path, so a gesture
captured in the panel behaves exactly like a shipped one. ``scroll``
additionally requires the fingertips-together distance test, which lives in
the engine (it is not a per-finger property).

Angles: ``pip_angle_deg`` is the interior angle at the middle joint —
180 deg = perfectly straight, 0 deg = folded back on itself. ``FingerLatch``
is the two-threshold hysteresis latch classifying one finger from that
angle (extend/curl thresholds come from config: per-finger calibrated
values with the global pair as fallback).
"""
from __future__ import annotations

import math

from .types import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    Point,
)

# Fingers that gate signature matching, in canonical order.
FINGERS: tuple[str, ...] = ("index", "middle", "ring", "pinky")
# Fingers that are sampled/displayed (calibration, panel readout).
ALL_FINGERS: tuple[str, ...] = ("thumb",) + FINGERS

Signature = dict  # finger name -> "ext" | "curl" | "any"; missing == "any"

# (mcp, pip, tip) landmark indices per finger; the thumb's "pip" is its IP
# joint (the thumb has no PIP — MCP/IP/TIP is its hinge chain).
FINGER_JOINTS: dict[str, tuple[int, int, int]] = {
    "thumb": (THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}

BUILTINS: dict[str, Signature] = {
    "pointer": {"index": "ext", "middle": "curl", "ring": "curl", "pinky": "curl"},
    "open_palm": {"index": "ext", "middle": "ext", "ring": "ext", "pinky": "ext"},
    # scroll ALSO requires dist(INDEX_TIP, MIDDLE_TIP)/scale < together_max,
    # enforced engine-side — the signature alone is deliberately reserved
    # (a captured custom gesture may not claim it; see check_conflicts).
    "scroll": {"index": "ext", "middle": "ext", "ring": "curl", "pinky": "curl"},
    "horns": {"index": "ext", "middle": "curl", "ring": "curl", "pinky": "ext"},
}

# Back-compat: config entries written as {"pose": "<name>", ...} before the
# signature schema existed. Only "horns" ever shipped.
LEGACY_POSES: dict[str, Signature] = {"horns": BUILTINS["horns"]}

_VALID_STATES = frozenset({"ext", "curl", "any"})


def pip_angle_deg(lm: tuple[Point, ...], mcp: int, pip: int, tip: int) -> float:
    """Interior angle at the PIP joint, degrees: 180 = perfectly straight
    (MCP, PIP, TIP colinear), shrinking toward 0 as the finger curls back on
    itself. Scale/rotation invariant (unlike a tip-to-wrist distance ratio,
    which is sensitive to hand tilt/yaw toward the camera)."""
    ax, ay = lm[mcp].x - lm[pip].x, lm[mcp].y - lm[pip].y  # PIP -> MCP
    bx, by = lm[tip].x - lm[pip].x, lm[tip].y - lm[pip].y  # PIP -> TIP
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na <= 1e-9 or nb <= 1e-9:
        return 180.0  # degenerate (coincident points): treat as straight
    cos_a = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.degrees(math.acos(cos_a))


def compute_finger_angles(lm: tuple[Point, ...]) -> dict[str, float]:
    """All five fingers' joint angles (thumb included — sampled for
    calibration/display even though it never gates matching)."""
    return {
        f: pip_angle_deg(lm, *FINGER_JOINTS[f]) for f in ALL_FINGERS
    }


class FingerLatch:
    """Two-threshold hysteresis latch for one finger's extended/curled
    state, replacing a single hard-cutoff test. Mirrors the engage/release
    pattern used by pinch detection: a finger only becomes "extended" once
    its angle clears the (high) extend threshold, and only reverts to
    "curled" once it drops below the (lower) curl threshold — a finger
    sitting between the two keeps whatever it last was, instead of
    flickering every frame the way a single instant cutoff does."""

    __slots__ = ("extended",)

    def __init__(self) -> None:
        self.extended = False

    def update(self, angle_deg: float, extend_at: float, curl_at: float) -> bool:
        if self.extended and angle_deg <= curl_at:
            self.extended = False
        elif not self.extended and angle_deg >= extend_at:
            self.extended = True
        return self.extended

    def reset(self) -> None:
        self.extended = False


def normalize_signature(raw: object) -> Signature | None:
    """Validate/canonicalize a raw signature dict. Returns None if invalid.

    Unknown finger names or states are invalid (never silently dropped —
    a typo like "indx" must not yield a match-anything gesture). Thumb is
    accepted but forced to "any" (non-gating in v1). A signature with no
    gating constraint at all ("any" across the board) is invalid — it
    would match every hand."""
    if not isinstance(raw, dict) or not raw:
        return None
    out: Signature = {}
    for finger, state in raw.items():
        if finger not in ALL_FINGERS or state not in _VALID_STATES:
            return None
        out[finger] = "any" if finger == "thumb" else state
    if not any(out.get(f) in ("ext", "curl") for f in FINGERS):
        return None
    return out


def signatures_conflict(a: Signature, b: Signature) -> bool:
    """True unless at least one GATING finger separates the two signatures
    (one requires "ext" where the other requires "curl"). "any" (or a
    missing finger) is compatible with everything, so a permissive
    signature conflicts with everything it doesn't explicitly contradict —
    e.g. {"index": "ext"} alone conflicts with pointer AND scroll AND
    open_palm AND horns."""
    for f in FINGERS:
        sa, sb = a.get(f, "any"), b.get(f, "any")
        if {sa, sb} == {"ext", "curl"}:
            return False
    return True


def check_conflicts(sig: Signature, named: dict[str, Signature]) -> list[str]:
    """Names in ``named`` whose signatures conflict with ``sig``."""
    return [name for name, other in named.items() if signatures_conflict(sig, other)]


def signature_from_states(ext: dict[str, bool]) -> Signature:
    """Capture flow: freeze the four gating fingers' latched states into a
    signature. Thumb is deliberately omitted (== "any")."""
    return {f: ("ext" if ext.get(f) else "curl") for f in FINGERS}


def normalize_custom_entries(entries: list) -> tuple[list[dict], list[str]]:
    """Canonicalize config ``custom_gestures`` entries.

    Accepts both the v2 shape {name, signature: {...}, hold_ms, cooldown_ms,
    action} and the legacy shape {name, pose: "horns", ...}. Returns
    (parsed, skipped_names): parsed entries are
    {name, signature, hold_ms, cooldown_ms, action}; invalid entries land in
    skipped_names so the caller can warn at startup, never silently
    (preserves the original engine parse semantics)."""
    parsed: list[dict] = []
    skipped: list[str] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            skipped.append(str(entry))
            continue
        pose = str(entry.get("pose", ""))
        name = str(entry.get("name", pose or "custom"))
        if "signature" in entry:
            sig = normalize_signature(entry.get("signature"))
        elif pose:
            sig = LEGACY_POSES.get(pose)
        else:
            sig = None
        action = entry.get("action")
        if sig is None or not isinstance(action, dict) or not action:
            skipped.append(name)
            continue
        # Coercion failures (junk strings, JSON NaN/Infinity — json.loads
        # accepts both, and non-finite floats would poison config.json and
        # the panel's SSE stream) skip like any other invalid entry: this
        # parser must NEVER raise, whatever a hand-edited config contains.
        try:
            hold_ms = float(entry.get("hold_ms", 300.0))
            cooldown_ms = float(entry.get("cooldown_ms", 1200.0))
        except (TypeError, ValueError):
            skipped.append(name)
            continue
        if not (math.isfinite(hold_ms) and math.isfinite(cooldown_ms)):
            skipped.append(name)
            continue
        parsed.append({
            "name": name,
            "signature": dict(sig),
            "hold_ms": hold_ms,
            "cooldown_ms": cooldown_ms,
            "action": dict(action),
        })
    return parsed, skipped
