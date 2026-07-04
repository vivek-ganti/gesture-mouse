"""Shared data contract for the gesture-mouse pipeline.

Pure Python: no MediaPipe, numpy, or macOS imports. Everything downstream of
the tracker speaks these types and nothing else — this module is the seam a
future Swift/Vision port replaces the tracker behind.

Conventions frozen here (and in every recorded fixture):
- Landmarks are PIXEL coordinates in the already-mirrored frame (selfie view),
  origin top-left, y-down. Conversion from MediaPipe normalized coords happens
  in exactly one place: the tracker.
- All timestamps are milliseconds from one process-lifetime monotonic clock.
  Temporal rules are always expressed in ms of ts_ms, never frame counts.
- MediaPipe z is never used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# MediaPipe hand landmark indices (21 points).
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def dist(a: Point, b: Point) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


@dataclass(frozen=True)
class LandmarkFrame:
    """One tracked camera frame. landmarks is None when no hand is present."""

    ts_ms: float
    handedness: str | None            # "Left" / "Right" / None
    landmarks: tuple[Point, ...] | None  # 21 pixel-coord points, mirrored frame
    img_w: int
    img_h: int
    confidence: float                 # 0.0 when no hand
    scale: float                      # px: max(dist(5,17), 0.7*dist(0,9)); 0.0 when no hand
    source: str                       # camera name or "replay:<path>"

    @property
    def hand_present(self) -> bool:
        return self.landmarks is not None and len(self.landmarks) == 21


@dataclass(frozen=True)
class CursorSample:
    """Filtered, mapped cursor state in SCREEN coordinates (global display
    space, top-left origin of primary display, y-down — CGEvent convention)."""

    x: float
    y: float
    speed_px_s: float
    frozen: bool          # True while a detector holds the cursor still
    ts_ms: float


class Phase(Enum):
    DOWN = "down"
    MOVE = "move"
    UP = "up"
    TRIGGER = "trigger"   # one-shot action (system gestures)


@dataclass(frozen=True)
class Intent:
    """What the engine wants done. The only engine → synth vocabulary.

    Names and payloads (payload keys in parens):
      move          MOVE     (x, y)                 cursor move
      left          DOWN/UP  (x, y, click_count)    left button; click_count 1 or 2
      drag          MOVE     (x, y)                 leftMouseDragged stream
      right         DOWN/UP  (x, y)
      scroll        MOVE     (dy_px)                pixel scroll, +down / -up
      space_prev    TRIGGER  ()                     Ctrl+Left
      space_next    TRIGGER  ()                     Ctrl+Right
      mission_control TRIGGER ()                    Ctrl+Up
      app_expose    TRIGGER  ()                     Ctrl+Down
      launchpad     TRIGGER  ()                     open -a Launchpad
      show_desktop  TRIGGER  ()                     fn+F11
      tab_next      TRIGGER  ()                     Ctrl+Tab
      tab_prev      TRIGGER  ()                     Ctrl+Shift+Tab
    """

    name: str
    phase: Phase
    payload: dict[str, Any] = field(default_factory=dict)
    ts_ms: float = 0.0


class SessionState(Enum):
    IDLE = "idle"            # camera off
    WARMUP = "warmup"        # camera opening / exposure ramp
    ACTIVE = "active"        # tracking loop running
    SUSPENDED = "suspended"  # camera on, synthesis muted (real input won)


class EngineState(Enum):
    CLUTCH_WAIT = "clutch_wait"  # hand may be present; no cursor motion yet
    POINTER = "pointer"          # cursor control
    PINCHED = "pinched"          # left pinch held (click pending or dragging)
    RIGHT_PINCH = "right_pinch"
    SCROLL = "scroll"
    PALM = "palm"                # open-hand system-gesture mode, cursor frozen
    HANDS_LOST = "hands_lost"


@dataclass(frozen=True)
class StateSnapshot:
    """Read-only status for the indicator / preview / logging."""

    session_state: SessionState
    engine_state: EngineState | None
    hand_present: bool
    fps: float
    latency_ms: float
    suspend_reason: str | None = None  # "mouse" | "keyboard" | "meeting" | None
