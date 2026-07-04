# Module contracts

Read `gesture_mouse/types.py` and `gesture_mouse/config.py` first — they are the
source of truth for data shapes and tunables. This file pins each module's
public API so modules built independently integrate cleanly. If you must
deviate, keep the call-site shape identical and document why in the module
docstring.

Full product design: `/Users/vivek/.claude/plans/like-how-wispr-flow-peppy-valiant.md`
(gesture rules, thresholds, state machine, invariants — normative).

Global rules:
- `engine.py`, `palm.py`, `filters.py`, `types.py`: pure logic. No macOS, cv2,
  mediapipe, or numpy imports. No wall-clock reads — time only from `ts_ms`.
- All temporal logic in milliseconds of `LandmarkFrame.ts_ms`, never frame counts.
- MediaPipe z is never used anywhere.
- venv: `.venv/bin/python` (3.12). Test with `.venv/bin/python -m pytest`.
- Do not `git commit`.

## Pipeline order (per frame, in `__main__.py`)

```
tracker.read() -> LandmarkFrame | None (None = frame grab failed; skip)
  -> [recorder.write(frame) if recording]
  -> cursor_pipeline.update(frame) -> CursorSample          # filters.py
  -> engine.update(frame, cursor) -> EngineOutput           # engine.py (delegates PALM to palm.py)
  -> apply EngineOutput.freeze / drag / rebase to cursor_pipeline (affects NEXT frame)
  -> guards checks (mouse divergence, keyboard mute) -> may suspend instead of executing
  -> for intent in EngineOutput.intents: synth.execute(intent)
  -> indicator.set_state(snapshot); preview.show(...)
```

## filters.py

```python
class OneEuro:
    def __init__(self, mincutoff: float, beta: float, dcutoff: float = 1.0): ...
    def filter(self, value: float, ts_ms: float) -> float   # measured dt, never fixed 1/30
    def reset(self) -> None
    def set_mincutoff(self, mincutoff: float) -> None

class CursorPipeline:
    def __init__(self, cfg: Config, screen_w: float, screen_h: float): ...
    def update(self, frame: LandmarkFrame) -> CursorSample
        # anchor = INDEX_MCP; One Euro x/y; control-box map to screen; +rebase
        # offset; clamp to screen; speed estimate (px/s, filtered); pixel
        # quantization when speed < ~2 px/frame; while frozen: hold last output
        # position (still update filters/speed internally); no hand: hold last.
    def rebase(self, to_x: float, to_y: float) -> None
        # adjust offset so current output == (to_x, to_y); used on clutch engage,
        # drag unfreeze, PALM exit
    def set_frozen(self, frozen: bool) -> None
    def set_drag(self, drag: bool) -> None        # switches mincutoff <-> drag_mincutoff
    def pinch(self, name: str, raw: float, ts_ms: float) -> float
        # per-name One Euro (pinch_mincutoff) for pinch distances; engine calls this
    def reset(self) -> None
```

## engine.py

```python
@dataclass
class EngineOutput:
    intents: list[Intent]
    freeze: bool          # cursor frozen next frame
    drag: bool            # drag filter mode next frame
    rebase: tuple[float, float] | None   # rebase request (screen coords)

class GestureEngine:
    def __init__(self, cfg: Config, pinch_filter): ...
        # pinch_filter: callable (name, raw, ts_ms) -> float — pass CursorPipeline.pinch
    state: EngineState
    def update(self, frame: LandmarkFrame, cursor: CursorSample) -> EngineOutput
    def notify_suspended(self) -> None    # force-release held buttons state (emit UPs first)
    def reset(self) -> None
```

Implements (rules + exact thresholds in the plan doc §Gesture vocabulary):
CLUTCH_WAIT -> pointer-pose 150ms -> POINTER (emit rebase, no cursor jump);
left pinch with conditional position latch + release latch + hysteresis +
debounce + scale-stability gate; drag (12px distance-only unfreeze, in-drag
release 0.60, minor-axis dead-band, drag filter mode); double click (500ms /
15px window, relaxed 0.44/0.42 thresholds, click_count=2); right = thumb-middle
pinch with index-extended requirement + argmin arbitration; scroll (pose+speed
gate entry, joystick velocity -> scroll MOVE intents with dy_px per frame,
zero velocity when exit-counting starts); HANDS_LOST (freeze on first missing
frame, auto-UP after hands_lost_ms, reacquire via clutch 250ms); anchor jump
>25% frame in one frame = HANDS_LOST; handedness pinned to cfg.hand (frames
with wrong handedness = hand absent). Position history deque for latches.
Engine keeps last N (ts, x, y) cursor samples internally for the latch lookback.
PALM: engine computes the all-5-extended pose; while POINTER (no pinch active)
pose sustained palm.enter_ms -> state PALM (freeze cursor), delegate each frame
to PalmDetector.update(); PALM exits when pose ends (rebase on exit).

## palm.py

```python
class PalmDetector:
    def __init__(self, cfg: PalmConfig, bindings: dict[str, str]): ...
    active: bool
    def update(self, frame: LandmarkFrame, open_palm: bool) -> list[Intent]
        # swipe: unidirectional anchor velocity > swipe_min_vel_fw_s frame-widths/s
        # AND displacement > swipe_min_disp_frac within swipe_window_ms, dominant
        # axis picks direction; TRIGGER intent named by bindings[...]; refractory;
        # pose must drop before re-trigger.
        # spread metric m = mean dist(5 fingertips, palm centroid)/scale;
        # pinch-in: m > spread_open falling below spread_closed within window;
        # spread-out: m < spread_in_start rising above spread_out within window.
    def reset(self) -> None
```

Note: spread-out starts from a curled hand (not open palm) — engine also routes
frames to PalmDetector while the hand is curled-but-present in POINTER with
cursor speed low, OR simpler: PalmDetector.update is called EVERY frame with the
open_palm flag; it manages its own windows. Keep it self-contained.

## tracker.py

```python
def list_cameras() -> list[str]          # AVFoundation localizedName order == cv2 index order
def camera_index(name: str) -> int | None

class CameraTracker:
    def __init__(self, cfg: Config, model_path: str = "hand_landmarker.task"): ...
    def open(self) -> None               # cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    def read(self) -> LandmarkFrame | None   # None = grab failed
        # rotate (cfg.camera.rotate) -> mirror (cfg.camera.mirror) -> BGR->RGB ->
        # detect_for_video(ts from ONE process-lifetime monotonic ms clock; the
        # Tasks API raises on non-increasing ts, so the clock survives close/open)
        # -> pick hand matching cfg.hand (else treat as absent) -> pixel coords ->
        # scale = max(dist(5,17), 0.7*dist(0,9)) -> LandmarkFrame
    def close(self) -> None              # release camera fully (IDLE = zero camera use)

class Recorder:
    def __init__(self, path: str, header: dict): ...   # header line 0: mirror, rotate,
    def write(self, frame: LandmarkFrame) -> None      # img dims, config snapshot, source
    def close(self) -> None

class ReplayTracker:                      # same read()/close() duck-type as CameraTracker
    def __init__(self, path: str): ...
    def read(self) -> LandmarkFrame | None    # None at EOF; source="replay:<path>"
    header: dict
```

JSONL: line 0 = header dict; then one dict per frame
`{"ts_ms":..., "handedness":..., "lm":[[x,y]*21]|null, "w":..., "h":..., "conf":..., "scale":...}`.

## synth.py

```python
class Synth:
    def __init__(self): ...              # CGEventSource with userData tag 0x6D6F7573
    def execute(self, intent: Intent) -> None
        # move: mouseMoved (or leftMouseDragged while left button down)
        # left DOWN/UP: clickState from payload click_count; posts at payload x,y
        # drag MOVE: leftMouseDragged
        # right DOWN/UP; scroll MOVE: pixel-unit scroll wheel event (dy_px)
        # TRIGGERs: space_prev/space_next=Ctrl+Left/Right, mission_control=Ctrl+Up,
        # app_expose=Ctrl+Down, show_desktop=fn+F11 (kVK_F11=103 + maskSecondaryFn),
        # launchpad=subprocess.Popen(["open","-a","Launchpad"])
        # All positions clamped to main display bounds. Key chords: flags set on
        # both down and up; small inter-event delay not needed at HID tap.
    def release_all(self) -> None        # UP for any held button; call from atexit/finally
    left_down: bool
    last_pos: tuple[float, float]        # last synthetic cursor position posted
    last_chord_ts: float                 # time.monotonic() of last synthetic key chord

def real_cursor_pos() -> tuple[float, float]   # CGEventGetLocation(CGEventCreate(None))
def screen_size() -> tuple[float, float]       # CGDisplayBounds(CGMainDisplayID())
```

## guards.py

```python
class Guards:
    def __init__(self, cfg: SuspendConfig, synth: Synth): ...
    def mouse_moved_physically(self) -> bool
        # real_cursor_pos() vs synth.last_pos divergence > mouse_divergence_px
    def keyboard_active(self) -> bool
        # Quartz.CGEventSourceSecondsSinceLastEventType(
        #   kCGEventSourceStateHIDSystemState, kCGEventKeyDown) * 1000 < keyboard_mute_ms
        # BUT ignore if within ~300ms of synth.last_chord_ts (our own chords).
        # No event tap, no Input Monitoring, no runloop needed.
```

## hotkeys.py

```python
def register_hotkeys(toggle_cb, panic_cb, cfg: HotkeyConfig) -> None
    # Carbon RegisterEventHotKey via ctypes; handler thread runs its own event
    # loop; callbacks invoked from that thread — they must only set
    # threading.Event flags. No TCC permission required. If ctypes proves
    # unreliable, fallback: pip package `quickmachotkey` (document in module).
```

## indicator.py / preview.py

```python
class Indicator:                          # borderless NSPanel corner dot
    def __init__(self): ...               # canJoinAllSpaces | fullScreenAuxiliary,
    def set_state(self, snap: StateSnapshot) -> None   # click-through, ~14px dot,
    def close(self) -> None               # colors per plan §Feedback UI
    # Created/updated from the main thread only (the hot loop IS the main thread).

class Preview:                            # cv2 tuning window
    def __init__(self, cfg: Config): ...
    def show(self, bgr_frame, lm_frame: LandmarkFrame, snap: StateSnapshot,
             pinch_values: dict[str, float]) -> int    # returns cv2.waitKey(1) code
        # privacy mode default: skeleton on black, never the camera image.
        # Draw: skeleton, control box, anchor dot, pinch meter vs thresholds,
        # fps/latency, warmup/fps-drop banners.
    def close(self) -> None
```

## __main__.py

argparse: `--list-cameras`, `--camera NAME`, `--config PATH`, `--replay FILE`,
`--record FILE`, `--no-preview`, `--no-privacy`, `--start-active`.
Flow: preflight permissions (skip for --replay/--list-cameras) -> wiring ->
session FSM (IDLE camera-off default; toggle/panic via threading.Event from
hotkeys; WARMUP until first confident frame or 2s; SUSPENDED on guard trip,
resume only via clutch reacquire or toggle) -> hot loop per Pipeline order ->
live-tune keys ([ ] mincutoff, ; ' beta, b box, p privacy, q quit) ->
try/finally + atexit -> synth.release_all(). PerfTimer prints p50/p95 per stage
every 5s. Replay mode: no synth posts (print intents instead) unless
`--replay-post` given.

## tests/

Pure-logic tests only need engine/palm/filters/types/config. Synthetic fixture
builders (no camera): craft landmark sequences for: careful 500ms stationary
click (expect exactly DOWN,UP no drag); fly-and-pinch; drag (freeze->unfreeze->
release latch); double-click; right-click with thumb transit past index (expect
NO left click); scroll enter/exit; palm swipe each direction (+ refractory);
five-finger pinch-in and spread-out; hands-lost mid-drag (auto mouseUp);
wrong-handedness ignored. Golden assertion = exact Intent name/phase sequence.
