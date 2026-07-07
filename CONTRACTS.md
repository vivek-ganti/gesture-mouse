# Module contracts

Read `gesture_mouse/types.py` and `gesture_mouse/config.py` first — they are the
source of truth for data shapes and tunables. This file pins each module's
public API so modules built independently integrate cleanly. If you must
deviate, keep the call-site shape identical and document why in the module
docstring.

Full product design: `/Users/vivek/.claude/plans/like-how-wispr-flow-peppy-valiant.md`
(gesture rules, thresholds, state machine, invariants — normative).

Global rules:
- `engine.py`, `palm.py`, `filters.py`, `types.py`, `signatures.py`,
  `calibration.py`: pure logic. No macOS, cv2, mediapipe, or numpy imports.
  No wall-clock reads — time only from `ts_ms`.
- `panel.py`: stdlib ONLY (http.server/json/threading/queue/secrets/...) and
  NO gesture_mouse imports at all — it moves plain dicts. Its server threads
  touch nothing outside PanelServer's own state; every app effect flows
  through the command queue drained on the main thread.
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

class LandmarkSmoother:
    """Per-landmark-index One Euro filtering for POSE CLASSIFICATION, kept
    separate from CursorPipeline's own anchor filter (cfg.pose.smoothing_*,
    typically a higher mincutoff than the cursor filter -- pose tests only
    need a stable boolean, not px-accurate tracking)."""
    def __init__(self, cfg: Config): ...
    def smooth(self, frame: LandmarkFrame) -> tuple[Point, ...] | None
        # None when frame.landmarks is None; else all 21 points filtered
    def set_mincutoff(self, mincutoff: float) -> None   # live-tune, see engine.py
    def reset(self) -> None
```

## signatures.py

```python
FINGERS = ("index", "middle", "ring", "pinky")   # gate matching
ALL_FINGERS = ("thumb",) + FINGERS               # sampled/displayed only
Signature = dict   # finger -> "ext" | "curl" | "any"; missing key == "any"
FINGER_JOINTS: dict[str, (mcp, pip, tip)]        # thumb uses MCP/IP/TIP
BUILTINS: dict[str, Signature]                   # pointer/open_palm/scroll/horns
LEGACY_POSES = {"horns": BUILTINS["horns"]}      # {"pose": "horns"} back-compat

def pip_angle_deg(lm, mcp, pip, tip) -> float    # interior angle: 180=straight, 0=folded
def compute_finger_angles(lm) -> dict[str, float]   # all five, thumb included

class FingerLatch:                                # two-threshold hysteresis latch
    extended: bool                                # starts curled
    def update(self, angle_deg, extend_at, curl_at) -> bool
    def reset(self) -> None

def normalize_signature(raw) -> Signature | None  # unknown finger/state = invalid;
                                                  # thumb forced "any"; all-"any" invalid
def signatures_conflict(a, b) -> bool  # True unless a gating finger separates them
def check_conflicts(sig, named: dict[str, Signature]) -> list[str]
def signature_from_states(ext: dict[str, bool]) -> Signature  # capture: 4 fingers, no thumb
def normalize_custom_entries(entries) -> (parsed, skipped_names)
    # v2 {name, signature, hold_ms, cooldown_ms, action} AND legacy
    # {pose: "horns", ...} -> canonical entries; invalid -> skipped, never silent
```

The ONE definition of "what a pose is": built-in poses and user gestures run
through the same signature matcher in the engine (engine._match). scroll is
signature + an engine-side fingertips-together test (`scroll.together_max`) —
its signature is reserved (a capture matching it is rejected as a conflict).

## calibration.py

```python
STEPS: tuple[CalibStep, ...]   # pointer, open_palm, scroll, horns, fist, relaxed
    # CalibStep: id, label, instruction, expected: {finger: "ext"|"curl"|"ignore"}
    # relaxed = all "ignore" (validation-only: must match NOTHING)
THUMB_SOURCES = {"open_palm": "ext", "fist": "curl"}   # thumb clusters (advisory)

class CalibrationSession:      # SETTLE_MS=750, TARGET=90, MIN=45, TIMEOUT=15000
    state: "await_step" | "sampling" | "done"
    def begin_step(step_id, ts_ms); def add_sample(ts_ms, angles); def cancel()
    def progress() -> dict     # step/step_i/collected/needed/done_steps/failed_step
    def compute(defaults, custom_sigs) -> CalibrationResult   # raises unless done

def derive_thresholds(ext, curl, relaxed, defaults, min_gap=15, min_hyst=8) -> FingerResult
    # NORMATIVE math: ext_low=P10(E), curl_high=P90(C), gap=ext_low-curl_high;
    # <40 samples either side -> "insufficient"; gap<15 -> "overlap" (keep defaults);
    # else mid±band/2 with band=min(max(8, 0.3*gap), gap); relaxed nudge: if
    # P75(R)>=extend, raise extend to min(P75(R)+5, ext_low-3) if it fits, else
    # "relaxed_overlap"; clamp extend<=178, curl>=2, round 0.1.
# compute() also VALIDATES: replays each step's samples through fresh
# FingerLatches at the derived thresholds (own signature >=90% match after a
# 10-sample warm-up; relaxed matches nothing >=95%, from both latch seeds).
```

Applied by __main__ on calibrate_apply: per-finger results with status
ok/relaxed_overlap land in `cfg.pose.fingers[finger] = {extend, curl}`
(thumb additionally `"advisory": true` — stored, never gates), then
`store.save()`. Smoothing params are never touched by calibration.

## panel.py

```python
COMMAND_TYPES  # camera_start/stop/switch, panic, quit, set_setting,
               # calibrate_start/begin_step/cancel/apply,
               # capture_start/cancel, save_gesture, delete_gesture
@dataclass(frozen=True) class PanelCommand: type: str; payload: dict

class PanelServer:
    def __init__(port, html_path, token=None)   # token auto: secrets.token_urlsafe(16)
    def start() -> str      # binds 127.0.0.1 (EADDRINUSE -> port 0); daemon thread;
                            # returns http://127.0.0.1:PORT/?token=...
    def publish(event, data)     # main thread; serialize once, fan out to per-client
                                 # bounded queues (32, drop-on-full); "frame" events
                                 # throttled to >=66ms apart, others always pass
    def poll_commands() -> list[PanelCommand]    # main loop drains every tick
    def has_clients() -> bool;  def close()

def frame_event(**primitives) -> dict   # pure SSE "frame" shaper (schema below)
```

Endpoints: `GET /` = index.html (re-read per request; no token needed);
`GET /events?token=` = SSE (last "config" event replayed to late joiners,
then throttled "frame" events; ": ping" keepalives); `POST /command?token=`
= 202/400/403/413. SECURITY (non-optional — save_gesture carries shell
argv): 127.0.0.1 bind + per-run URL token + Host-header allowlist
(127.0.0.1/localhost only, DNS-rebinding guard) checked before routing.
NO camera image ever crosses the socket — smoothed landmarks only.

"frame" schema (shaped by frame_event): ts, session, engine, hand, fps,
latency_ms, suspend_reason, img_w/h, landmarks ([[x,y]*21]|null, SMOOTHED —
the classifier's own input), fingers {finger: {angle, ext}} (thumb ext=null),
thresholds {finger: {extend, curl, calibrated}}, pinch {left,right},
pinch_cfg, palm {open, phase, m, disp_frac}, mode, calibration
(progress()+results|null), capture ({active, signature, stable_ms, done,
conflicts}|null), toast ({id, level, text}|null, rides ~3s, dedup by id).
"config" schema: settings (whitelisted dotted paths), custom_gestures
(normalized), builtin_gestures, cameras, camera_index, calib_defaults,
calib_steps, pose_fingers.

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
    def retune_pose_smoothing(self) -> None
        # live-tune: push cfg.pose.smoothing_mincutoff into the already-
        # constructed LandmarkSmoother filters (see __main__.py live-tune keys)
    def reload_customs(self) -> None
        # re-parse cfg.custom_gestures WITHOUT a reset (panel save/delete +
        # config hot-reload); latches/holds/other transients untouched
    finger_angles: dict[str, float]       # property; smoothed per-finger angles
                                          # (thumb incl.), {} when no valid hand
    finger_states: dict[str, bool]        # property; latched ext per gating
                                          # finger, read WITHOUT updating
    smoothed_landmarks: tuple[Point, ...] | None   # property; classifier input
```

Finger extension (`_ext`) is an ANGLE metric, not the old tip-to-wrist
ratio: `signatures.pip_angle_deg` computes the interior angle at the PIP
joint (180 = straight, 0 = folded), and a per-finger `signatures.FingerLatch`
(engine keeps `_FingerState`/`_pip_angle_deg` aliases) latches "extended" /
"curled" with real hysteresis — per-finger calibrated thresholds from
`cfg.pose.fingers[name]` when present, else the global
`extend_angle_deg`/`curl_angle_deg` pair, read fresh every frame. ALL pose
tests (built-in and custom) go through ONE generic matcher,
`_match(signature, lm)`, which updates every gating finger's latch each
call (FingerLatch.update is idempotent within a frame at a fixed angle, so
multiple pose tests per frame stay safe). Pose tests run on landmarks
pre-smoothed by a `LandmarkSmoother` (One Euro per landmark index,
`cfg.pose.smoothing_*`) — distance-based signals (pinch, scale, anchor-jump,
scroll/tab joystick deflection) keep using RAW landmarks, since they already
have their own, better-tuned filters or need unfiltered continuous motion.

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
class SwipePhase(Enum): IDLE; ARMING; ARMED; COOLDOWN

class PalmDetector:
    def __init__(self, cfg: PalmConfig, bindings: dict[str, str],
                 debug: bool = False): ...
    active: bool                     # last frame's open-palm flag (preview)
    last_m: float | None             # latest spread metric (preview meter)
    last_disp_frac: float            # net swipe travel so far (preview meter)
    phase: SwipePhase
    armed: bool                      # property: phase is ARMED
    engaged: bool                    # property: phase is not IDLE -> engine
                                     #   holds PALM (cursor frozen) while True
    events: deque[(ts_ms, str)]      # human-readable events when debug=True;
                                     #   caller drains and prints (module is pure)
    def update(self, frame: LandmarkFrame, open_palm: bool) -> list[Intent]
    def disarm(self) -> None         # engine calls on pinch/scroll entry
    def reset(self) -> None
```

Swipe model (arm-at-rest -> net displacement; replaces the old per-frame
velocity/pose-during-motion stack, which never fired on real data):

- ARM: open palm (engine-debounced flag) AND palm-center speed <
  `arm_max_speed_fw_s`, held `arm_hold_ms` -> ARMED. Pose is tested ONLY here,
  at rest — motion blur destroys finger landmarks during the swipe itself.
- Tracked point: PALM CENTER = mean of landmarks {0,5,9,13,17}. Never fingertips.
- ARMED: origin refreshes every still frame (drift can't accumulate); once
  motion starts, fire on NET displacement >= `swipe_min_disp_frac` of the
  frame span, dominant axis >= `swipe_axis_dominance` x the minor, within
  `swipe_max_duration_ms`. No velocity math.
- Hand-absent frames NEVER clear the swipe trajectory or disarm unless the
  gap exceeds `swipe_gap_bridge_ms` — MediaPipe drops the hand for a few
  frames in the middle of every real fast swipe (blur + tracking-ROI loss).
- After a fire: `swipe_cooldown_ms`, then a FULL re-arm for the next swipe.
  Firing also walls off the spread history so the post-swipe hand relaxation
  can't read as pinch_in/spread_out.

Spread gestures (unchanged): m = mean dist(5 fingertips, palm centroid)/scale;
pinch-in: m > spread_open falling below spread_closed within spread_window_ms;
spread-out: m < spread_in_start rising above spread_out. Their history DOES
clear on hand loss (those gestures happen at rest, where absence is real).

Engine integration: PALM state entry = debounced open-pose hold (`enter_ms`)
OR `palm.engaged`; exit only when pose gone AND detector back to IDLE (freeze
survives arm -> swipe -> cooldown; rebase on exit). Swipe-bound intents
forward from CLUTCH_WAIT / POINTER / PALM / HANDS_LOST (arming is the
deliberate-intent gate; no cursor clutch required; the detector only emits on
hand-present frames), never from PINCHED/RIGHT_PINCH/SCROLL — those disarm
the detector on entry. pinch_in/spread_out keep their PALM-or-slow-POINTER
forwarding gate. The anchor-teleport -> HANDS_LOST guard is skipped in PALM
(a real flick moves the anchor >25% of the frame in one frame).

### Custom gestures (engine + synth + panel editor)

- Config: `Config.custom_gestures` = list of dicts, v2 shape {name,
  signature: {finger: "ext"|"curl"|"any"}, hold_ms, cooldown_ms, action} or
  legacy {name, pose: "horns", ...}. Parsed via
  `signatures.normalize_custom_entries` at engine construction AND on
  `engine.reload_customs()` (panel save/delete, config hot-reload); invalid
  entries go to `engine.custom_skipped` (caller prints a warning).
- Engine: each entry's signature is tested by the same `_match` used by
  built-in poses. Held (debounced) hold_ms with a mostly-still cursor from
  CLUTCH_WAIT/POINTER, no pinch in flight -> Intent("custom:<name>", TRIGGER,
  {"action": <dict>}) + per-gesture cooldown.
- Panel editor: capture-by-demonstration (signature_from_states of the live
  latches, 1s stable, conflict check vs BUILTINS + other customs) -> name +
  action form -> save_gesture command -> cfg mutation + store.save() +
  engine.reload_customs(), all on the main thread.
- Synth: `custom_action_argv(action) -> argv | None` (pure, tested): key taps
  (incl. bare modifiers) via osascript System Events, shell argv, or reuse of
  a system trigger_command. Fire-and-forget Popen. Also the validator the
  panel save path uses to reject malformed actions.

## tracker.py

```python
# Capture backend: camera.backend "avf" (default) = native AVCaptureSession,
# device selected by NAME -> uniqueID (label and video cannot disagree);
# "opencv" = cv2.VideoCapture fallback (index-based; order unreliable).
def list_cameras() -> list[str]          # AVFoundation localizedName order
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

Preview renders onto a FIXED internal canvas (`cfg.preview.canvas_w/h`, not
the camera's actual capture size — AVFoundation/OpenCV presets are not
guaranteed) in every session state, so the window never resizes/jumps.
`_fit_frame_to_canvas` stretch-resizes the real/black frame into it every
draw (stretch, not letterbox: privacy mode's synthetic skeleton makes minor
distortion low-cost, and it needs no padding-offset math); `_draw_skeleton`
rescales landmark points (in the frame's OWN original pixel space) to match.
`_compute_regions` hands every text element (status, hint, camera list) a
reserved, non-overlapping y-position instead of each draw function picking
one independently. Does NOT attempt to work around `cv2.imshow`'s Retina/
HiDPI window-scaling behavior on macOS (open, unfixed OpenCV limitation,
upstream issue #20403) — a correctly laid out fixed canvas stays legible
regardless of what scale the OS renders the window at.

## __main__.py

argparse: `--list-cameras`, `--camera NAME`, `--config PATH`, `--replay FILE`,
`--record FILE`, `--preview` (opt-in cv2 debug window; `--no-preview` is a
deprecated no-op), `--no-privacy`, `--start-active`, `--panel-port N`,
`--no-panel`, `--no-open`.
Flow: preflight permissions (skip for --replay/--list-cameras) -> wiring ->
panel start (URL printed + browser opened unless --no-open/replay) ->
session FSM (IDLE camera-off default; toggle/panic via threading.Event from
hotkeys; WARMUP until first confident frame or 2s; SUSPENDED on guard trip,
resume only via clutch reacquire or toggle) -> hot loop per Pipeline order,
plus per tick: panel.poll_commands() drained (settings whitelist _SETTINGS;
calibration/capture mode ownership) and _publish_panel (config event when
dirty + throttled frame event when clients connected) -> live-tune keys with
--preview ([ ] mincutoff, ; ' beta, - = pose angles incl. per-finger pairs,
, . pose smoothing, b box, p privacy, h help, 1-9 camera, q quit) ->
try/finally + atexit -> synth.release_all() + panel.close().
`ui_mode` ("normal" | "calibrating" | "capturing"): mode entry does
release_all + engine.notify_suspended; while non-normal the engine/pipeline
still tick but guards are SKIPPED and intents are DROPPED (the user is
deliberately mousing in the browser); mode exit resets the engine and
re-arms the guards. PerfTimer prints p50/p95 per stage every 5s. Replay
mode: no synth posts (print intents instead) unless `--replay-post`; the
panel runs against replay too (camera-free frontend data path).

## tests/

Pure-logic tests only need engine/palm/filters/types/config. Synthetic fixture
builders (no camera): craft landmark sequences for: careful 500ms stationary
click (expect exactly DOWN,UP no drag); fly-and-pinch; drag (freeze->unfreeze->
release latch); double-click; right-click with thumb transit past index (expect
NO left click); scroll enter/exit; palm swipe each direction (+ refractory);
five-finger pinch-in and spread-out; hands-lost mid-drag (auto mouseUp);
wrong-handedness ignored. Golden assertion = exact Intent name/phase sequence.
