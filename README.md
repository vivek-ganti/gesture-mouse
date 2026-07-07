# gesture-mouse

Control the macOS screen with hand gestures from any camera: move the cursor,
click, double-click, right-click, drag, scroll, and the trackpad system
gestures (Spaces, Mission Control, App Exposé, Launchpad, Show Desktop) —
hands in the air, no hardware. MediaPipe hand tracking → pure-logic gesture
engine → raw Quartz CGEvents. macOS 15+, Apple Silicon, Python 3.12.

## Quick start

```bash
git clone https://github.com/vivek-ganti/gesture-mouse.git && cd gesture-mouse
./gesture-mouse --start-active
```

That's the whole install. The first run takes ~1-2 minutes to create a venv
and install dependencies automatically (needs Python 3.12 on your `PATH` —
`brew install python@3.12` if you don't have it; the script tells you if it's
missing); every run after that starts in under a second.

Grant the two permission prompts. A **control panel opens in your browser**
(local only — `http://127.0.0.1:8765`): live hand view, per-finger readout,
camera picker, settings, a gesture editor, and the calibration wizard.

**Calibrate first** (Calibrate tab, ~1 minute): hold six simple poses for a
few seconds each while the app measures YOUR hand's finger angles and sets
personal recognition thresholds — this is what makes gestures feel natural
instead of requiring robot-precise poses.

Then: **hold your index finger up (other fingers curled) for ~150 ms** — the
corner dot turns green and the cursor is yours. Pinch thumb+index to click.
Open your palm and swipe to switch Spaces. Type or grab the real mouse any
time — gestures suspend instantly.

## Setup

Nothing manual — `./gesture-mouse` bootstraps itself on first run (see Quick
start above). If you'd rather set it up by hand (or need a non-default Python
path):

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
# hand_landmarker.task (MediaPipe model) already sits in the repo root.
```

### Permissions

Two TCC grants, both attached to the **app hosting Python** (Terminal, iTerm2,
VS Code — switching terminals means granting again):

- **Camera** — prompted on first run.
  System Settings deep link: `x-apple.systempreferences:com.apple.preference.security?Privacy_Camera`
- **Accessibility** — required for CGEvent posting; the prompt only adds the
  host app to the list, you must flip the switch yourself.
  Deep link: `x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility`

`python -m gesture_mouse` runs a preflight checklist and refuses a
half-permissioned start. The global hotkeys need **no** permission at all, so
the panic key can never be silently dead.

### Pick a camera

```bash
./gesture-mouse --list-cameras    # names only, no camera opened
./gesture-mouse --bench-cameras   # opens each: read speed + brightness verdict
./gesture-mouse --camera "FaceTime HD Camera"
```

Capture is **native AVFoundation** (`camera.backend: "avf"`, the default):
the device is selected by its name → unique ID, so the printed/persisted name
and the actual video can never disagree (the old OpenCV index-based capture
made that correlation unreliable — device order shuffles between
enumerations; it remains available as `"backend": "opencv"`). At startup
every candidate camera is **probed** (read speed + real image content) and
dead virtual feeds (phone-cam apps with no phone connected) are skipped.
Per-camera `mirror` / `rotate` / `orientation` live in `config.json`.

Handedness labels also depend on the camera's mirror convention, so
`"hand": "auto"` (default) tracks whichever hand appears; pin `"right"` /
`"left"` only if a second person's hand keeps photobombing.

### If something looks off

```bash
.venv/bin/python tools/record.py --seconds 12 --show --out clip.jsonl  # watch the feed + detection live
.venv/bin/python tools/analyze_recording.py clip.jsonl                 # detection %, poses, pinch depth, intents fired
```

## Usage

```bash
./gesture-mouse                  # IDLE; ⌃⌥G to start
./gesture-mouse --start-active   # camera on immediately
```

`./gesture-mouse` is a launcher script — it creates `.venv` and installs
dependencies the first time it's run, and just finds them every time after
(works from any directory, e.g. `~/Documents/vivek-code/gesture-mouse/gesture-mouse`),
so you never have to type `.venv/bin/python -m gesture_mouse` or activate the
venv yourself. It forwards every flag, e.g. `./gesture-mouse --list-cameras`.
(`.venv/bin/python -m gesture_mouse ...` still works identically if you
prefer it or the venv is already active in your shell.)

| Flag | Effect |
|---|---|
| `--config PATH` | alternate config.json (hot-reloaded on change) |
| `--record FILE` | tee every tracked frame to a JSONL fixture |
| `--replay FILE` | replay a fixture: prints intents, posts nothing |
| `--replay-post` | with `--replay`: actually post the replayed intents |
| `--panel-port N` | web panel port (default 8765, `config.json → panel`) |
| `--no-panel` | disable the web panel |
| `--no-open` | don't auto-open the panel in the browser |
| `--preview` | ALSO open the cv2 debug window (off by default) |
| `--no-privacy` | show the camera image in the cv2 debug window |
| `--debug-gestures` | print swipe arming/candidate/rejection events live |

Hotkeys (global, configurable in `config.json`): **⌃⌥G** toggle IDLE⇄ACTIVE ·
**⌃⌥Esc** panic → IDLE, releasing all buttons.

## The control panel

`./gesture-mouse` starts a local web panel (printed URL, auto-opened; local
only — bound to 127.0.0.1, token-guarded, and it NEVER receives camera
images, only hand-landmark coordinates):

- **Live** — hand skeleton, state, and the per-finger readout: each finger's
  knuckle angle on a 0–180° track with its curl/extend thresholds marked.
  If a gesture ever misses, this view shows exactly which finger read wrong
  and by how much — no more guessing.
- **Gestures** — built-in gestures as cards, plus YOUR gestures: click Add,
  hold any pose to the camera for a second, name it, pick what it does
  (key tap like Option for Wispr Flow dictation, a shell command, or a
  system action). Collisions with existing gestures are detected up front.
- **Calibrate** — the wizard that fits recognition to your hand. Run it
  first; re-run any time lighting/camera changes.
- **Camera** — click to switch; persisted across restarts.
- **Settings** — every important threshold as a slider, applied live and
  saved to config.json.

Closing the tab changes nothing (the app keeps running); reopen the printed
URL any time.

The corner dot shows state everywhere (incl. fullscreen): gray idle · pulsing
warmup · white clutch-wait · green pointer · blue pinched · purple scroll ·
orange palm · yellow hands-lost · red suspended (M/K = mouse/keyboard reason).

### The cv2 debug window (`--preview`)

The old OpenCV tuning window still exists behind `--preview` (fixed 960x540
canvas, skeleton + meters). Its live-tune keys: `[` / `]` cursor mincutoff ·
`;` / `'` cursor beta · `-` / `=` gesture forgiveness (shifts angle
thresholds, including calibrated per-finger pairs) · `,` / `.` gesture
smoothing · `b` control-box overlay · `p` privacy · `1`-`9` switch camera ·
`h` help · `q` quit. PerfTimer prints per-stage p50/p95 every 5 s.

## Gesture cheat sheet

| Action | Gesture | Notes |
|---|---|---|
| Engage (clutch) | Pointer pose (index extended, others curled) held 150 ms | Presence is never intent; after engaging, a relaxed hand keeps control |
| Cursor move | Move hand; anchor is the index knuckle | Absolute map from a sub-rectangle of the frame to the screen |
| Left click | Thumb–index pinch | < 0.35 for 100 ms → down; open > 0.48 for 66 ms → up |
| Drag | Pinch, hold, move | Cursor frozen until 12 px of travel, so careful clicks never micro-drag |
| Double click | Two pinch taps < 500 ms and < 15 px apart | Posted with clickState 2; singles are never delayed |
| Right click | Thumb–**middle** pinch, index kept extended | Tap-only |
| Scroll | Index+middle extended and together, ring+pinky folded → vertical joystick | Entry needs the pose 100 ms *and* a slow cursor; cursor frozen while scrolling |
| Tab switch | Same two-finger scroll pose → flick **horizontally** | Right = next tab (Ctrl+Tab), left = previous (Ctrl+Shift+Tab); one per flick, 500 ms refractory |
| Space left/right | Open hand, **hold still a beat, then flick** left/right | The brief still moment "arms" the swipe (ARMED tag in the preview); works even without clutching in first |
| Mission Control | Open hand, hold still a beat, flick up | |
| App Exposé | Open hand, hold still a beat, flick down | |
| Launchpad | Five-finger pinch-in (open palm → fist) | Uses the continuous spread meter (bottom-left "P" bar), not the arming pose |
| Show Desktop | Five-finger spread-out (fist → open palm) | Same meter, opposite direction |
| Suspend / resume | Grab the real mouse or type / pointer pose 250 ms or ⌃⌥G | See safety model |
| Panic | ⌃⌥Esc | Buttons released, camera off |

### How swipes work (and how to debug them)

Swipes use an **arm-at-rest** model, the recipe every shipped MediaPipe
gesture project converges on: hold your open hand still for ~150 ms (the
"ARMED" tag appears in the preview and the "S" bar starts tracking), then
flick in one direction — the hand must travel about a quarter of the camera
frame. During the flick itself nothing else matters: not finger poses (motion
blur destroys them — the pose only gates the *arming*, tested while your
hand is still), and not even continuous tracking (MediaPipe routinely loses
a fast-moving hand for a few frames; the detector bridges gaps up to 350 ms).
After each swipe there's a ~1 s cooldown, then arm again for the next one.
The open-palm arming pose ignores the thumb (its extension is unreliable to
detect); Launchpad/Show Desktop read the continuous five-finger spread value
("P" meter) instead and need no arming.

If a swipe won't fire, run with `--debug-gestures`: every arming start/abort
(with the reason), every candidate evaluation, and every rejection prints to
the terminal — and `tools/analyze_recording.py` prints the same forensics
plus dropout histograms for a recorded clip.

Bindings for the six system gestures are remappable in `config.json`
(`bindings`: swipe_left/right/up/down, pinch_in, spread_out → any of
space_prev, space_next, mission_control, app_expose, launchpad, show_desktop).

## Custom gestures (map any pose to any shortcut)

**Use the panel** (Gestures tab → Add gesture): hold any hand pose to the
camera for a second — the app captures which fingers are up/down as the
gesture's signature, warns if it collides with an existing gesture, then you
name it and pick its action. Saved to `config.json`, live immediately.

A gesture = a per-finger signature (which of index/middle/ring/pinky are
extended vs curled; the thumb never gates — its hinge reads unreliably) +
`hold_ms` (hold the pose this long, mostly-still hand) + an action +
`cooldown_ms` refractory. Built-in poses use the exact same mechanism, so
captured gestures behave identically to shipped ones.

The JSON stays fully editable by hand (hot-reloaded while running) — and
it's the only place the `"any"` finger state is available:

```json
"custom_gestures": [
  { "name": "dictate", "pose": "horns", "hold_ms": 300,
    "action": { "type": "key", "key": "option" } },
  { "name": "calc",
    "signature": { "index": "curl", "middle": "ext", "ring": "ext", "pinky": "curl" },
    "hold_ms": 300,
    "action": { "type": "shell", "argv": ["open", "-a", "Calculator"] } }
]
```

(The legacy `"pose": "horns"` 🤘 form keeps working.) Action types:

| type | fields | does |
|---|---|---|
| `key` | `key`, optional `modifiers` | taps a key — bare modifiers work (`"option"` = Wispr Flow's dictation toggle), or chords like `{"key": "d", "modifiers": ["command"]}` |
| `shell` | `argv` | runs a command, e.g. `["open", "-a", "Snaply"]` |
| `trigger` | `name` | reuses a system action (`mission_control`, `show_desktop`, ...) |

The ships-by-default example is exactly the dictation case: hold the horns
sign ~300 ms → Option is tapped → Wispr Flow starts listening. Entries with
an invalid signature/unknown pose are skipped with a warning, never silently.

## Tuning (One Euro, Casiez procedure)

Two knobs (`config.json → one_euro`): `mincutoff` (jitter at rest) and `beta`
(lag at speed).

1. Set `beta = 0`. Hold your hand still: lower `mincutoff` (`[` key) until the
   cursor stops shimmering, and no lower.
2. Move fast between targets: raise `beta` (`'` key) until the cursor stops
   trailing, and no higher.

Offline, with a recording:

```bash
.venv/bin/python tools/record.py --seconds 10 --out me.jsonl   # opens camera
.venv/bin/python tools/tune.py me.jsonl                        # sweep, headless
.venv/bin/python tools/make_fixture.py all                     # synthetic fixtures
./gesture-mouse --replay fixtures/click.jsonl
```

`tools/tune.py` prints jitter_px (RMS motion at rest) and lag_px / lag_ms
(distance behind a fast hand) per parameter combo. Add `--noise-px 1.5` when
sweeping noiseless synthetic fixtures. Everything hot-reloads: edit
`config.json` while running and the new values apply within a second.

### If a gesture feels unreliable (misses often, needs an exaggerated pose)

**Calibrate first** (panel → Calibrate). Every finger's extended/curled
state comes from the angle at its middle knuckle — 180° is perfectly
straight, 0° is folded flat — smoothed (One Euro, same technique as cursor
smoothing) so per-frame camera jitter can't flip the reading, then latched
with two thresholds so a finger between them keeps its last state instead
of flickering. The shipped defaults (`pose.extend_angle_deg` 160 /
`curl_angle_deg` 130) are one-size guesses; the wizard replaces them with
per-finger values measured from YOUR hand (`config.json → pose.fingers`).

Then diagnose, don't guess: the panel's Live tab shows each finger's angle
against its thresholds in real time. Do the failing gesture and watch which
finger reads wrong — nudge that finger's threshold in Settings (or re-run
calibration in different lighting).

Still off after that:
- `pose.smoothing_mincutoff` (Settings slider): heavier smoothing kills
  more jitter but adds a touch of lag before a gesture registers.
- `pose_jitter_grace_ms` (default 120 ms): how long a pose hold (clutch
  engage, scroll entry, open-palm) tolerates dropped detection before
  resetting — raise it in bad lighting.
- `palm.forward_max_speed_px_s` (default 300) caps how fast the cursor can
  be moving for a five-finger pinch/spread to still register; lower it if
  accidental Launchpad/Show Desktop triggers happen during normal cursor
  use, raise it if a deliberate gesture with a drifting hand gets dropped.
- Last resort, record the failing gesture and look at the numbers:
  `./gesture-mouse --record clip.jsonl`, perform it 5×, then
  `tools/analyze_recording.py clip.jsonl` prints per-finger angle
  percentiles against your thresholds.

## Safety model

- **Real input wins, instantly, and stays won.** Every frame, *before* any
  synthetic event is posted: if the real cursor sits > 8 px from the last
  position we posted (someone grabbed the mouse), or a physical key went down
  in the last 1.5 s, all held buttons are released and the session SUSPENDS
  with the reason on the indicator.
- **Resume is deliberate, never a timer**: hold the pointer pose 250 ms
  (clutch reacquire) or press the toggle hotkey. Control resumes wherever the
  real pointer is — no jump.
- **Panic hotkey** (⌃⌥Esc) needs zero TCC permissions, so it works even when
  something else is broken: releases buttons, closes the camera, parks IDLE.
- **No exit path leaves a button down**: hands lost mid-drag auto-releases
  after 265 ms; quit, exceptions, and Ctrl-C all pass through a central
  try/finally + atexit `release_all()`.
- **When in doubt, do nothing**: every discrete gesture has hysteresis,
  temporal debounce, and a hand-scale stability gate.
- **Privacy**: IDLE keeps the camera fully closed; the preview defaults to
  skeleton-on-black (the camera image never reaches the screen unless you
  pass `--no-privacy` or hit `p`); recordings store landmark coordinates,
  never images. Every synthetic event is tagged (`0x6D6F7573`) so observers
  can tell it from real input.
