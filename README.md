# gesture-mouse

Control the macOS screen with hand gestures from any camera: move the cursor,
click, double-click, right-click, drag, scroll, and the trackpad system
gestures (Spaces, Mission Control, App Exposé, Launchpad, Show Desktop) —
hands in the air, no hardware. MediaPipe hand tracking → pure-logic gesture
engine → raw Quartz CGEvents. macOS 15+, Apple Silicon, Python 3.12.

## Quick start (60 seconds)

```bash
git clone https://github.com/vivek-ganti/gesture-mouse.git && cd gesture-mouse
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m gesture_mouse --start-active
```

Grant the two permission prompts, then: **hold your index finger up (other
fingers curled) for ~150 ms** — the corner dot turns green and the cursor is
yours. Pinch thumb+index to click. Open your palm and swipe to switch Spaces.
Type or grab the real mouse any time — gestures suspend instantly.

## Setup

```bash
python3.12 -m venv .venv          # Python 3.12 (MediaPipe wheel requirement)
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
.venv/bin/python -m gesture_mouse --list-cameras    # names only, no camera opened
.venv/bin/python -m gesture_mouse --bench-cameras   # opens each: read speed + brightness verdict
.venv/bin/python -m gesture_mouse --camera "FaceTime HD Camera"
```

Names are a *preference*, not gospel: macOS enumeration order is unstable and
can disagree with OpenCV's index table (virtual phone-cam apps like Iriun make
this worse — with no phone connected they serve ~1 fps black frames). At
startup every candidate camera is **probed** (read speed + real image content)
and the first live one wins; the chosen device is printed. Per-camera
`mirror` / `rotate` / `orientation` live in `config.json`.

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
.venv/bin/python -m gesture_mouse                  # IDLE; ⌃⌥G to start
.venv/bin/python -m gesture_mouse --start-active   # camera on immediately
```

| Flag | Effect |
|---|---|
| `--config PATH` | alternate config.json (hot-reloaded on change) |
| `--record FILE` | tee every tracked frame to a JSONL fixture |
| `--replay FILE` | replay a fixture: prints intents, posts nothing |
| `--replay-post` | with `--replay`: actually post the replayed intents |
| `--no-preview` | disable the cv2 tuning window |
| `--no-privacy` | show the camera image in the preview (default: skeleton on black) |

Hotkeys (global, configurable in `config.json`): **⌃⌥G** toggle IDLE⇄ACTIVE ·
**⌃⌥Esc** panic → IDLE, releasing all buttons.

Live-tune keys (preview window focused): `[` / `]` mincutoff down/up ·
`;` / `'` beta down/up · `b` control-box overlay · `p` privacy mode ·
`q` quit. PerfTimer prints per-stage p50/p95 every 5 s.

The corner dot shows state everywhere (incl. fullscreen): gray idle · pulsing
warmup · white clutch-wait · green pointer · blue pinched · purple scroll ·
orange palm · yellow hands-lost · red suspended (M/K = mouse/keyboard reason).

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
| Space left/right | Open palm, swipe left/right | Open palm = system-gesture mode, cursor frozen |
| Mission Control | Open palm, swipe up | |
| App Exposé | Open palm, swipe down | |
| Launchpad | Five-finger pinch-in (open palm → fist) | |
| Show Desktop | Five-finger spread-out (fist → open palm) | |
| Suspend / resume | Grab the real mouse or type / pointer pose 250 ms or ⌃⌥G | See safety model |
| Panic | ⌃⌥Esc | Buttons released, camera off |

Bindings for the six system gestures are remappable in `config.json`
(`bindings`: swipe_left/right/up/down, pinch_in, spread_out → any of
space_prev, space_next, mission_control, app_expose, launchpad, show_desktop).

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
.venv/bin/python -m gesture_mouse --replay fixtures/click.jsonl --no-preview
```

`tools/tune.py` prints jitter_px (RMS motion at rest) and lag_px / lag_ms
(distance behind a fast hand) per parameter combo. Add `--noise-px 1.5` when
sweeping noiseless synthetic fixtures. Everything hot-reloads: edit
`config.json` while running and the new values apply within a second.

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
