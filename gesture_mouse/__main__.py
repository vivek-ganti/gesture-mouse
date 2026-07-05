"""gesture-mouse entry point: argparse, permission preflight, wiring, and the
session FSM around the 30 Hz hot loop (plan doc §State machine, §Pipeline).

Session FSM::

    IDLE (camera off, launch default)
      --toggle hotkey-->  WARMUP (camera opening / exposure ramp)
      --first confident frame or 2 s-->  ACTIVE (hot loop posts intents)
      --guard trip (mouse/keyboard)-->  SUSPENDED (camera on, synthesis muted)
      --engine clutch reacquire (pointer pose 250 ms)-->  ACTIVE
      --toggle / panic hotkey (from anywhere)-->  IDLE

Hotkey callbacks run on the Carbon hotkey thread and ONLY set
``threading.Event`` flags; this main thread polls them and does all real work
(the indicator/preview are main-thread-only). Guards are checked BEFORE any
intent is posted each frame; a trip releases every held button first
(``synth.release_all()``), then the engine forgets its held state WITHOUT
re-emitting UPs (``engine.notify_suspended()``). Central try/finally + atexit
guarantee no exit path leaves a synthetic button down.

Replay mode (``--replay FILE``) runs the identical pipeline from a JSONL
fixture with no permission preflight, no camera, no hotkeys and no synthesis:
intents are printed instead of posted, unless ``--replay-post`` is given.
"""
from __future__ import annotations

import argparse
import atexit
import dataclasses
import signal
import sys
import threading
import time
from pathlib import Path

from . import permissions
from .config import ConfigStore
from .engine import EngineOutput, GestureEngine
from .filters import CursorPipeline
from .guards import Guards
from .hotkeys import pump_events, register_hotkeys
from .indicator import Indicator
from .palm import PalmDetector
from .preview import Preview
from .synth import Synth, real_cursor_pos, screen_size
from .tracker import CameraTracker, Recorder, ReplayTracker, bench_cameras, list_cameras
from .types import EngineState, Intent, LandmarkFrame, SessionState, StateSnapshot

_REPO_ROOT = Path(__file__).resolve().parents[1]

_WARMUP_MAX_S = 2.0        # exposure-ramp ceiling before ACTIVE regardless
_WARMUP_MIN_CONF = 0.5     # a hand this confident ends WARMUP early
_CAMERA_FAIL_MAX_S = 5.0   # continuous grab failure before dropping to IDLE
_IDLE_TICK_S = 0.05        # IDLE poll cadence (camera off; nearly zero CPU)
_RELOAD_PERIOD_S = 1.0     # config.json mtime poll cadence
_CAMERA_REFRESH_PERIOD_S = 3.0  # camera-list re-enumeration cadence
_PERF_PERIOD_S = 5.0       # PerfTimer print cadence
_TUNE_STEP = 1.25          # [ ] multiplicative step for mincutoff
_BETA_STEP = 1.5           # ; ' multiplicative step for beta


class PerfTimer:
    """Per-stage duration collector; prints p50/p95 (ms) every ``period_s``."""

    def __init__(self, period_s: float = _PERF_PERIOD_S) -> None:
        self._period = period_s
        self._t_print = time.monotonic()
        self._order: list[str] = []
        self._samples: dict[str, list[float]] = {}

    def add(self, stage: str, ms: float) -> None:
        bucket = self._samples.get(stage)
        if bucket is None:
            bucket = self._samples[stage] = []
            self._order.append(stage)
        bucket.append(ms)

    @staticmethod
    def _pct(sorted_vals: list[float], q: float) -> float:
        return sorted_vals[int(q * (len(sorted_vals) - 1))]

    def maybe_print(self) -> None:
        now = time.monotonic()
        if now - self._t_print < self._period:
            return
        self._t_print = now
        parts = []
        for stage in self._order:
            vals = sorted(self._samples[stage])
            if not vals:
                continue
            parts.append(
                f"{stage} {self._pct(vals, 0.50):.1f}/{self._pct(vals, 0.95):.1f}"
            )
            self._samples[stage].clear()
        if parts:
            print("[perf p50/p95 ms] " + "  ".join(parts))


def _fmt_intent(intent: Intent) -> str:
    payload = " ".join(
        f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in intent.payload.items()
    )
    return (
        f"[{intent.ts_ms:10.1f} ms] {intent.name:<15} "
        f"{intent.phase.value.upper():<7} {payload}".rstrip()
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="gesture_mouse",
        description="Control the macOS cursor with hand gestures from any camera.",
    )
    ap.add_argument("--list-cameras", action="store_true",
                    help="print available camera names and exit")
    ap.add_argument("--bench-cameras", action="store_true",
                    help="open each camera, measure read latency/brightness, exit")
    ap.add_argument("--camera", metavar="NAME",
                    help="camera to use (overrides config camera.name)")
    ap.add_argument("--config", metavar="PATH",
                    default=str(_REPO_ROOT / "config.json"),
                    help="config.json path (default: repo config.json)")
    ap.add_argument("--replay", metavar="FILE",
                    help="replay a JSONL fixture instead of opening a camera")
    ap.add_argument("--record", metavar="FILE",
                    help="tee every tracked frame to a JSONL fixture")
    ap.add_argument("--no-preview", action="store_true",
                    help="disable the cv2 tuning window")
    ap.add_argument("--no-privacy", action="store_true",
                    help="show the camera image in the preview (default: skeleton on black)")
    ap.add_argument("--start-active", action="store_true",
                    help="open the camera immediately instead of waiting for the toggle hotkey")
    ap.add_argument("--replay-post", action="store_true",
                    help="POST replayed intents as real input (default: print only)")
    return ap.parse_args(argv)


class App:
    """Owns the wiring, the session FSM, and both loop flavors."""

    def __init__(self, store: ConfigStore, args: argparse.Namespace) -> None:
        self.store = store
        self.cfg = store.config
        self.args = args
        self.replay_mode = args.replay is not None
        self.post_events = (not self.replay_mode) or args.replay_post

        try:
            sw, sh = screen_size()
        except Exception:  # headless replay (no window server)
            sw, sh = 1440.0, 900.0
        self.pipeline = CursorPipeline(self.cfg, sw, sh)
        self.palm = PalmDetector(self.cfg.palm, self.cfg.bindings)
        self.engine = GestureEngine(self.cfg, self.pipeline.pinch, self.palm)
        self.synth: Synth | None = Synth() if self.post_events else None
        # Guards exist whenever we post real input — including --replay-post,
        # where physical input must still be able to stop the replay.
        self.guards: Guards | None = (
            Guards(self.cfg.suspend, self.synth) if self.synth is not None else None
        )
        self.preview: Preview | None = None if args.no_preview else Preview(self.cfg)
        self.indicator: Indicator | None = None if self.replay_mode else Indicator()
        self.tracker: CameraTracker | ReplayTracker | None = None
        self.recorder: Recorder | None = None

        self.session = SessionState.IDLE
        self.suspend_reason: str | None = None
        self.toggle_evt = threading.Event()
        self.panic_evt = threading.Event()

        self.perf = PerfTimer()
        self._read_fail_since: float | None = None  # first failed grab (monotonic s)
        self._warmup_deadline = 0.0
        self._drag_state = False     # last EngineOutput.drag (for re-tuning)
        self._fps = 0.0
        self._latency_ms = 0.0
        self._last_loop_t: float | None = None
        self._next_reload = time.monotonic() + _RELOAD_PERIOD_S
        self._quit = False
        self._cleaned = False
        # Camera picker: press 1-9 in the preview window to switch. The list
        # is cheap to enumerate but not free, so it's cached and refreshed
        # only periodically (new devices attach/detach rarely, mid-session).
        self._camera_names: list[str] = [] if self.replay_mode else list_cameras()
        self._next_camera_refresh = time.monotonic() + _CAMERA_REFRESH_PERIOD_S

    # -- lifecycle -----------------------------------------------------------

    def run(self) -> int:
        atexit.register(self.cleanup)
        try:
            if self.replay_mode:
                return self._run_replay()
            return self._run_camera()
        except KeyboardInterrupt:
            print("\ninterrupted — releasing buttons and exiting")
            return 0
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Idempotent teardown; the release_all() call is the atexit invariant
        (no exit path leaves a synthetic button down)."""
        if self._cleaned:
            return
        self._cleaned = True
        if self.synth is not None:
            self.synth.release_all()
        for closer in (self.recorder, self.tracker, self.preview, self.indicator):
            if closer is not None:
                try:
                    closer.close()
                except Exception:
                    pass  # teardown must never mask the original error

    # -- shared pipeline step --------------------------------------------------

    def _step(self, frame: LandmarkFrame) -> EngineOutput:
        """CONTRACTS.md pipeline order: filter -> engine -> apply freeze /
        drag / rebase back to the pipeline (affects the NEXT frame)."""
        t0 = time.perf_counter()
        cursor = self.pipeline.update(frame)
        t1 = time.perf_counter()
        out = self.engine.update(frame, cursor)
        t2 = time.perf_counter()
        self.pipeline.set_frozen(out.freeze)
        self.pipeline.set_drag(out.drag)
        self._drag_state = out.drag
        if out.rebase is not None:
            self.pipeline.rebase(*out.rebase)
        self.perf.add("filter", (t1 - t0) * 1e3)
        self.perf.add("engine", (t2 - t1) * 1e3)
        return out

    def _tee(self, frame: LandmarkFrame) -> None:
        if self.args.record is None:
            return
        if self.recorder is None:  # header needs real dims: first frame
            self.recorder = Recorder(self.args.record, {
                "source": frame.source,
                "mirror": self.cfg.camera.mirror,
                "rotate": self.cfg.camera.rotate,
                "img_w": frame.img_w,
                "img_h": frame.img_h,
                "config": dataclasses.asdict(self.cfg),
            })
        self.recorder.write(frame)

    def _snapshot(self, hand_present: bool) -> StateSnapshot:
        engine_state = (
            None if self.session is SessionState.IDLE else self.engine.state
        )
        return StateSnapshot(
            session_state=self.session,
            engine_state=engine_state,
            hand_present=hand_present,
            fps=self._fps,
            latency_ms=self._latency_ms,
            suspend_reason=self.suspend_reason,
        )

    def _update_rates(self, t_frame_start: float) -> None:
        now = time.perf_counter()
        self._latency_ms = (now - t_frame_start) * 1e3
        if self._last_loop_t is not None:
            dt = now - self._last_loop_t
            if dt > 0.0:
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst
        self._last_loop_t = now

    # -- live tuning -----------------------------------------------------------

    def _apply_filter_tuning(self, msg: str) -> None:
        # set_drag(current state) re-reads mincutoff/drag_mincutoff from cfg;
        # set_beta pushes the new beta into the live x/y filters.
        self.pipeline.set_drag(self._drag_state)
        self.pipeline.set_beta(self.cfg.one_euro.beta)
        print(f"[tune] {msg}")

    def _handle_key(self, key: int) -> None:
        """Live-tune keys from cv2.waitKey: [ ] mincutoff, ; ' beta,
        b box overlay, p privacy, 1-9 switch camera, q quit."""
        if key < 0:
            return
        ch = chr(key & 0xFF) if 32 <= (key & 0xFF) < 127 else ""
        oe = self.cfg.one_euro
        if ch in ("q", "Q"):
            self._quit = True
        elif ch == "[":
            oe.mincutoff = max(0.05, oe.mincutoff / _TUNE_STEP)
            self._apply_filter_tuning(f"mincutoff={oe.mincutoff:.3f}")
        elif ch == "]":
            oe.mincutoff = min(20.0, oe.mincutoff * _TUNE_STEP)
            self._apply_filter_tuning(f"mincutoff={oe.mincutoff:.3f}")
        elif ch == ";":
            oe.beta = oe.beta / _BETA_STEP
            self._apply_filter_tuning(f"beta={oe.beta:.5f}")
        elif ch == "'":
            oe.beta = min(1.0, max(oe.beta, 1e-5) * _BETA_STEP)
            self._apply_filter_tuning(f"beta={oe.beta:.5f}")
        elif ch == "b" and self.preview is not None:
            self.preview.show_box = not self.preview.show_box
            print(f"[tune] control-box overlay {'on' if self.preview.show_box else 'off'}")
        elif ch == "p":
            opts = self.cfg.options
            opts.privacy_preview = not opts.privacy_preview
            print(f"[tune] privacy preview {'on' if opts.privacy_preview else 'off'}")
        elif ch.isdigit() and ch != "0":
            self._switch_camera(int(ch) - 1)

    def _switch_camera(self, index: int) -> None:
        """Digit-key camera picker (see the numbered list drawn in the
        preview). If a camera is currently open, switches it live and resets
        tracking state (a different physical device invalidates the filters'
        motion history and the engine's pinch/hand-loss timers). If not
        (IDLE, or --no-preview never enumerated it), just records the
        preference for the next activation. Either way the choice is
        persisted to config.json so it survives a restart."""
        if index < 0 or index >= len(self._camera_names):
            print(f"[camera] no camera at slot {index + 1}")
            return
        name = self._camera_names[index]
        if isinstance(self.tracker, CameraTracker):
            switched = self.tracker.switch_camera(index)
            if switched is None:
                print(f"[camera] switch to {name!r} failed — staying on "
                      f"{self.tracker.source!r}")
                return
            self.pipeline.reset()
            self.engine.reset()
            self._seed_cursor()
            print(f"[camera] switched to {switched!r}")
        else:
            print(f"[camera] will use {name!r} next time the camera opens")
        self.cfg.camera.name = name
        self.store.save()

    def _maybe_refresh_cameras(self) -> None:
        now = time.monotonic()
        if now < self._next_camera_refresh:
            return
        self._next_camera_refresh = now + _CAMERA_REFRESH_PERIOD_S
        self._camera_names = list_cameras()

    def _current_camera_index(self) -> int | None:
        if not isinstance(self.tracker, CameraTracker):
            return None
        try:
            return self._camera_names.index(self.tracker.source)
        except ValueError:
            return None

    def _maybe_reload_config(self) -> None:
        now = time.monotonic()
        if now < self._next_reload:
            return
        self._next_reload = now + _RELOAD_PERIOD_S
        if self.store.reload_if_changed():
            self._apply_filter_tuning("config.json reloaded")

    # -- session FSM -----------------------------------------------------------

    def _seed_cursor(self) -> None:
        """Start (or resume) control from the real pointer: rebase the mapping
        there and sync synth.last_pos so the mouse-divergence guard does not
        trip on stale coordinates from before the pause."""
        try:
            pos = real_cursor_pos()
        except Exception:
            return
        self.pipeline.rebase(*pos)
        if self.synth is not None:
            self.synth.last_pos = pos

    def _activate(self) -> None:
        assert isinstance(self.tracker, CameraTracker)
        try:
            self.tracker.open()
        except Exception as exc:
            print(f"camera open failed: {exc}", file=sys.stderr)
            self.session = SessionState.IDLE
            return
        self.pipeline.reset()
        self.engine.reset()
        self._seed_cursor()
        self._fps = 0.0
        self._last_loop_t = None
        self.suspend_reason = None
        self.session = SessionState.WARMUP
        self._warmup_deadline = time.monotonic() + _WARMUP_MAX_S
        print("WARMUP — camera opening / exposure ramp")

    def _deactivate(self, why: str) -> None:
        if self.synth is not None:
            self.synth.release_all()
        self.engine.reset()
        self.pipeline.reset()
        if isinstance(self.tracker, CameraTracker):
            self.tracker.close()  # IDLE = zero camera use
        self.session = SessionState.IDLE
        self.suspend_reason = None
        print(f"IDLE ({why}) — camera off; toggle hotkey to resume")

    def _suspend(self, reason: str) -> None:
        """Guard trip: UPs first (synth), then the engine forgets held state
        WITHOUT re-emitting, then park. Camera stays on."""
        if self.synth is not None:
            self.synth.release_all()
        self.engine.notify_suspended()
        self.session = SessionState.SUSPENDED
        self.suspend_reason = reason
        print(
            f"SUSPENDED ({reason}) — resume: pointer pose "
            f"{self.cfg.clutch.reacquire_ms:.0f} ms or toggle hotkey"
        )

    def _resume(self) -> None:
        self.session = SessionState.ACTIVE
        self.suspend_reason = None
        self._seed_cursor()
        if self.guards is not None:
            self.guards.rearm()  # pre-resume keystrokes never count against us
        print("ACTIVE (clutch reacquired)")

    def _guard_trip(self) -> str | None:
        """Checked BEFORE posting intents each ACTIVE frame. Guards.rearm() at
        every ACTIVE entry baselines the keyboard check, so the toggle-hotkey
        keystroke (or the typing that caused a keyboard suspend) is ignored
        while any keyDown *after* activation mutes immediately."""
        if self.guards is None:
            return None
        if self.guards.mouse_moved_physically():
            return "mouse"
        if self.guards.keyboard_active():
            return "keyboard"
        return None

    def _poll_hotkeys(self) -> None:
        # Hotkey events dispatch on THIS thread when the queue is pumped —
        # nothing else pumps with --no-preview, so do it explicitly.
        pump_events()
        if self.panic_evt.is_set():
            self.panic_evt.clear()
            self.toggle_evt.clear()  # panic beats any queued toggle
            if self.session is not SessionState.IDLE:
                self._deactivate("panic")
            return
        if self.toggle_evt.is_set():
            self.toggle_evt.clear()
            if self.session is SessionState.IDLE:
                self._activate()
            else:
                self._deactivate("toggle")

    # -- camera loop -------------------------------------------------------------

    def _run_camera(self) -> int:
        self.tracker = CameraTracker(
            self.cfg, model_path=str(_REPO_ROOT / "hand_landmarker.task")
        )
        register_hotkeys(self.toggle_evt.set, self.panic_evt.set, self.cfg.hotkeys)
        print(
            f"hotkeys: toggle={self.cfg.hotkeys.toggle}  "
            f"panic={self.cfg.hotkeys.panic}"
        )
        if self.args.start_active:
            self._activate()
        else:
            print("IDLE — camera off; press the toggle hotkey to start")
        while not self._quit:
            self._poll_hotkeys()
            if self.session is SessionState.IDLE:
                self._idle_tick()
            else:
                self._frame_tick()
        return 0

    def _idle_tick(self) -> None:
        self._maybe_refresh_cameras()
        snap = self._snapshot(hand_present=False)
        if self.indicator is not None:
            self.indicator.set_state(snap)
        if self.preview is not None:
            key = self.preview.show(
                None, None, snap, {}, cameras=self._camera_names,
                current_camera_index=self._current_camera_index(),
            )
            self._handle_key(key)
        time.sleep(_IDLE_TICK_S)

    def _frame_tick(self) -> None:
        assert isinstance(self.tracker, CameraTracker)
        t0 = time.perf_counter()
        frame = self.tracker.read()
        t1 = time.perf_counter()
        self.perf.add("read", (t1 - t0) * 1e3)
        if frame is None:
            self._camera_fail_tick()
            return
        self._read_fail_since = None
        self._tee(frame)

        if self.session is SessionState.WARMUP and (
            (frame.hand_present and frame.confidence >= _WARMUP_MIN_CONF)
            or time.monotonic() >= self._warmup_deadline
        ):
            self.session = SessionState.ACTIVE
            self._seed_cursor()
            if self.guards is not None:
                self.guards.rearm()  # the toggle keystroke predates this moment
            print("ACTIVE — clutch: hold the pointer pose "
                  f"{self.cfg.clutch.engage_ms:.0f} ms")

        if self.session is SessionState.WARMUP:
            # The engine deliberately does NOT tick during WARMUP: its intents
            # would be dropped, leaving orphan held-button state (a dropped
            # left DOWN) for ACTIVE to unwind with an orphan drag/UP stream.
            self._finish_tick(t0, frame)
            return

        out = self._step(frame)

        t2 = time.perf_counter()
        if self.session is SessionState.ACTIVE:
            reason = self._guard_trip()   # BEFORE posting, every frame
            if reason is not None:
                self._suspend(reason)
            elif self.synth is not None:
                for intent in out.intents:
                    self.synth.execute(intent)
        elif self.session is SessionState.SUSPENDED:
            # Engine left its reacquire-clutch: the user held the pointer
            # pose for clutch.reacquire_ms. (Toggle is the other way out.)
            if self.engine.state is EngineState.POINTER:
                self._resume()
        self.perf.add("synth", (time.perf_counter() - t2) * 1e3)
        self._finish_tick(t0, frame)

    def _camera_fail_tick(self) -> None:
        """Grab failed. The engine cannot tick without a frame, but the safety
        invariants still must: release any held button within hands_lost_ms,
        keep hotkeys/UI responsive (no busy-spin), and give up on a camera
        that stays dead (unplugged mid-session)."""
        now = time.monotonic()
        if self._read_fail_since is None:
            self._read_fail_since = now
        fail_ms = (now - self._read_fail_since) * 1000.0
        if fail_ms >= _CAMERA_FAIL_MAX_S * 1000.0:
            self._read_fail_since = None
            self._deactivate("camera failure")
            return
        if fail_ms >= self.cfg.hands_lost_ms and self.session is SessionState.ACTIVE:
            self._suspend("camera")  # release_all + engine.notify_suspended
        self._maybe_refresh_cameras()
        snap = self._snapshot(hand_present=False)
        if self.indicator is not None:
            self.indicator.set_state(snap)
        if self.preview is not None:
            key = self.preview.show(
                None, None, snap, {}, cameras=self._camera_names,
                current_camera_index=self._current_camera_index(),
            )
            self._handle_key(key)
        time.sleep(0.01)

    def _finish_tick(self, t0: float, frame: LandmarkFrame) -> None:
        t3 = time.perf_counter()
        self._update_rates(t0)
        self._maybe_refresh_cameras()
        snap = self._snapshot(frame.hand_present)
        if self.indicator is not None:
            self.indicator.set_state(snap)
        if self.preview is not None:
            assert isinstance(self.tracker, CameraTracker)
            bgr = None if self.cfg.options.privacy_preview else self.tracker.last_bgr
            key = self.preview.show(
                bgr, frame, snap, self.engine.pinch_values,
                palm_debug=self.engine.palm_debug, cameras=self._camera_names,
                current_camera_index=self._current_camera_index(),
            )
            self._handle_key(key)
        self.perf.add("ui", (time.perf_counter() - t3) * 1e3)
        self.perf.maybe_print()
        self._maybe_reload_config()

    # -- replay loop ------------------------------------------------------------

    def _run_replay(self) -> int:
        self.tracker = ReplayTracker(self.args.replay)
        header = self.tracker.header
        mode = "POSTING" if self.post_events else "printing intents"
        print(
            f"replay: {self.args.replay} "
            f"(source={header.get('source')!r}, {mode})"
        )
        if self.post_events:
            # A posting replay is real synthetic input: the panic/toggle
            # hotkeys and the physical-input guards must be able to stop it.
            register_hotkeys(self.toggle_evt.set, self.panic_evt.set, self.cfg.hotkeys)
            if self.guards is not None:
                self.guards.rearm()
        self.session = SessionState.ACTIVE
        n_frames = n_intents = 0
        prev_ts: float | None = None
        while not self._quit:
            t0 = time.perf_counter()
            frame = self.tracker.read()
            self.perf.add("read", (time.perf_counter() - t0) * 1e3)
            if frame is None:
                break
            if self.post_events:
                # Pace posting to the recorded timeline: unthrottled posting
                # floods events faster than the OS applies them to the cursor
                # (breaking the mouse guard's recent-post window) and would
                # replay drags at physically impossible speed.
                if prev_ts is not None:
                    time.sleep(max(0.0, min(0.1, (frame.ts_ms - prev_ts) / 1000.0)))
                prev_ts = frame.ts_ms
            self._tee(frame)
            out = self._step(frame)
            if self.post_events:
                pump_events()  # hotkey delivery: nothing else pumps in replay
                if self.panic_evt.is_set() or self.toggle_evt.is_set():
                    print("replay stopped by hotkey")
                    break
                trip = None
                if self.guards is not None:
                    if self.guards.mouse_moved_physically():
                        trip = "mouse"
                    elif self.guards.keyboard_active():
                        trip = "keyboard"
                if trip is not None:
                    if self.synth is not None:
                        self.synth.release_all()
                    print(f"replay stopped: physical input detected ({trip})")
                    break
            for intent in out.intents:
                n_intents += 1
                if self.post_events and self.synth is not None:
                    self.synth.execute(intent)
                else:
                    print(_fmt_intent(intent))
            self._update_rates(t0)
            if self.preview is not None:
                snap = self._snapshot(frame.hand_present)
                self._handle_key(
                    self.preview.show(None, frame, snap, self.engine.pinch_values,
                                       palm_debug=self.engine.palm_debug)
                )
            self.perf.maybe_print()
            n_frames += 1
        print(f"replay done: {n_frames} frames, {n_intents} intents, "
              f"final engine state {self.engine.state.value}")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Default-action SIGTERM (`kill`, session logout) would skip atexit and
    # the run() finally block, leaving a mid-drag synthetic button down
    # system-wide; converting it to SystemExit runs both.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    if args.list_cameras:  # no permission preflight, no camera open
        cams = list_cameras()
        if not cams:
            print("no cameras found")
            return 1
        for i, name in enumerate(cams):
            print(f"[{i}] {name}")
        return 0

    if args.bench_cameras:
        bench_cameras()
        return 0

    store = ConfigStore(args.config)
    cfg = store.config
    if args.camera:
        cfg.camera.name = args.camera
    if args.no_privacy:
        cfg.options.privacy_preview = False

    if args.replay is None:  # preflight skipped for --replay/--list-cameras
        if not permissions.preflight(require=True):
            return 1

    return App(store, args).run()


if __name__ == "__main__":
    raise SystemExit(main())
