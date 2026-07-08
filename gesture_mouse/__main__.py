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
import math
import signal
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import permissions, signatures
from .calibration import STEPS as CALIB_STEPS
from .calibration import CalibrationResult, CalibrationSession
from .config import ConfigStore
from .engine import EngineOutput, GestureEngine
from .filters import CursorPipeline
from .guards import Guards
from .hotkeys import pump_events, register_hotkeys
from .indicator import Indicator
from .palm import PalmDetector
from .panel import PanelCommand, PanelServer, frame_event
from .preview import Preview
from .synth import Synth, custom_action_argv, real_cursor_pos, screen_size
from .tracker import CameraTracker, Recorder, ReplayTracker, bench_cameras, list_cameras
from .types import (
    INDEX_TIP,
    MIDDLE_TIP,
    EngineState,
    Intent,
    LandmarkFrame,
    Phase,
    SessionState,
    StateSnapshot,
    dist,
)

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
_POSE_SMOOTH_STEP = 1.25   # , . multiplicative step for pose smoothing mincutoff
_ANGLE_STEP = 5.0          # - = additive step (deg) for pose extend/curl thresholds
_MIN_ANGLE_GAP = 5.0       # keep curl_angle_deg clearly below extend_angle_deg
_CAPTURE_STABLE_MS = 1000.0   # pose must hold this long to freeze a capture
_CAPTURE_TIMEOUT_S = 20.0     # capture mode auto-cancels after this
_TOAST_TTL_S = 3.0            # a toast rides frame events this long (dedup by id)

# Panel set_setting whitelist: dotted config path -> (min, max, kind).
# kind: "bool" toggles; "filter"/"pose" additionally re-apply live tuning
# (same callbacks the keyboard live-tune keys use); None = plain clamp+set.
# Anything not listed here is NOT settable from the browser — the panel is
# localhost-only and token-guarded, but shell-reachable surface still stays
# minimal by construction.
_SETTINGS: dict[str, tuple[float, float, str | None]] = {
    "pose.extend_angle_deg": (100.0, 178.0, "pose"),
    "pose.curl_angle_deg": (2.0, 150.0, "pose"),
    "pose.smoothing_mincutoff": (0.1, 20.0, "pose"),
    "pose_jitter_grace_ms": (0.0, 500.0, None),
    "pinch.left_engage": (0.05, 1.0, None),
    "pinch.left_release": (0.05, 1.2, None),
    "pinch.right_engage": (0.05, 1.0, None),
    "pinch.right_release": (0.05, 1.2, None),
    "scroll.gain": (10.0, 2000.0, None),
    "scroll.together_max": (0.1, 1.0, None),
    "scroll.invert": (0.0, 1.0, "bool"),
    "one_euro.mincutoff": (0.05, 20.0, "filter"),
    "one_euro.beta": (0.00005, 1.0, "filter"),
    "palm.forward_max_speed_px_s": (30.0, 1500.0, None),
    "options.privacy_preview": (0.0, 1.0, "bool"),
}


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
    ap.add_argument("--preview", action="store_true",
                    help="open the cv2 debug window (the web panel is the primary UI)")
    ap.add_argument("--no-preview", action="store_true",
                    help=argparse.SUPPRESS)  # deprecated no-op: off is the default now
    ap.add_argument("--no-privacy", action="store_true",
                    help="show the camera image in the cv2 debug window "
                         "(default: skeleton on black; the web panel never shows images)")
    ap.add_argument("--panel-port", type=int, metavar="N",
                    help="web control panel port (default: config panel.port, 8765)")
    ap.add_argument("--no-panel", action="store_true",
                    help="disable the web control panel entirely")
    ap.add_argument("--no-open", action="store_true",
                    help="don't auto-open the panel in the browser")
    ap.add_argument("--start-active", action="store_true",
                    help="open the camera immediately instead of waiting for the toggle hotkey")
    ap.add_argument("--debug-gestures", action="store_true",
                    help="print swipe arming/candidate/rejection events live")
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
        self.palm = PalmDetector(self.cfg.palm, self.cfg.bindings,
                                 debug=self.cfg.options.debug_gestures)
        self.engine = GestureEngine(self.cfg, self.pipeline.pinch, self.palm)
        for skipped in self.engine.custom_skipped:
            print(f"[config] custom gesture {skipped!r} skipped "
                  f"(unknown pose or missing action) — see README")
        self.synth: Synth | None = Synth() if self.post_events else None
        # Guards exist whenever we post real input — including --replay-post,
        # where physical input must still be able to stop the replay.
        self.guards: Guards | None = (
            Guards(self.cfg.suspend, self.synth) if self.synth is not None else None
        )
        # cv2 window is opt-in since the web panel became the primary UI;
        # --no-preview is accepted as a deprecated no-op.
        if args.no_preview:
            print("[deprecated] --no-preview is now the default; "
                  "use --preview to open the cv2 debug window")
        self.preview: Preview | None = Preview(self.cfg) if args.preview else None
        self.indicator: Indicator | None = None if self.replay_mode else Indicator()
        self.tracker: CameraTracker | ReplayTracker | None = None
        self.recorder: Recorder | None = None

        self.session = SessionState.IDLE
        self.suspend_reason: str | None = None
        self.toggle_evt = threading.Event()
        self.panic_evt = threading.Event()

        # Web control panel (primary UI). Server thread touches nothing but
        # its own state; all effects flow through poll_commands() drained on
        # THIS thread; all data flows out via publish().
        self.panel: PanelServer | None = None
        if not args.no_panel:
            port = args.panel_port if args.panel_port else self.cfg.panel.port
            html = Path(__file__).resolve().parent / "web" / "index.html"
            self.panel = PanelServer(port, html)
        # ui_mode gates synthesis: while calibrating/capturing the engine and
        # pipeline still tick (the wizard needs live angles) but guards are
        # skipped and NO intents are posted — the user is deliberately using
        # the real mouse in the browser during these modes.
        self.ui_mode: str = "normal"
        self.calib: CalibrationSession | None = None
        self._calib_result_obj: CalibrationResult | None = None
        self._calib_result: dict | None = None
        self._capture: dict | None = None
        self._toast: dict | None = None
        self._toast_until = 0.0
        self._toast_id = 0
        self._config_dirty = True   # first publish always sends config
        self._last_ts_ms = 0.0

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
        for closer in (self.recorder, self.tracker, self.preview,
                       self.indicator, self.panel):
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
        if self.cfg.options.debug_gestures:
            while self.palm.events:
                ts, msg = self.palm.events.popleft()
                print(f"[swipe {ts:8.0f}ms] {msg}")
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
        self._config_dirty = True   # keep the panel's sliders in sync
        print(f"[tune] {msg}")

    def _apply_pose_tuning(self, msg: str) -> None:
        self.engine.retune_pose_smoothing()
        self._config_dirty = True
        print(f"[tune] {msg}")

    def _handle_key(self, key: int) -> None:
        """Live-tune keys from cv2.waitKey: [ ] mincutoff, ; ' beta,
        - = pose extend/curl angle thresholds, , . pose smoothing mincutoff,
        b box overlay, p privacy, 1-9 switch camera, q quit."""
        if key < 0:
            return
        ch = chr(key & 0xFF) if 32 <= (key & 0xFF) < 127 else ""
        oe = self.cfg.one_euro
        p = self.cfg.pose
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
        elif ch in ("-", "="):
            # More forgiving (-) / stricter (=): shift the angle-hysteresis
            # thresholds. Applies to the per-finger CALIBRATED pairs too when
            # present — otherwise calibration would silently make these keys
            # inert for calibrated fingers.
            delta = -_ANGLE_STEP if ch == "-" else _ANGLE_STEP
            if ch == "-":
                p.extend_angle_deg = max(p.curl_angle_deg + _MIN_ANGLE_GAP,
                                          p.extend_angle_deg + delta)
                p.curl_angle_deg = max(0.0, p.curl_angle_deg + delta)
            else:
                p.extend_angle_deg = min(180.0, p.extend_angle_deg + delta)
                p.curl_angle_deg = min(p.extend_angle_deg - _MIN_ANGLE_GAP,
                                        p.curl_angle_deg + delta)
            self._shift_finger_pairs(delta)
            per = ("  (+ per-finger calibrated pairs)" if p.fingers else "")
            self._config_dirty = True
            print(f"[tune] pose extend={p.extend_angle_deg:.0f} "
                  f"curl={p.curl_angle_deg:.0f}{per}")
        elif ch == ",":
            p.smoothing_mincutoff = max(0.1, p.smoothing_mincutoff / _POSE_SMOOTH_STEP)
            self._apply_pose_tuning(f"pose smoothing mincutoff={p.smoothing_mincutoff:.3f}")
        elif ch == ".":
            p.smoothing_mincutoff = min(20.0, p.smoothing_mincutoff * _POSE_SMOOTH_STEP)
            self._apply_pose_tuning(f"pose smoothing mincutoff={p.smoothing_mincutoff:.3f}")
        elif ch == "b" and self.preview is not None:
            self.preview.show_box = not self.preview.show_box
            print(f"[tune] control-box overlay {'on' if self.preview.show_box else 'off'}")
        elif ch in ("h", "H", "?") and self.preview is not None:
            self.preview.show_help = not self.preview.show_help
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
            self._toast_msg("warn", f"no camera at slot {index + 1}")
            return
        name = self._camera_names[index]
        # Live-switch only with the camera actually OPEN: while IDLE the
        # tracker object exists but is closed, and switch_camera() on it
        # fails — which used to bail before the persist below, silently
        # ignoring a camera picked from the panel while the camera was off.
        if isinstance(self.tracker, CameraTracker) and self.session is not SessionState.IDLE:
            switched = self.tracker.switch_camera(index)
            if switched is None:
                self._toast_msg("error", f"switch to {name!r} failed — staying "
                                         f"on {self.tracker.source!r}")
                return
            self.pipeline.reset()
            self.engine.reset()
            self._seed_cursor()
            self._toast_msg("info", f"switched to {switched!r}")
        else:
            self._toast_msg("info", f"will use {name!r} when the camera starts")
        self.cfg.camera.name = name
        self.store.save()
        self._config_dirty = True

    def _maybe_refresh_cameras(self) -> None:
        now = time.monotonic()
        if now < self._next_camera_refresh:
            return
        self._next_camera_refresh = now + _CAMERA_REFRESH_PERIOD_S
        fresh = list_cameras()
        if fresh != self._camera_names:
            self._camera_names = fresh
            self._config_dirty = True   # panel camera list

    def _current_camera_index(self) -> int | None:
        if not isinstance(self.tracker, CameraTracker):
            return None
        try:
            return self._camera_names.index(self.tracker.source)
        except ValueError:
            return None

    def _reload_config_now(self) -> None:
        """Unthrottled reload-and-reapply: also called right before every
        panel-driven store.save() so a pending external config.json edit is
        merged in rather than clobbered by the full-file rewrite."""
        if self.store.reload_if_changed():
            # External edit (or another tool): re-apply live tuning AND
            # re-parse custom gestures so config.json edits hot-apply without
            # a session reset (panel edits go through the same paths).
            self.engine.reload_customs()
            self.engine.retune_pose_smoothing()
            self._apply_filter_tuning("config.json reloaded")

    def _maybe_reload_config(self) -> None:
        now = time.monotonic()
        if now < self._next_reload:
            return
        self._next_reload = now + _RELOAD_PERIOD_S
        self._reload_config_now()

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
        # Any exit to IDLE ends calibration/capture too — EVERY deactivation
        # path (panel stop, toggle/panic hotkey, camera failure) must clear
        # ui_mode, or the next session would silently drop all intents and
        # skip guards until a panel Cancel.
        self._exit_ui_mode()
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

    # -- web panel: lifecycle, commands, publishing -------------------------------

    def _start_panel(self) -> None:
        if self.panel is None:
            return
        try:
            url = self.panel.start()
        except Exception as exc:
            print(f"panel failed to start: {exc}", file=sys.stderr)
            self.panel = None
            return
        print(f"panel: {url}")
        if (self.cfg.panel.open_browser and not self.args.no_open
                and not self.replay_mode):
            try:
                webbrowser.open(url)
            except Exception:
                pass  # headless/no default browser: the printed URL suffices

    def _toast_msg(self, level: str, text: str) -> None:
        """Queue a toast for the panel; rides every frame event for a few
        seconds (frame events are throttled/droppable, so a single-frame
        toast could be lost) — the frontend dedupes by id."""
        self._toast_id += 1
        self._toast = {"id": self._toast_id, "level": level, "text": text}
        self._toast_until = time.monotonic() + _TOAST_TTL_S
        print(f"[panel] {level}: {text}")

    def _poll_panel(self) -> None:
        if self.panel is None:
            return
        for cmd in self.panel.poll_commands():
            try:
                self._dispatch_panel(cmd)
            except Exception as exc:  # a bad command must never kill the loop
                self._toast_msg("error", f"{cmd.type} failed: {exc}")

    def _dispatch_panel(self, cmd: PanelCommand) -> None:
        t, pl = cmd.type, cmd.payload
        if t == "camera_start":
            if not self.replay_mode and self.session is SessionState.IDLE:
                self._activate()
        elif t == "camera_stop":
            self._exit_ui_mode()  # replay mode: _deactivate below is skipped
            if not self.replay_mode and self.session is not SessionState.IDLE:
                self._deactivate("panel")
        elif t == "camera_switch":
            idx = pl.get("index")
            if isinstance(idx, int) and not self.replay_mode:
                self._switch_camera(idx)
                self._config_dirty = True
        elif t == "panic":
            self.panic_evt.set()
        elif t == "quit":
            self._quit = True
        elif t == "set_setting":
            self._apply_set_setting(str(pl.get("path", "")), pl.get("value"))
        elif t == "calibrate_start":
            self._calibrate_start()
        elif t == "calibrate_begin_step":
            if self.calib is not None:
                self.calib.begin_step(str(pl.get("step", "")), self._last_ts_ms)
                # A (re)begun step invalidates any computed results — without
                # this, "Redo this step" after completing all six would show
                # and APPLY thresholds from the superseded take.
                self._calib_result_obj = None
                self._calib_result = None
        elif t == "calibrate_cancel" or t == "capture_cancel":
            self._exit_ui_mode(reset=True)
        elif t == "calibrate_apply":
            self._calibrate_apply()
        elif t == "capture_start":
            self._capture_start()
        elif t == "save_gesture":
            self._save_gesture(pl)
        elif t == "delete_gesture":
            self._delete_gesture(str(pl.get("name", "")))
        elif t == "test_gesture":
            self._test_gesture(str(pl.get("name", "")))

    def _shift_finger_pairs(self, delta: float) -> None:
        """Shift every per-finger calibrated extend/curl pair by delta,
        keeping each pair's gap invariant — shared by the '-'/'=' keyboard
        tuner and the panel's global pose-angle sliders, so calibration
        never makes either control silently inert."""
        for name, th in self.cfg.pose.fingers.items():
            if isinstance(th, dict) and "extend" in th and "curl" in th:
                th["extend"] = max(2.0, min(180.0, float(th["extend"]) + delta))
                th["curl"] = max(0.0, min(th["extend"] - _MIN_ANGLE_GAP,
                                          float(th["curl"]) + delta))

    def _apply_set_setting(self, path: str, value) -> None:
        spec = _SETTINGS.get(path)
        if spec is None:
            self._toast_msg("error", f"setting {path!r} is not adjustable")
            return
        # Pick up any pending external config.json edit BEFORE mutating and
        # saving, or save() would silently clobber it (save() rewrites the
        # whole file from memory and bumps the stored mtime).
        self._reload_config_now()
        lo, hi, kind = spec
        obj = self.cfg
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        if kind == "bool":
            val: object = bool(value)
        else:
            try:
                fval = float(value)
            except (TypeError, ValueError):
                self._toast_msg("error", f"bad value for {path}: {value!r}")
                return
            if fval != fval or fval in (float("inf"), float("-inf")):
                self._toast_msg("error", f"bad value for {path}: {value!r}")
                return
            val = max(lo, min(hi, fval))
            # The extend/curl PAIR invariant (extend clearly above curl) is
            # what FingerLatch's within-frame idempotence rests on — enforce
            # it here exactly like the keyboard tuner does, and mirror the
            # global shift onto calibrated per-finger pairs.
            if path == "pose.extend_angle_deg":
                val = max(val, self.cfg.pose.curl_angle_deg + _MIN_ANGLE_GAP)
                self._shift_finger_pairs(val - self.cfg.pose.extend_angle_deg)
            elif path == "pose.curl_angle_deg":
                val = min(val, self.cfg.pose.extend_angle_deg - _MIN_ANGLE_GAP)
                self._shift_finger_pairs(val - self.cfg.pose.curl_angle_deg)
        setattr(obj, parts[-1], val)
        if kind == "filter":
            self._apply_filter_tuning(f"{path}={val}")
        elif kind == "pose":
            self._apply_pose_tuning(f"{path}={val}")
        self.store.save()
        self._config_dirty = True

    # -- calibration + capture modes ----------------------------------------

    def _enter_ui_mode(self, mode: str) -> bool:
        """Common entry ritual: nothing may stay held while synthesis is
        muted, and the engine restarts from a clean clutch on exit."""
        if self.session is SessionState.IDLE and not self.replay_mode:
            self._toast_msg("warn", "start the camera first")
            return False
        if self.synth is not None:
            self.synth.release_all()
        self.engine.notify_suspended()
        self.ui_mode = mode
        return True

    def _exit_ui_mode(self, reset: bool = False) -> None:
        was = self.ui_mode
        self.ui_mode = "normal"
        self.calib = None
        self._calib_result_obj = None
        self._calib_result = None
        self._capture = None
        if was != "normal" and reset:
            self.engine.reset()
            # The user was just mousing in the browser: re-baseline the
            # divergence guard or normal mode would instantly suspend.
            self._seed_cursor()
            if self.guards is not None:
                self.guards.rearm()

    def _calibrate_start(self) -> None:
        if self.cfg.options.extended_test != "radial":
            self._toast_msg("warn", "calibration needs the angle-based finger "
                                    "test (options.extended_test='radial')")
            return
        if not self._enter_ui_mode("calibrating"):
            return
        self.calib = CalibrationSession()
        self._calib_result_obj = None
        self._calib_result = None
        self._toast_msg("info", "calibration started — run all six steps")

    def _calibrate_apply(self) -> None:
        if self._calib_result_obj is None:
            self._toast_msg("warn", "no calibration results to apply yet")
            return
        applied = []
        for f, fr in self._calib_result_obj.fingers.items():
            if fr.extend is None or fr.curl is None:
                continue
            entry: dict = {"extend": fr.extend, "curl": fr.curl}
            if f == "thumb":
                entry["advisory"] = True  # stored, never gates (see PoseConfig)
            self.cfg.pose.fingers[f] = entry
            applied.append(f)
        self.store.save()
        self._config_dirty = True
        self._toast_msg("info", "calibration applied: " +
                        (", ".join(applied) if applied else "no fingers (kept defaults)"))
        self._exit_ui_mode(reset=True)

    def _capture_start(self) -> None:
        if not self._enter_ui_mode("capturing"):
            return
        self._capture = {"sig": None, "since": None,
                         "start": time.monotonic(), "done": False,
                         "conflicts": [], "stable_ms": 0.0}

    def _known_signatures(self, excluding: str | None = None) -> dict:
        named = dict(signatures.BUILTINS)
        parsed, _ = signatures.normalize_custom_entries(self.cfg.custom_gestures)
        for g in parsed:
            if g["name"] != excluding:
                named[g["name"]] = g["signature"]
        return named

    def _ui_mode_tick(self, frame: LandmarkFrame) -> None:
        """Per-frame work for the active ui_mode (called from both loops)."""
        self._last_ts_ms = frame.ts_ms
        if self.ui_mode == "calibrating" and self.calib is not None:
            self.calib.add_sample(frame.ts_ms, self.engine.finger_angles)
            if self.calib.state == "done" and self._calib_result_obj is None:
                self._calib_result_obj = self.calib.compute(
                    (self.cfg.pose.extend_angle_deg, self.cfg.pose.curl_angle_deg),
                    {n: s for n, s in self._known_signatures().items()
                     if n not in signatures.BUILTINS},
                )
                self._calib_result = self._calib_result_obj.as_dict()
        elif self.ui_mode == "capturing" and self._capture is not None:
            cap = self._capture
            if cap["done"]:
                return
            if time.monotonic() - cap["start"] > _CAPTURE_TIMEOUT_S:
                self._toast_msg("warn", "capture timed out — try again")
                self._exit_ui_mode(reset=True)
                return
            if not self.engine.finger_angles:
                cap["sig"] = None
                cap["since"] = None
                cap["stable_ms"] = 0.0
                return
            sig = signatures.signature_from_states(self.engine.finger_states)
            if sig != cap["sig"]:
                cap["sig"] = sig
                cap["since"] = frame.ts_ms
                cap["stable_ms"] = 0.0
                return
            cap["stable_ms"] = frame.ts_ms - (cap["since"] or frame.ts_ms)
            if cap["stable_ms"] >= _CAPTURE_STABLE_MS:
                cap["done"] = True
                cap["conflicts"] = signatures.check_conflicts(
                    sig, self._known_signatures())

    def _save_gesture(self, pl: dict) -> None:
        name = str(pl.get("name", "")).strip()
        sig = signatures.normalize_signature(pl.get("signature"))
        action = pl.get("action") if isinstance(pl.get("action"), dict) else None
        if not name or sig is None or action is None:
            self._toast_msg("error", "invalid gesture: needs a name, a "
                                     "signature and an action")
            return
        if name in signatures.BUILTINS:
            self._toast_msg("error", f"{name!r} is a built-in gesture name")
            return
        if custom_action_argv(action) is None:
            self._toast_msg("error", "invalid action (unknown type/key/trigger)")
            return
        conflicts = signatures.check_conflicts(
            sig, self._known_signatures(excluding=name))
        if conflicts:
            self._toast_msg("error", "pose collides with: " + ", ".join(conflicts))
            return
        try:
            hold_ms = float(pl.get("hold_ms", 300.0))
            cooldown_ms = float(pl.get("cooldown_ms", 1200.0))
            # json.loads happily parses NaN/Infinity (and 1e999 -> inf);
            # non-finite values would make config.json unparseable and wedge
            # every panel client's strict JSON.parse.
            if not (math.isfinite(hold_ms) and math.isfinite(cooldown_ms)):
                raise ValueError
        except (TypeError, ValueError):
            self._toast_msg("error", "hold/cooldown must be finite numbers")
            return
        hold_ms = max(0.0, min(60_000.0, hold_ms))
        cooldown_ms = max(0.0, min(600_000.0, cooldown_ms))
        entry = {"name": name, "signature": sig, "hold_ms": hold_ms,
                 "cooldown_ms": cooldown_ms, "action": dict(action)}
        self._reload_config_now()   # don't clobber a pending external edit
        kept = [g for g in self.cfg.custom_gestures
                if not (isinstance(g, dict)
                        and str(g.get("name", g.get("pose", ""))) == name)]
        kept.append(entry)
        self.cfg.custom_gestures[:] = kept   # in-place: modules hold this list
        self.store.save()
        self.engine.reload_customs()
        self._config_dirty = True
        self._toast_msg("info", f"gesture {name!r} saved")
        if self.ui_mode == "capturing":
            self._exit_ui_mode(reset=True)

    def _test_gesture(self, name: str) -> None:
        """Fire a saved gesture's ACTION right now, without doing the pose —
        the panel's 'Test' button. Answers "did I bind the right action"
        separately from "does the pose register" (the Live tab's per-finger
        readout + hold chip answer the latter)."""
        parsed, _ = signatures.normalize_custom_entries(self.cfg.custom_gestures)
        entry = next((g for g in parsed if g["name"] == name), None)
        if entry is None:
            self._toast_msg("warn", f"no gesture named {name!r}")
            return
        if self.synth is None:
            self._toast_msg("warn", "actions don't post in replay mode")
            return
        self.synth.execute(Intent(
            f"custom:{name}", Phase.TRIGGER, {"action": entry["action"]},
            self._last_ts_ms,
        ))
        self._toast_msg("info", f"tested {name!r} — action posted")

    def _delete_gesture(self, name: str) -> None:
        self._reload_config_now()   # don't clobber a pending external edit
        kept = [g for g in self.cfg.custom_gestures
                if not (isinstance(g, dict)
                        and str(g.get("name", g.get("pose", ""))) == name)]
        if len(kept) == len(self.cfg.custom_gestures):
            self._toast_msg("warn", f"no gesture named {name!r}")
            return
        self.cfg.custom_gestures[:] = kept
        self.store.save()
        self.engine.reload_customs()
        self._config_dirty = True
        self._toast_msg("info", f"gesture {name!r} deleted")

    # -- panel publishing -----------------------------------------------------

    def _config_event(self) -> dict:
        parsed, _ = signatures.normalize_custom_entries(self.cfg.custom_gestures)
        settings: dict = {}
        for path in _SETTINGS:
            obj = self.cfg
            for part in path.split("."):
                obj = getattr(obj, part)
            settings[path] = obj
        return {
            "settings": settings,
            "custom_gestures": parsed,
            "builtin_gestures": [
                {"name": n, "signature": dict(s), "editable": False}
                for n, s in signatures.BUILTINS.items()
            ],
            "cameras": list(self._camera_names),
            "camera_index": self._current_camera_index(),
            "calib_defaults": {"extend": self.cfg.pose.extend_angle_deg,
                               "curl": self.cfg.pose.curl_angle_deg},
            "calib_steps": [
                {"id": s.id, "label": s.label, "instruction": s.instruction,
                 "expected": dict(s.expected)}
                for s in CALIB_STEPS
            ],
            "pose_fingers": {k: dict(v) for k, v in self.cfg.pose.fingers.items()
                             if isinstance(v, dict)},
        }

    def _publish_panel(self, frame: LandmarkFrame | None, snap: StateSnapshot) -> None:
        if self.panel is None:
            return
        if self._config_dirty:
            # Config publishes even with zero clients: the server replays the
            # last config event to late joiners, so this primes reconnects.
            self.panel.publish("config", self._config_event())
            self._config_dirty = False
        if not self.panel.has_clients():
            return
        lm = self.engine.smoothed_landmarks if (frame is not None
                                                and frame.hand_present) else None
        # Scroll's extra gate made visible: normalized index-middle fingertip
        # distance vs scroll.together_max — without this in the Live view,
        # "scroll won't start" is undiagnosable (the per-finger rows can all
        # be right while the fingertips are simply too far apart).
        together = None
        if lm is not None and frame is not None and frame.scale > 0:
            together = dist(lm[INDEX_TIP], lm[MIDDLE_TIP]) / frame.scale
        thresholds = {}
        for f in signatures.FINGERS:
            per = self.cfg.pose.fingers.get(f)
            calibrated = isinstance(per, dict) and "extend" in per and "curl" in per
            if calibrated:
                ext_at, curl_at = float(per["extend"]), float(per["curl"])
            else:
                ext_at = self.cfg.pose.extend_angle_deg
                curl_at = self.cfg.pose.curl_angle_deg
            thresholds[f] = {"extend": ext_at, "curl": curl_at,
                             "calibrated": calibrated}
        pv = self.engine.pinch_values
        pd = self.engine.palm_debug
        calibration = None
        if self.calib is not None:
            calibration = self.calib.progress()
            calibration["results"] = self._calib_result
        capture = None
        if self._capture is not None:
            capture = {"active": True, "signature": self._capture["sig"],
                       "stable_ms": self._capture["stable_ms"],
                       "done": self._capture["done"],
                       "conflicts": self._capture["conflicts"]}
        toast = self._toast if time.monotonic() < self._toast_until else None
        p = self.cfg.pinch
        ev = frame_event(
            ts=frame.ts_ms if frame is not None else self._last_ts_ms,
            session=snap.session_state.value,
            engine=snap.engine_state.value if snap.engine_state else None,
            hand=snap.hand_present,
            fps=snap.fps,
            latency_ms=snap.latency_ms,
            suspend_reason=snap.suspend_reason,
            img_w=frame.img_w if frame is not None else self.cfg.camera.width,
            img_h=frame.img_h if frame is not None else self.cfg.camera.height,
            landmarks=[[pt.x, pt.y] for pt in lm] if lm is not None else None,
            finger_angles=self.engine.finger_angles,
            finger_states=self.engine.finger_states,
            thresholds=thresholds,
            pinch={"left": pv.get("left"), "right": pv.get("right")},
            pinch_cfg={"left_engage": p.left_engage, "left_release": p.left_release,
                       "right_engage": p.right_engage, "right_release": p.right_release},
            palm={"open": bool(pd.get("open")),
                  "phase": pd.get("phase"),
                  "m": pd.get("m") if isinstance(pd.get("m"), float) else None,
                  "disp_frac": pd.get("disp_frac")
                  if isinstance(pd.get("disp_frac"), float) else None,
                  "together": round(together, 3) if together is not None else None},
            mode=self.ui_mode,
            calibration=calibration,
            capture=capture,
            toast=toast,
            custom_hold=self.engine.custom_hold,
        )
        self.panel.publish("frame", ev)

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
        self._start_panel()
        if self.args.start_active:
            self._activate()
        else:
            print("IDLE — camera off; press the toggle hotkey to start, "
                  "or Start camera in the panel")
        while not self._quit:
            self._poll_hotkeys()
            self._poll_panel()
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
        self._publish_panel(None, snap)
        self._maybe_reload_config()
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
        # Track the frame clock from EVERY read frame (incl. WARMUP):
        # calibrate_begin_step stamps its settle/timeout windows with this,
        # and a 0.0/stale stamp would time the step out instantly.
        self._last_ts_ms = frame.ts_ms
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
        self._ui_mode_tick(frame)

        t2 = time.perf_counter()
        if self.ui_mode != "normal":
            # Calibrating/capturing: engine+pipeline tick (the wizard needs
            # live angles) but guards are skipped and intents are DROPPED —
            # the user is deliberately mousing in the browser right now, and
            # mode entry released everything, so nothing can be held.
            pass
        elif self.session is SessionState.ACTIVE:
            reason = self._guard_trip()   # BEFORE posting, every frame
            if reason is not None:
                self._suspend(reason)
            elif self.synth is not None:
                for intent in out.intents:
                    if intent.phase is Phase.TRIGGER:
                        # System gestures are rare enough per session that
                        # printing every one is never spam, and it's the only
                        # way to tell "the engine never recognized the
                        # gesture" apart from "it fired but macOS/the app in
                        # front didn't react" without opening the preview.
                        print(f"[gesture] {intent.name} (posting now)")
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
        self._publish_panel(None, snap)
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
        self._publish_panel(frame, snap)
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
        # Panel works against replay too — the only camera-free way to
        # exercise the full frontend data path.
        self._start_panel()
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
            self._poll_panel()
            self._ui_mode_tick(frame)
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
            snap = self._snapshot(frame.hand_present)
            if self.preview is not None:
                self._handle_key(
                    self.preview.show(None, frame, snap, self.engine.pinch_values,
                                       palm_debug=self.engine.palm_debug)
                )
            self._publish_panel(frame, snap)
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
    if args.debug_gestures:
        cfg.options.debug_gestures = True

    if args.replay is None:  # preflight skipped for --replay/--list-cameras
        if not permissions.preflight(require=True):
            return 1

    return App(store, args).run()


if __name__ == "__main__":
    raise SystemExit(main())
