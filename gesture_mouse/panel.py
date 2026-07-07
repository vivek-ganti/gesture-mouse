"""Local web control panel: a stdlib-only HTTP + SSE bridge to the app.

Hard constraints (CONTRACTS.md-bound):

- Imports ONLY the Python standard library — never AppKit/cv2/mediapipe/
  numpy and never any gesture_mouse module. The panel receives plain dicts
  and primitives and hands back plain dicts, so it cannot import-cycle into
  the tracker/engine stack and its tests run against real sockets with no
  camera, no mocks, and no macOS frameworks.

- Threading model: HTTP handler threads touch nothing outside PanelServer's
  own state. Every app effect flows OUT through a thread-safe command queue
  drained by the main loop (``poll_commands``); all data flows IN via
  ``publish``. The main thread is the only caller of ``publish`` /
  ``poll_commands`` / ``close``. Handler threads register/unregister client
  queues concurrently with a broadcast, hence ``_clients_lock``.

- Security (mandatory): ``save_gesture`` commands carry shell argv, so a
  CSRF from any web page against localhost would be remote code execution.
  Therefore: bind 127.0.0.1 ONLY; require the per-run bearer token as a
  ``?token=`` query param on BOTH /events and /command (403 otherwise); and
  reject any request whose Host header is not 127.0.0.1[:port] or
  localhost[:port] — the DNS-rebinding guard, applied before ANY routing.
  GET / (the page itself, which contains no secrets) skips only the token
  check, never the Host check. The printed/opened URL is
  ``http://127.0.0.1:PORT/?token=XYZ``; the page's JS reads
  ``location.search`` and appends it to its /events and /command requests.

Stalled-tab safety: ``publish`` serializes each event ONCE and fans the
bytes out via ``put_nowait`` into bounded per-client queues (maxsize 32,
drop-on-full) — a tab that stops reading fills its own queue and silently
loses events, but can never block the hot loop or grow memory. "frame"
events are additionally throttled to one per 66 ms (~15 fps is plenty for a
telemetry readout; other event names always pass). The LAST "config" bytes
are replayed to each newly connected client so a page refresh renders
immediately instead of waiting for the next config change.
"""
from __future__ import annotations

import errno
import json
import queue
import secrets
import threading
import time
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

COMMAND_TYPES: frozenset[str] = frozenset({
    "camera_start", "camera_stop", "camera_switch", "panic", "quit",
    "set_setting",
    "calibrate_start", "calibrate_begin_step", "calibrate_cancel",
    "calibrate_apply",
    "capture_start", "capture_cancel",
    "save_gesture", "delete_gesture",
})

# Broadcast tuning. 66 ms ~= 15 fps: half the camera's native rate, plenty
# for a live readout, and it halves the per-frame JSON serialization cost.
_FRAME_MIN_INTERVAL_S: float = 0.066
_CLIENT_QUEUE_MAX: int = 32
# /command bodies are small JSON; anything bigger is abuse, not a command.
_MAX_COMMAND_BYTES: int = 64 * 1024
_SSE_PING_INTERVAL_S: float = 15.0

# Canonical finger order for the "fingers" readout (mirrors
# signatures.ALL_FINGERS, restated here because panel.py may not import it).
_FINGER_ORDER: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")

_ALLOWED_HOSTNAMES: frozenset[str] = frozenset({"127.0.0.1", "localhost"})

_FALLBACK_HTML: bytes = (
    b"<!doctype html><meta charset='utf-8'><title>gesture-mouse</title>"
    b"<p>panel.html not found; the panel server is running but has no UI "
    b"to serve.</p>"
)


@dataclass(frozen=True)
class PanelCommand:
    """One user action from the panel, drained by the main loop."""

    type: str
    payload: dict


def frame_event(
    *,
    ts: float,
    session: str,
    engine: str | None,
    hand: bool,
    fps: float,
    latency_ms: float,
    suspend_reason: str | None,
    img_w: int,
    img_h: int,
    landmarks: list | None,
    finger_angles: dict,
    finger_states: dict,
    thresholds: dict,
    pinch: dict,
    pinch_cfg: dict,
    palm: dict,
    mode: str,
    calibration: dict | None = None,
    capture: dict | None = None,
    toast: dict | None = None,
) -> dict:
    """Shape one "frame" SSE event dict from primitive inputs.

    A PURE, deliberately dumb dict-shaper: no gesture_mouse types cross
    this boundary and the output is schema-exact per CONTRACTS.md (golden-
    tested). Rounding keeps the ~15 Hz JSON stream small: landmark coords,
    finger angles, fps and latency to 1 decimal. The thumb's "ext" is
    always None — it never gates poses (see signatures.py), so the panel
    renders it display-only.
    """
    fingers = {
        f: {
            "angle": round(float(finger_angles[f]), 1),
            "ext": None if f == "thumb" else finger_states.get(f),
        }
        for f in _FINGER_ORDER
        if f in finger_angles
    }
    return {
        "ts": ts,
        "session": session,
        "engine": engine,
        "hand": hand,
        "fps": round(float(fps), 1),
        "latency_ms": round(float(latency_ms), 1),
        "suspend_reason": suspend_reason,
        "img_w": img_w,
        "img_h": img_h,
        "landmarks": (
            None
            if landmarks is None
            else [[round(float(x), 1), round(float(y), 1)] for x, y in landmarks]
        ),
        "fingers": fingers,
        "thresholds": thresholds,
        "pinch": pinch,
        "pinch_cfg": pinch_cfg,
        "palm": palm,
        "mode": mode,
        "calibration": calibration,
        "capture": capture,
        "toast": toast,
    }


class PanelServer:
    """Owns the HTTP server thread, the SSE client queues, and the command
    queue. Constructed and driven by the main loop; see module docstring
    for the threading contract."""

    def __init__(self, port: int, html_path, token: str | None = None) -> None:
        self._port = int(port)
        self.html_path = Path(html_path)
        self.token = token if token is not None else secrets.token_urlsafe(16)
        self._commands: queue.Queue[PanelCommand] = queue.Queue()
        self._clients: set[queue.Queue] = set()
        self._clients_lock = threading.Lock()
        self._last_config: bytes | None = None
        self._last_frame_monotonic: float = float("-inf")
        self._httpd: _PanelHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def server_address(self) -> tuple[str, int]:
        if self._httpd is None:
            raise RuntimeError("PanelServer.start() has not been called")
        return self._httpd.server_address  # type: ignore[return-value]

    def start(self) -> str:
        """Bind 127.0.0.1 and serve on a daemon thread; returns the URL
        (with token) to print/open. A busy port falls back to an ephemeral
        one rather than failing startup — the URL is the source of truth."""
        try:
            httpd = _PanelHTTPServer(("127.0.0.1", self._port), _Handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            httpd = _PanelHTTPServer(("127.0.0.1", 0), _Handler)
        httpd.panel = self
        self._httpd = httpd
        # Tight poll_interval: serve_forever only re-checks its shutdown flag
        # between polls, so the default 0.5 s would stall every close() by
        # that much (felt at app quit, and 15x per test run).
        self._thread = threading.Thread(
            target=lambda: httpd.serve_forever(poll_interval=0.05),
            name="panel-http",
            daemon=True,
        )
        self._thread.start()
        return f"http://127.0.0.1:{httpd.server_address[1]}/?token={self.token}"

    def publish(self, event: str, data: dict) -> None:
        """Broadcast one SSE event to every connected client.

        Serializes ONCE, then fans the bytes out with put_nowait — a full
        (stalled) client queue drops the event rather than blocking the
        main loop. Called from the main thread only, but takes the clients
        lock so a broadcast never races a handler thread's register/
        unregister (and so the config replay snapshot stays consistent).
        """
        if event == "frame":
            now = time.monotonic()
            if now - self._last_frame_monotonic < _FRAME_MIN_INTERVAL_S:
                return
            self._last_frame_monotonic = now
        payload = json.dumps(data, separators=(",", ":"))
        chunk = f"event: {event}\ndata: {payload}\n\n".encode("utf-8")
        with self._clients_lock:
            if event == "config":
                self._last_config = chunk
            for q in self._clients:
                try:
                    q.put_nowait(chunk)
                except queue.Full:
                    pass  # stalled tab: drop, never block or buffer unboundedly

    def poll_commands(self) -> list[PanelCommand]:
        """Drain queued panel commands without blocking (main loop, 1x/frame)."""
        out: list[PanelCommand] = []
        while True:
            try:
                out.append(self._commands.get_nowait())
            except queue.Empty:
                return out

    def has_clients(self) -> bool:
        with self._clients_lock:
            return bool(self._clients)

    def close(self) -> None:
        if self._httpd is None:
            return
        with self._clients_lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(None)  # sentinel: wake SSE loops so they exit now
            except queue.Full:
                pass  # stalled tab's daemon thread dies with the process
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._httpd = None
        self._thread = None

    # -- handler-thread side (called from _Handler only) --------------------

    def _register(self) -> tuple[queue.Queue, bytes | None]:
        # Snapshot the config replay under the same lock as registration so
        # a concurrent publish is either in the snapshot or in the queue.
        q: queue.Queue = queue.Queue(maxsize=_CLIENT_QUEUE_MAX)
        with self._clients_lock:
            self._clients.add(q)
            return q, self._last_config

    def _unregister(self, q: queue.Queue) -> None:
        with self._clients_lock:
            self._clients.discard(q)


class _PanelHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying a back-reference to its PanelServer."""

    daemon_threads = True  # a stuck handler must never block process exit
    panel: PanelServer

    def handle_error(self, request, client_address) -> None:
        # Vanished tabs produce routine BrokenPipe/reset noise — never worth
        # a stderr traceback in the middle of the hot loop's output.
        pass


class _Handler(BaseHTTPRequestHandler):
    server: _PanelHTTPServer
    server_version = "gesture-mouse-panel"
    sys_version = ""  # don't advertise the Python version

    def log_message(self, format: str, *args) -> None:
        pass  # default logs every request to stderr; the hot loop owns stdout

    # -- guards (before ANY routing; see module docstring) -------------------

    def _host_ok(self) -> bool:
        # DNS-rebinding guard: a hostile page can make the browser resolve
        # attacker.com to 127.0.0.1, but cannot forge the Host header.
        host = self.headers.get("Host", "")
        name, sep, port = host.partition(":")
        if name not in _ALLOWED_HOSTNAMES:
            return False
        return (not sep) or port.isdigit()

    def _token_ok(self, query: str) -> bool:
        supplied = urllib.parse.parse_qs(query).get("token", [""])[0]
        try:
            return secrets.compare_digest(supplied, self.server.panel.token)
        except TypeError:  # non-ASCII garbage in the query string
            return False

    # -- responses -----------------------------------------------------------

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        if code >= 400:
            # Any request body was never read; keeping the connection alive
            # would misparse those bytes as the next request line.
            self.close_connection = True

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if not self._host_ok():
            self._send_json(403, {"error": "forbidden Host"})
            return
        if parsed.path == "/":
            self._serve_html()  # no secrets in the page; token rides the URL
            return
        if not self._token_ok(parsed.query):
            self._send_json(403, {"error": "missing or bad token"})
            return
        if parsed.path == "/events":
            self._serve_events()
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if not self._host_ok():
            self._send_json(403, {"error": "forbidden Host"})
            return
        if not self._token_ok(parsed.query):
            self._send_json(403, {"error": "missing or bad token"})
            return
        if parsed.path != "/command":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_json(400, {"error": "missing Content-Length"})
            return
        if length < 0:
            self._send_json(400, {"error": "bad Content-Length"})
            return
        if length > _MAX_COMMAND_BYTES:
            self._send_json(413, {"error": "body too large"})
            return
        body = self.rfile.read(length)
        try:
            obj = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, {"error": "invalid JSON"})
            return
        if not isinstance(obj, dict) or obj.get("type") not in COMMAND_TYPES:
            self._send_json(400, {"error": "unknown command type"})
            return
        payload = obj.get("payload", {})
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "payload must be an object"})
            return
        self.server.panel._commands.put(
            PanelCommand(type=obj["type"], payload=payload)
        )
        self._send_json(202, {"ok": True})

    # -- routes ---------------------------------------------------------------

    def _serve_html(self) -> None:
        # Re-read per request: the panel page stays live-editable without an
        # app restart. A missing file serves a stub rather than 500 so the
        # server (and its /events + /command API) keeps working regardless.
        try:
            body = self.server.panel.html_path.read_bytes()
        except OSError:
            body = _FALLBACK_HTML
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self) -> None:
        # Register BEFORE the response headers go out: once the client has
        # seen the headers it is guaranteed to receive subsequent publishes.
        panel = self.server.panel
        q, replay = panel._register()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.write(b": connected\n\n")
            if replay is not None:
                self.wfile.write(replay)  # last config: refresh renders now
            while True:
                try:
                    chunk = q.get(timeout=_SSE_PING_INTERVAL_S)
                except queue.Empty:
                    chunk = b": ping\n\n"  # keep intermediaries from timing out
                if chunk is None:
                    return  # close() sentinel
                self.wfile.write(chunk)
        except OSError:  # includes BrokenPipeError/ConnectionResetError
            return  # tab closed or socket died: just unregister below
        finally:
            panel._unregister(q)
            self.close_connection = True
