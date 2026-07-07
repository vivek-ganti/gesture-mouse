"""panel.py tests: real sockets against a live PanelServer, no HTTP mocks.

Each test gets a fresh server on an ephemeral port (fresh throttle state,
fresh client set). SSE streams are read straight off http.client responses
byte-by-byte and parsed into (event, data) pairs; comment blocks
(": connected", ": ping") are asserted where meaningful and skipped
elsewhere. Timing-sensitive assertions (throttle window, disconnect
detection) use markers and bounded poll loops instead of bare sleeps so the
whole file stays fast and deterministic.
"""
from __future__ import annotations

import http.client
import json
import re
import socket
import threading
import time

import pytest

from gesture_mouse.panel import (
    COMMAND_TYPES,
    PanelCommand,
    PanelServer,
    frame_event,
)

HTML_BODY = b"<html><body>PANEL_OK</body></html>"
TOKEN = "testtok"


@pytest.fixture
def server(tmp_path):
    html = tmp_path / "panel.html"
    html.write_bytes(HTML_BODY)
    srv = PanelServer(port=0, html_path=html, token=TOKEN)
    srv.start()
    yield srv
    srv.close()


def connect(server, timeout=3.0):
    host, port = server.server_address
    return http.client.HTTPConnection(host, port, timeout=timeout)


def sse_connect(server, token=TOKEN):
    conn = connect(server)
    conn.request("GET", f"/events?token={token}")
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "text/event-stream"
    return conn, resp


def sse_close(conn, resp):
    """Actually close an SSE client socket.

    For a response with no Content-Length, http.client hands socket
    ownership to the response object, so ``conn.close()`` alone never sends
    FIN while ``resp`` is alive — the response must be closed first or the
    server can't observe the disconnect.
    """
    resp.close()
    conn.close()


def read_block(resp):
    """One raw SSE block (comment or event), terminated by a blank line."""
    buf = b""
    while not buf.endswith(b"\n\n"):
        ch = resp.read(1)
        if not ch:
            raise AssertionError(f"SSE stream closed mid-block: {buf!r}")
        buf += ch
    return buf.decode("utf-8")


def read_event(resp):
    """Next (event, data) pair, skipping comment blocks."""
    while True:
        block = read_block(resp)
        if block.startswith(":"):
            continue
        event = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        return event, data


def post_command(server, obj=None, token=TOKEN, raw=None):
    conn = connect(server)
    body = raw if raw is not None else json.dumps(obj).encode()
    path = "/command" + (f"?token={token}" if token is not None else "")
    conn.request("POST", path, body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    payload = resp.read()
    conn.close()
    return resp.status, (json.loads(payload) if payload else None)


# ---------------------------------------------------------------- contract


def test_command_types_exact():
    assert COMMAND_TYPES == frozenset({
        "camera_start", "camera_stop", "camera_switch", "panic", "quit",
        "set_setting",
        "calibrate_start", "calibrate_begin_step", "calibrate_cancel",
        "calibrate_apply",
        "capture_start", "capture_cancel",
        "save_gesture", "delete_gesture",
    })


def test_frame_event_golden():
    out = frame_event(
        ts=1234.5,
        session="active",
        engine="POINTER",
        hand=True,
        fps=29.97,
        latency_ms=12.34,
        suspend_reason=None,
        img_w=640,
        img_h=480,
        landmarks=[[12.34, 56.78]] + [[100.0, 200.5]] * 20,
        finger_angles={"thumb": 150.06, "index": 179.94, "middle": 42.01,
                       "ring": 41.62, "pinky": 3.49},
        # thumb True on purpose: the shaper must force its "ext" to None.
        finger_states={"thumb": True, "index": True, "middle": False,
                       "ring": False, "pinky": False},
        thresholds={
            "index": {"extend": 155.0, "curl": 120.0, "calibrated": True},
            "middle": {"extend": 160.0, "curl": 130.0, "calibrated": False},
            "ring": {"extend": 160.0, "curl": 130.0, "calibrated": False},
            "pinky": {"extend": 160.0, "curl": 130.0, "calibrated": False},
        },
        pinch={"left": 0.42, "right": None},
        pinch_cfg={"left_engage": 0.38, "left_release": 0.52,
                   "right_engage": 0.36, "right_release": 0.5},
        palm={"open": False, "phase": None, "m": 1.23, "disp_frac": 0.0},
        mode="normal",
        calibration=None,
        capture=None,
        toast={"id": 3, "level": "info", "text": "saved"},
    )
    assert out == {
        "ts": 1234.5,
        "session": "active",
        "engine": "POINTER",
        "hand": True,
        "fps": 30.0,
        "latency_ms": 12.3,
        "suspend_reason": None,
        "img_w": 640,
        "img_h": 480,
        "landmarks": [[12.3, 56.8]] + [[100.0, 200.5]] * 20,
        "fingers": {
            "thumb": {"angle": 150.1, "ext": None},
            "index": {"angle": 179.9, "ext": True},
            "middle": {"angle": 42.0, "ext": False},
            "ring": {"angle": 41.6, "ext": False},
            "pinky": {"angle": 3.5, "ext": False},
        },
        "thresholds": {
            "index": {"extend": 155.0, "curl": 120.0, "calibrated": True},
            "middle": {"extend": 160.0, "curl": 130.0, "calibrated": False},
            "ring": {"extend": 160.0, "curl": 130.0, "calibrated": False},
            "pinky": {"extend": 160.0, "curl": 130.0, "calibrated": False},
        },
        "pinch": {"left": 0.42, "right": None},
        "pinch_cfg": {"left_engage": 0.38, "left_release": 0.52,
                      "right_engage": 0.36, "right_release": 0.5},
        "palm": {"open": False, "phase": None, "m": 1.23, "disp_frac": 0.0},
        "mode": "normal",
        "calibration": None,
        "capture": None,
        "toast": {"id": 3, "level": "info", "text": "saved"},
    }
    # The whole point of the shaper: what it emits must survive the SSE wire.
    assert json.loads(json.dumps(out)) == out


def test_frame_event_none_fields():
    out = frame_event(
        ts=1.0, session="idle", engine=None, hand=False, fps=0.0,
        latency_ms=0.0, suspend_reason=None, img_w=0, img_h=0,
        landmarks=None, finger_angles={}, finger_states={}, thresholds={},
        pinch={"left": None, "right": None}, pinch_cfg={},
        palm={"open": False, "phase": None, "m": None, "disp_frac": None},
        mode="normal",
    )
    assert out["landmarks"] is None
    assert out["fingers"] == {}
    assert out["calibration"] is None
    assert out["capture"] is None
    assert out["toast"] is None


# ------------------------------------------------------------------- GET /


def test_get_root_serves_html_and_rereads(server):
    conn = connect(server)
    conn.request("GET", "/")  # no token: the page itself needs none
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "text/html; charset=utf-8"
    assert resp.read() == HTML_BODY
    conn.close()
    # Live-editable: the file is re-read on every request.
    server.html_path.write_bytes(b"<p>v2</p>")
    conn = connect(server)
    conn.request("GET", "/")
    resp = conn.getresponse()
    assert resp.read() == b"<p>v2</p>"
    conn.close()


def test_get_root_fallback_when_html_missing(tmp_path):
    srv = PanelServer(port=0, html_path=tmp_path / "nope.html", token=TOKEN)
    srv.start()
    try:
        conn = connect(srv)
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        assert b"panel.html" in resp.read()  # stub page, not a 500
        conn.close()
    finally:
        srv.close()


# ---------------------------------------------------------------- security


def test_dns_rebinding_guard_on_all_routes(server):
    cases = [
        ("GET", "/", None),
        ("GET", f"/events?token={TOKEN}", None),
        ("POST", f"/command?token={TOKEN}", b'{"type": "panic"}'),
    ]
    for method, path, body in cases:
        conn = connect(server)
        conn.request(method, path, body=body, headers={"Host": "evil.com"})
        resp = conn.getresponse()
        assert resp.status == 403, (method, path)
        resp.read()
        conn.close()
    assert server.poll_commands() == []  # the evil POST was never enqueued
    # Legit Host spellings pass.
    for host in ("localhost", "127.0.0.1", f"127.0.0.1:{server.server_address[1]}"):
        conn = connect(server)
        conn.request("GET", "/", headers={"Host": host})
        resp = conn.getresponse()
        assert resp.status == 200, host
        resp.read()
        conn.close()


def test_events_requires_token(server):
    for path in ("/events", "/events?token=wrong"):
        conn = connect(server)
        conn.request("GET", path)
        resp = conn.getresponse()
        assert resp.status == 403, path
        resp.read()
        conn.close()


def test_command_requires_token(server):
    assert post_command(server, {"type": "panic"}, token=None)[0] == 403
    assert post_command(server, {"type": "panic"}, token="wrong")[0] == 403
    assert server.poll_commands() == []


def test_unknown_paths_404_after_auth(server):
    conn = connect(server)
    conn.request("GET", "/nope")  # token check precedes routing -> 403 not 404
    resp = conn.getresponse()
    assert resp.status == 403
    resp.read()
    conn.close()
    conn = connect(server)
    conn.request("GET", f"/nope?token={TOKEN}")
    resp = conn.getresponse()
    assert resp.status == 404
    resp.read()
    conn.close()
    conn = connect(server)
    conn.request("POST", f"/nope?token={TOKEN}", body=b"{}")
    resp = conn.getresponse()
    assert resp.status == 404
    resp.read()
    conn.close()


# --------------------------------------------------------------------- SSE


def test_events_stream_in_order(server):
    conn, resp = sse_connect(server)
    assert read_block(resp).startswith(": connected")
    server.publish("config", {"settings": {"scroll.gain": 350.0}})
    server.publish("frame", {"ts": 1.0})
    assert read_event(resp) == ("config", {"settings": {"scroll.gain": 350.0}})
    assert read_event(resp) == ("frame", {"ts": 1.0})
    sse_close(conn, resp)


def test_late_joiner_gets_last_config(server):
    server.publish("config", {"v": 1})
    server.publish("config", {"v": 2})  # only the LAST config is replayed
    conn, resp = sse_connect(server)
    assert read_block(resp).startswith(": connected")
    assert read_event(resp) == ("config", {"v": 2})
    sse_close(conn, resp)


def test_frame_throttle_passes_one_of_burst(server):
    conn, resp = sse_connect(server)
    read_block(resp)  # ": connected"
    t0 = time.monotonic()
    for i in range(10):
        server.publish("frame", {"i": i})
    burst_s = time.monotonic() - t0
    server.publish("config", {"marker": "flush"})  # config is never throttled
    frames = []
    while True:
        event, data = read_event(resp)
        if event == "config":
            assert data == {"marker": "flush"}
            break
        assert event == "frame"
        frames.append(data)
    assert burst_s < 0.066  # sanity: the burst really fit one throttle window
    assert frames == [{"i": 0}]
    sse_close(conn, resp)


def test_config_never_throttled(server):
    conn, resp = sse_connect(server)
    read_block(resp)
    for i in range(10):
        server.publish("config", {"i": i})
    got = [read_event(resp) for _ in range(10)]
    assert got == [("config", {"i": i}) for i in range(10)]
    sse_close(conn, resp)


def test_stalled_client_never_blocks_publish(server):
    host, port = server.server_address
    # Raw socket that sends the request then never reads a single byte back.
    stalled = socket.create_connection((host, port), timeout=5)
    stalled.sendall(
        f"GET /events?token={TOKEN} HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n\r\n".encode()
    )
    deadline = time.monotonic() + 2.0
    while not server.has_clients() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.has_clients()

    # Pump enough bytes to fill the socket buffer AND the 32-slot queue: the
    # stalled client must silently drop; publish() must never block.
    filler = {"pad": "x" * 2048}
    worst = 0.0
    t0 = time.monotonic()
    for i in range(100):
        t = time.monotonic()
        server.publish("frame", {"i": i})   # some pass the 66ms throttle
        server.publish("config", filler)    # all pass; ~200 KiB total
        worst = max(worst, time.monotonic() - t)
        time.sleep(0.002)
    assert worst < 0.25
    assert time.monotonic() - t0 < 3.0

    # Server still fully functional for a fresh client.
    conn, resp = sse_connect(server)
    assert read_block(resp).startswith(": connected")
    server.publish("config", {"marker": "alive"})
    while True:
        event, data = read_event(resp)
        if event == "config" and data == {"marker": "alive"}:
            break
    sse_close(conn, resp)
    stalled.close()


def test_has_clients_lifecycle(server):
    assert not server.has_clients()
    conn, resp = sse_connect(server)
    read_block(resp)  # registration precedes the response headers
    assert server.has_clients()
    sse_close(conn, resp)
    # The server notices the disconnect on its next write attempt.
    deadline = time.monotonic() + 2.0
    while server.has_clients() and time.monotonic() < deadline:
        server.publish("config", {"tick": time.monotonic()})
        time.sleep(0.05)
    assert not server.has_clients()


# ----------------------------------------------------------- POST /command


def test_command_valid_202_and_enqueued(server):
    status, body = post_command(
        server, {"type": "camera_switch", "payload": {"index": 2}})
    assert (status, body) == (202, {"ok": True})
    status, body = post_command(server, {"type": "panic"})  # payload defaults {}
    assert (status, body) == (202, {"ok": True})
    assert server.poll_commands() == [
        PanelCommand(type="camera_switch", payload={"index": 2}),
        PanelCommand(type="panic", payload={}),
    ]
    assert server.poll_commands() == []  # drained


def test_command_rejects_bad_requests(server):
    assert post_command(server, {"type": "rm_rf", "payload": {}})[0] == 400
    assert post_command(server, raw=b'{"type": "panic",')[0] == 400
    assert post_command(server, raw=b'["type", "panic"]')[0] == 400
    assert post_command(server, {"type": "panic", "payload": ["nope"]})[0] == 400
    assert server.poll_commands() == []


def test_command_body_over_64k_rejected(server):
    conn = connect(server)
    # Send only headers claiming an oversized body: the server must reject
    # from Content-Length alone, without reading (or waiting for) the body.
    conn.request("POST", f"/command?token={TOKEN}", body=None,
                 headers={"Content-Type": "application/json",
                          "Content-Length": str(64 * 1024 + 1)})
    resp = conn.getresponse()
    assert resp.status == 413
    resp.read()
    conn.close()
    assert server.poll_commands() == []


def test_command_body_at_cap_accepted(server):
    cmd = {"type": "save_gesture", "payload": {"pad": ""}}
    overhead = len(json.dumps(cmd).encode())
    cmd["payload"]["pad"] = "x" * (64 * 1024 - overhead)
    raw = json.dumps(cmd).encode()
    assert len(raw) == 64 * 1024
    status, body = post_command(server, raw=raw)
    assert (status, body) == (202, {"ok": True})
    cmds = server.poll_commands()
    assert len(cmds) == 1 and cmds[0].type == "save_gesture"


# --------------------------------------------------------------- lifecycle


def test_token_autogenerated(tmp_path):
    html = tmp_path / "auto.html"
    html.write_bytes(HTML_BODY)
    srv = PanelServer(port=0, html_path=html)  # token=None -> auto-generated
    try:
        url = srv.start()
        m = re.fullmatch(r"http://127\.0\.0\.1:(\d+)/\?token=([A-Za-z0-9_-]+)", url)
        assert m, url
        assert int(m.group(1)) == srv.server_address[1]
        assert "panel-http" in {t.name for t in threading.enumerate()}
        conn = connect(srv)
        conn.request("GET", f"/events?token={m.group(2)}")
        resp = conn.getresponse()
        assert resp.status == 200
        conn.close()
    finally:
        srv.close()


def test_port_in_use_falls_back_to_ephemeral(tmp_path, server):
    busy_port = server.server_address[1]
    html = tmp_path / "second.html"
    html.write_bytes(HTML_BODY)
    srv2 = PanelServer(port=busy_port, html_path=html, token="tok2")
    try:
        url = srv2.start()
        real_port = srv2.server_address[1]
        assert real_port != busy_port
        assert url == f"http://127.0.0.1:{real_port}/?token=tok2"
    finally:
        srv2.close()
