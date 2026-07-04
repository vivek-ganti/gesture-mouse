"""Camera tracking: AVFoundation device enumeration, MediaPipe HandLandmarker
wrapper, and the JSONL Recorder/ReplayTracker pair.

This module is the ONLY place that touches cv2 / mediapipe / AVFoundation and
the ONLY place normalized landmarks become pixel coordinates. Everything
downstream speaks `types.LandmarkFrame`.

Timestamp invariant: MediaPipe's Tasks API (VIDEO mode) raises on
non-increasing timestamps. All `detect_for_video` timestamps come from ONE
process-lifetime monotonic clock (`_CLOCK_T0`, fixed at import) that is NEVER
reset — so IDLE⇄ACTIVE cycles (close() then open(), which recreates the
landmarker) can never hand the task an older timestamp than a previous cycle.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import IO, Any

import cv2
import numpy as np

from .config import Config
from .types import INDEX_MCP, MIDDLE_MCP, PINKY_MCP, WRIST, LandmarkFrame, Point, dist

# --- process-lifetime monotonic clock (module import = process zero point) ---
_CLOCK_T0 = time.monotonic()
_last_detect_ts_ms = 0  # last int ms handed to detect_for_video, module-wide


def _next_detect_ts_ms() -> int:
    """Strictly-increasing int milliseconds since process start.

    Module-level (not per-tracker) so a recreated landmarker — or two frames
    landing inside the same millisecond — can never see a non-increasing ts.
    """
    global _last_detect_ts_ms
    now = int((time.monotonic() - _CLOCK_T0) * 1000.0)
    if now <= _last_detect_ts_ms:
        now = _last_detect_ts_ms + 1
    _last_detect_ts_ms = now
    return now


# ------------------------------ camera enumeration ---------------------------

def _avf_devices() -> list[Any]:
    """All AVFoundation video devices, deduped by uniqueID.

    Primary path: the (deprecated) devicesWithMediaType_ enumeration — ON
    PURPOSE, because OpenCV's AVFoundation backend builds its
    VideoCapture-index table from exactly that API, and this list's job is to
    map names to cv2 indexes. Enumerating via AVCaptureDeviceDiscoverySession
    (which also surfaces Continuity/DeskView devices the deprecated API may
    order or filter differently) can silently point a name at the wrong
    device. DiscoverySession is only the fallback when the deprecated API
    returns nothing.
    """
    try:
        import AVFoundation as AVF
    except ImportError:  # non-macOS or pyobjc missing: no cameras enumerable
        return []

    devices: list[Any] = []
    try:
        devices = list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo))
    except Exception:
        devices = []
    if not devices:
        type_names = (
            "AVCaptureDeviceTypeBuiltInWideAngleCamera",
            "AVCaptureDeviceTypeExternalUnknown",   # pre-macOS 14 external/USB/virtual
            "AVCaptureDeviceTypeExternal",          # macOS 14+ replacement
            "AVCaptureDeviceTypeContinuityCamera",  # iPhone
            "AVCaptureDeviceTypeDeskViewCamera",
        )
        device_types = [
            t for n in type_names if (t := getattr(AVF, n, None)) is not None
        ]
        try:
            session = AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
                device_types, AVF.AVMediaTypeVideo,
                AVF.AVCaptureDevicePositionUnspecified,
            )
            if session is not None:
                devices = list(session.devices())
        except Exception:
            devices = []

    seen: set[str] = set()
    unique: list[Any] = []
    for d in devices:
        uid = str(d.uniqueID())
        if uid not in seen:
            seen.add(uid)
            unique.append(d)
    return unique


def list_cameras() -> list[str]:
    """localizedName of every video capture device, in stable device order.

    OpenCV's AVFoundation backend builds its index→device table from the same
    AVFoundation device enumeration, so position i here is the index to pass
    to cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION). This is an empirically
    stable convention, not an API guarantee — CameraTracker.open() therefore
    falls back to index 0 with a warning when a configured name is missing,
    and the preview window makes a wrong-camera mapping immediately obvious.
    """
    return [str(d.localizedName()) for d in _avf_devices()]


def camera_index(name: str) -> int | None:
    """Index of the named camera for cv2.VideoCapture, or None if not found."""
    for i, cam in enumerate(list_cameras()):
        if cam == name:
            return i
    return None


def bench_cameras(n_frames: int = 10, width: int = 640, height: int = 480,
                  fps: int = 30) -> None:
    """Open every enumerated camera, time reads, print a table. Diagnoses
    wrong-device mappings and dead virtual cameras (~1000ms black frames)."""
    cams = list_cameras()
    if not cams:
        print("no cameras found")
        return
    for idx, name in enumerate(cams):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print(f"[{idx}] {name!r}: FAILED TO OPEN")
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.read()  # absorb open/exposure ramp
        times: list[float] = []
        brightness: list[float] = []
        for _ in range(n_frames):
            t0 = time.monotonic()
            ok, bgr = cap.read()
            times.append(time.monotonic() - t0)
            if ok and bgr is not None:
                brightness.append(float(bgr.mean()))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rfps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        avg_ms = sum(times) / len(times) * 1000.0 if times else float("inf")
        bright = sum(brightness) / len(brightness) if brightness else -1.0
        verdict = "OK"
        if avg_ms > 500.0:
            verdict = "SLOW (dead/virtual device?)"
        elif 0.0 <= bright < 2.0:
            verdict = "BLACK FRAMES (lens covered or virtual device?)"
        print(
            f"[{idx}] {name!r}: {w}x{h} @{rfps:.0f}fps  "
            f"read {avg_ms:.0f}ms ({1000.0/avg_ms:.0f}fps achieved)  "
            f"brightness {bright:.0f}  -> {verdict}"
        )


# --------------------------------- CameraTracker -----------------------------

_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class CameraTracker:
    """Owns the cv2 capture and the HandLandmarker; emits LandmarkFrames.

    Construction is side-effect free (no camera, no model load) so tools and
    tests can build one headless; open() acquires everything, close() releases
    everything (IDLE = zero camera use), and open() after close() works — the
    landmarker is recreated but the timestamp clock is module-level and
    survives, keeping detect_for_video timestamps strictly increasing.
    """

    def __init__(self, cfg: Config, model_path: str = "hand_landmarker.task") -> None:
        self._cfg = cfg
        self._model_path = self._resolve_model_path(model_path)
        self._cap: cv2.VideoCapture | None = None
        self._landmarker: Any = None
        self._source: str = cfg.camera.name or "camera"
        # Rotated+mirrored BGR image of the last successful grab (additive to
        # the CONTRACTS.md surface): the preview's non-privacy mode needs the
        # camera image, which read()'s LandmarkFrame deliberately omits.
        self.last_bgr: np.ndarray | None = None

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        p = Path(model_path)
        if p.exists():
            return str(p)
        repo_root = Path(__file__).resolve().parents[1]
        candidate = repo_root / model_path
        return str(candidate) if candidate.exists() else str(p)

    def _candidate_order(self) -> tuple[list[int], list[str]]:
        """Camera indexes to try, configured name's index first. The
        AVFoundation enumeration order is NOT stable across runs and may not
        match OpenCV's internal index table, so the name is a *preference*,
        never trusted — every candidate is probed before being accepted."""
        cams = list_cameras()
        order = list(range(max(len(cams), 1)))
        name = self._cfg.camera.name
        preferred: int | None = None
        if name:
            preferred = camera_index(name)
            if preferred is None:
                warnings.warn(
                    f"camera {name!r} not found (available: {cams!r}); probing all"
                )
        if preferred is None:  # else prefer the built-in over virtual devices
            for i, cam in enumerate(cams):
                if "facetime" in cam.lower() or "built-in" in cam.lower():
                    preferred = i
                    break
        if preferred is not None and preferred in order:
            order.remove(preferred)
            order.insert(0, preferred)
        return order, cams

    @staticmethod
    def _probe_cap(cap: cv2.VideoCapture) -> tuple[float, float]:
        """(avg read ms, avg brightness) after absorbing the exposure ramp.
        Dead virtual cameras (phone-cam apps with no phone) show up as ~1000ms
        reads and/or pure-black (0.0) frames."""
        ramp_until = time.monotonic() + 0.8
        while time.monotonic() < ramp_until:
            cap.read()
        times: list[float] = []
        brights: list[float] = []
        for _ in range(5):
            t0 = time.monotonic()
            ok, bgr = cap.read()
            times.append(time.monotonic() - t0)
            if ok and bgr is not None:
                brights.append(float(bgr.mean()))
        if not brights:
            return float("inf"), -1.0
        return (sum(times) / len(times) * 1000.0,
                sum(brights) / len(brights))

    def open(self) -> None:
        # Landmarker first: a model-load failure must not leave the camera
        # acquired while the app drops back to IDLE (IDLE = zero camera use).
        self._landmarker = self._make_landmarker()
        order, cams = self._candidate_order()
        cfg = self._cfg.camera
        fallback: tuple[cv2.VideoCapture, int, str] | None = None
        try:
            for idx in order:
                label = cams[idx] if idx < len(cams) else f"camera[{idx}]"
                cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
                if not cap.isOpened():
                    cap.release()
                    print(f"camera[{idx}] {label!r}: failed to open, trying next")
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
                cap.set(cv2.CAP_PROP_FPS, cfg.fps)
                read_ms, bright = self._probe_cap(cap)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(
                    f"camera[{idx}] {label!r}: {w}x{h} probe read {read_ms:.0f}ms "
                    f"brightness {bright:.1f}"
                )
                if read_ms < 500.0 and bright > 0.5:
                    self._cap = cap
                    self._source = label
                    return
                # Dead-looking feed: keep the first one as a fallback in case
                # every camera probes dark (pitch-black room), else move on.
                if fallback is None:
                    fallback = (cap, idx, label)
                    print("  looks dead (slow or black) — trying next camera")
                else:
                    cap.release()
            if fallback is not None:
                cap, idx, label = fallback
                fallback = None
                self._cap = cap
                self._source = label
                print(
                    f"  WARNING: no camera produced a live image; using "
                    f"camera[{idx}] {label!r} anyway. If the preview stays "
                    f"black, check the room light / pick another with --camera."
                )
                return
            self._landmarker.close()
            self._landmarker = None
            raise RuntimeError(f"no openable camera (enumerated: {cams!r})")
        finally:
            if fallback is not None and self._cap is not fallback[0]:
                fallback[0].release()


    def _make_landmarker(self) -> Any:
        # Imported lazily so merely importing tracker.py stays cheap and
        # headless-safe (mediapipe init logs / loads native libs).
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=self._model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        return vision.HandLandmarker.create_from_options(options)

    def read(self) -> LandmarkFrame | None:
        """One frame through the full convention pipeline; None = grab failed."""
        if self._cap is None or self._landmarker is None:
            raise RuntimeError("CameraTracker.read() before open()")
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            return None
        rotate = self._cfg.camera.rotate % 360
        if rotate in _ROTATE_CODES:  # rotate BEFORE mirror (config contract)
            bgr = cv2.rotate(bgr, _ROTATE_CODES[rotate])
        if self._cfg.camera.mirror:
            bgr = cv2.flip(bgr, 1)  # selfie view; mirrored BEFORE detection
        self.last_bgr = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._detect(rgb)

    def _detect(self, rgb: np.ndarray) -> LandmarkFrame:
        """Detect on an already rotated+mirrored RGB frame -> LandmarkFrame."""
        import mediapipe as mp

        img_h, img_w = rgb.shape[:2]
        ts = _next_detect_ts_ms()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_image, ts)

        absent = LandmarkFrame(
            ts_ms=float(ts), handedness=None, landmarks=None,
            img_w=img_w, img_h=img_h, confidence=0.0, scale=0.0, source=self._source,
        )
        if not result.hand_landmarks:
            return absent

        # Report ANY detected hand, truthfully labeled — the tracker does NOT
        # filter by handedness. The engine pins tracking to cfg.hand; doing it
        # here too made a wrong label (mirror convention mismatch, left-handed
        # use) indistinguishable from "no hand" in recordings and stats.
        for lms, hd_categories in zip(result.hand_landmarks, result.handedness):
            if not hd_categories:
                continue
            label = hd_categories[0].category_name  # "Left" / "Right"
            # MediaPipe reports handedness assuming a mirrored (selfie) input
            # image (per the Hand Landmarker docs), and we mirror before
            # detection — so on our mirrored frame "Right" IS the user's
            # physical right hand. With mirror off (raw frame) the docs say to
            # swap the label.
            if not self._cfg.camera.mirror:
                label = {"Left": "Right", "Right": "Left"}.get(label, label)
            # Normalized -> PIXEL coords: the ONE place this happens.
            points = tuple(Point(lm.x * img_w, lm.y * img_h) for lm in lms)
            scale = max(
                dist(points[INDEX_MCP], points[PINKY_MCP]),
                0.7 * dist(points[WRIST], points[MIDDLE_MCP]),
            )
            return LandmarkFrame(
                ts_ms=float(ts), handedness=label, landmarks=points,
                img_w=img_w, img_h=img_h,
                confidence=float(hd_categories[0].score), scale=scale,
                source=self._source,
            )
        return absent

    def close(self) -> None:
        """Release camera and landmarker fully. The ts clock is NOT reset —
        a later open() recreates the landmarker against the same clock."""
        self.last_bgr = None  # never keep a camera image alive in IDLE
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass  # never let teardown mask the reason we're closing
            finally:
                self._landmarker = None


# ------------------------------ Recorder / Replay ----------------------------
#
# JSONL format (frozen by CONTRACTS.md): line 0 is a header dict (mirror,
# rotate, img dims, config snapshot, source); each following line is
# {"ts_ms":..., "handedness":..., "lm":[[x,y]*21]|null, "w":..., "h":...,
#  "conf":..., "scale":...} — landmarks already in mirrored-frame pixel coords.


class Recorder:
    """Tee for LandmarkFrames -> JSONL; header is written immediately."""

    def __init__(self, path: str, header: dict) -> None:
        self._fh: IO[str] | None = open(path, "w", encoding="utf-8")
        self._fh.write(json.dumps(header) + "\n")
        self._fh.flush()

    def write(self, frame: LandmarkFrame) -> None:
        if self._fh is None:
            raise RuntimeError("Recorder.write() after close()")
        row = {
            "ts_ms": frame.ts_ms,
            "handedness": frame.handedness,
            "lm": [[p.x, p.y] for p in frame.landmarks] if frame.landmarks is not None else None,
            "w": frame.img_w,
            "h": frame.img_h,
            "conf": frame.confidence,
            "scale": frame.scale,
        }
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()  # crash mid-recording still yields a usable fixture

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class ReplayTracker:
    """Replays a recorded JSONL through the CameraTracker duck-type:
    open() is a no-op, read() yields frames in file order then None forever."""

    def __init__(self, path: str) -> None:
        self._path = str(path)
        self._fh: IO[str] | None = open(path, "r", encoding="utf-8")
        header_line = self._fh.readline()
        if not header_line.strip():
            self._fh.close()
            self._fh = None
            raise ValueError(f"{path}: missing JSONL header line")
        self.header: dict = json.loads(header_line)
        self._source = f"replay:{self._path}"

    def open(self) -> None:  # duck-type parity with CameraTracker
        pass

    def read(self) -> LandmarkFrame | None:
        if self._fh is None:
            return None
        while True:
            line = self._fh.readline()
            if not line:  # EOF: keep returning None
                return None
            if line.strip():
                break
        d = json.loads(line)
        lm = d["lm"]
        landmarks = tuple(Point(float(x), float(y)) for x, y in lm) if lm is not None else None
        return LandmarkFrame(
            ts_ms=float(d["ts_ms"]),
            handedness=d["handedness"],
            landmarks=landmarks,
            img_w=int(d["w"]),
            img_h=int(d["h"]),
            confidence=float(d["conf"]),
            scale=float(d["scale"]),
            source=self._source,
        )

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
