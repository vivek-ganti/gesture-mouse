"""Per-user calibration of finger extend/curl angle thresholds.

Pure logic (CONTRACTS.md global rules): no macOS / cv2 / mediapipe / numpy
imports, no wall-clock reads — time arrives only via ``ts_ms`` parameters,
so a whole calibration run is replayable and unit-testable.

Why calibrate: the global ``pose.extend_angle_deg`` / ``curl_angle_deg``
pair is a one-size guess. Real hands differ per finger — a pinky that never
straightens past 150 deg, a ring finger that cannot curl below 100 deg while
its neighbors extend. The panel walks the user through six posed steps
(``STEPS``), samples ``signatures.compute_finger_angles`` output per frame,
and derives a per-finger threshold pair.

Why percentiles, not min/max: every take contains transition frames (the
hand settling into the pose) plus MediaPipe flicker. P10 of the extended
cluster / P90 of the curled cluster bound the *reliable cores* of each
cluster; the ``SETTLE_MS`` discard removes the worst of the transition and
the percentiles absorb the rest.

Why a hysteresis band, not one cutoff: thresholds feed
``signatures.FingerLatch``. The band (30% of the gap, floored at
``min_hyst``, never wider than the gap itself) is centered mid-gap so a
finger hovering near its rest angle cannot flicker across a single line.

Why the relaxed nudge: a genuinely relaxed half-open hand often idles ABOVE
the mid-gap extend threshold (~130-150 deg). If the relaxed step's P75
clears the derived extend threshold, extend is raised to just above the
relaxed cluster (capped safely below the extended cluster's floor) so an
idle hand does not latch fingers extended and trigger poses. If that raise
cannot fit, the derived pair is kept but flagged ``relaxed_overlap`` so the
UI can warn about false engages.

Why the validation replay: derived thresholds are only trusted once they
reproduce the user's own takes. Each non-relaxed step's recorded angles are
replayed through fresh latches and must re-match that step's expected
signature; the relaxed take must match NO known signature (built-in or
custom) from EITHER latch initialization — ``FingerLatch`` is sticky, so a
mid-band relaxed hand keeps whatever state it started in, and both start
states must be safe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .signatures import ALL_FINGERS, BUILTINS, FINGERS, FingerLatch, Signature

# Thumb never gates matching, so no step *poses* it explicitly — but it is
# sampled on every frame, and exactly two steps have a known thumb state
# worth clustering: open_palm (thumb out with the palm) and fist (thumb
# wrapped). Its derived thresholds are advisory (panel display/persistence).
THUMB_SOURCES: dict[str, str] = {"open_palm": "ext", "fist": "curl"}

_RELAXED_STEP_ID = "relaxed"


@dataclass(frozen=True)
class CalibStep:
    """One guided pose the user holds while angles are sampled.

    ``expected`` maps each gating finger to "ext" | "curl" | "ignore":
    it is both the cluster router for ``compute()`` and the signature the
    validation replay must reproduce. The relaxed step is all-"ignore" —
    it feeds no ext/curl cluster and exists only for the relaxed nudge and
    the must-match-nothing validation."""

    id: str
    label: str
    instruction: str
    expected: dict


STEPS: tuple[CalibStep, ...] = (
    CalibStep(
        "pointer", "Point",
        "Hold up just your index finger, like pointing at the screen.",
        {"index": "ext", "middle": "curl", "ring": "curl", "pinky": "curl"},
    ),
    CalibStep(
        "open_palm", "Open palm",
        "Open hand — all four fingers extended, relaxed spread.",
        {"index": "ext", "middle": "ext", "ring": "ext", "pinky": "ext"},
    ),
    CalibStep(
        "scroll", "Two up",
        "Two fingers up, together — like a peace sign with the fingers touching.",
        {"index": "ext", "middle": "ext", "ring": "curl", "pinky": "curl"},
    ),
    CalibStep(
        "horns", "Rock sign",
        "Rock sign — index and pinky up, middle and ring curled.",
        {"index": "ext", "middle": "curl", "ring": "curl", "pinky": "ext"},
    ),
    CalibStep(
        "fist", "Fist",
        "Make a relaxed fist.",
        {"index": "curl", "middle": "curl", "ring": "curl", "pinky": "curl"},
    ),
    CalibStep(
        _RELAXED_STEP_ID, "Rest",
        "Let your hand rest naturally, half-open, doing nothing.",
        {"index": "ignore", "middle": "ignore", "ring": "ignore", "pinky": "ignore"},
    ),
)

_STEP_BY_ID: dict[str, CalibStep] = {s.id: s for s in STEPS}
_ALL_STEP_IDS: frozenset[str] = frozenset(_STEP_BY_ID)


def percentile(vals, q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]. No numpy: sorts a
    copy and interpolates between the two bracketing order statistics
    (numpy's default "linear" method). n == 1 and all-equal inputs simply
    return that value."""
    if not vals:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    rank = (q / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(s[lo])
    frac = rank - lo
    return float(s[lo]) * (1.0 - frac) + float(s[hi]) * frac


@dataclass
class FingerResult:
    """Derived threshold pair (or the reason there is none) for one finger.

    ``extend``/``curl`` are None when calibration keeps the defaults
    (status "insufficient" or "overlap"); ``ext_low``/``curl_high``/``gap``
    are the cluster statistics the decision was made from (None when the
    clusters were too small to compute)."""

    finger: str
    extend: float | None
    curl: float | None
    ext_low: float | None
    curl_high: float | None
    gap: float | None
    status: str  # "ok" | "insufficient" | "overlap" | "relaxed_overlap"
    note: str


def derive_thresholds(
    ext: list[float],
    curl: list[float],
    relaxed: list[float],
    defaults: tuple[float, float],
    min_gap: float = 15.0,
    min_hyst: float = 8.0,
    finger: str = "",
) -> FingerResult:
    """Turn one finger's sampled angle clusters into a latch threshold pair.

    Normative math (tests pin it): ext_low = P10(ext), curl_high =
    P90(curl); the pair straddles the gap midpoint with a hysteresis band
    of ``min(max(min_hyst, 0.3 * gap), gap)``; the relaxed cluster may then
    nudge extend upward (see module docstring). Fewer than 40 samples in
    either cluster, or a gap under ``min_gap``, keeps the defaults."""
    label = finger or "finger"
    if len(ext) < 40 or len(curl) < 40:
        note = (
            f"not enough clean samples for the {label} "
            f"({len(ext)} extended / {len(curl)} curled, need 40 of each) — "
            f"keeping the default thresholds {defaults[0]:.1f}/{defaults[1]:.1f}"
        )
        return FingerResult(finger, None, None, None, None, None, "insufficient", note)

    ext_low = percentile(ext, 10.0)
    curl_high = percentile(curl, 90.0)
    gap = ext_low - curl_high
    if gap < min_gap:
        note = (
            f"the {label}'s extended and curled readings overlap: extended "
            f"low is {ext_low:.1f} deg, curled high is {curl_high:.1f} deg — "
            f"a gap of {gap:.1f} deg (need at least {min_gap:.1f}) — "
            f"keeping the default thresholds"
        )
        return FingerResult(finger, None, None, ext_low, curl_high, gap, "overlap", note)

    mid = (ext_low + curl_high) / 2.0
    band = min(max(min_hyst, 0.3 * gap), gap)
    extend = mid + band / 2.0
    curl_thr = mid - band / 2.0
    status = "ok"
    note = (
        f"derived from a {gap:.1f} deg gap "
        f"(curled up to {curl_high:.1f}, extended from {ext_low:.1f})"
    )

    if relaxed:
        r75 = percentile(relaxed, 75.0)
        if r75 >= extend:
            candidate = min(r75 + 5.0, ext_low - 3.0)
            if candidate > extend and candidate - curl_thr >= min_hyst:
                extend = candidate
                note += (
                    f"; extend raised to {candidate:.1f} so your relaxed "
                    f"{label} stays disengaged"
                )
            else:
                status = "relaxed_overlap"
                note = (
                    f"your relaxed {label} reads as extended — "
                    f"expect occasional false engages"
                )

    extend = min(extend, 178.0)
    curl_thr = max(curl_thr, 2.0)
    return FingerResult(
        finger, round(extend, 1), round(curl_thr, 1),
        ext_low, curl_high, gap, status, note,
    )


@dataclass
class CalibrationResult:
    """Full calibration outcome: per-finger thresholds, human-readable
    warnings for anything non-"ok", and the validation replay verdicts."""

    fingers: dict[str, FingerResult]
    warnings: list[str] = field(default_factory=list)
    validation: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """JSON-ready shape consumed by the panel."""
        return {
            "fingers": {
                f: {
                    "extend": r.extend,
                    "curl": r.curl,
                    "gap": r.gap,
                    "status": r.status,
                    "note": r.note,
                }
                for f, r in self.fingers.items()
            },
            "warnings": list(self.warnings),
            "validation": dict(self.validation),
        }


def _signature_matches(sig: Signature, extended: dict[str, bool]) -> bool:
    # Same equality rule as the engine matcher: every non-"any" gating
    # entry must agree with the latched state; missing fingers are "any".
    for f in FINGERS:
        want = sig.get(f, "any")
        if want == "ext" and not extended[f]:
            return False
        if want == "curl" and extended[f]:
            return False
    return True


class CalibrationSession:
    """Sample collector + finite-state machine for one calibration run.

    States: "await_step" (idle, waiting for the UI to start a step),
    "sampling" (a step is live), "done" (every step in ``STEPS`` has
    completed at least once — only then is ``compute()`` meaningful).
    Steps may be re-begun at any time (retry replaces that step's samples
    on completion; a timed-out attempt is discarded without touching a
    previous successful take)."""

    SETTLE_MS: float = 750.0        # discard the hand-settling transition
    TARGET_SAMPLES: int = 90        # ~3 s at 30 fps
    MIN_SAMPLES: int = 45           # enough for stable P10/P90 (>= 40 + margin)
    STEP_TIMEOUT_MS: float = 15000.0

    def __init__(self) -> None:
        self._samples: dict[str, list[dict[str, float]]] = {}  # completed takes
        self._completed: set[str] = set()
        self._current: str | None = None
        self._begin_ts: float = 0.0
        self._attempt: list[dict[str, float]] = []
        self._failed: str | None = None

    @property
    def state(self) -> str:
        if self._current is not None:
            return "sampling"
        if self._completed >= _ALL_STEP_IDS:
            return "done"
        return "await_step"

    def begin_step(self, step_id: str, ts_ms: float) -> None:
        """Start (or restart — retry is always allowed) sampling a step."""
        if step_id not in _STEP_BY_ID:
            raise ValueError(f"unknown calibration step: {step_id!r}")
        self._current = step_id
        self._begin_ts = float(ts_ms)
        self._attempt = []
        self._failed = None

    def add_sample(self, ts_ms: float, angles: dict[str, float]) -> None:
        """Feed one frame's finger angles. No-op unless sampling; empty
        dicts (no valid hand) and settle-window frames are discarded. The
        step resolves off the incoming ``ts_ms`` stream: TARGET_SAMPLES
        completes it early; hitting STEP_TIMEOUT_MS completes it with
        whatever it has if that is at least MIN_SAMPLES, else fails the
        attempt (samples discarded, ``progress()`` exposes the failure so
        the UI can offer a retry)."""
        if self._current is None:
            return
        if ts_ms - self._begin_ts >= self.STEP_TIMEOUT_MS:
            if len(self._attempt) >= self.MIN_SAMPLES:
                self._complete()
            else:
                self._failed = self._current
                self._current = None
                self._attempt = []
            return
        if not angles:
            return
        if ts_ms - self._begin_ts < self.SETTLE_MS:
            return
        self._attempt.append(dict(angles))
        if len(self._attempt) >= self.TARGET_SAMPLES:
            self._complete()

    def _complete(self) -> None:
        assert self._current is not None
        self._samples[self._current] = self._attempt
        self._completed.add(self._current)
        self._current = None
        self._attempt = []

    def cancel(self) -> None:
        """Abort the whole run: all takes and progress are dropped."""
        self._samples = {}
        self._completed = set()
        self._current = None
        self._attempt = []
        self._failed = None

    def progress(self) -> dict:
        """JSON-ready snapshot for the panel UI."""
        if self._current is not None:
            step_i: int | None = next(
                i for i, s in enumerate(STEPS) if s.id == self._current
            )
        else:
            step_i = next(
                (i for i, s in enumerate(STEPS) if s.id not in self._completed),
                None,
            )
        return {
            "state": self.state,
            "step": self._current,
            "step_i": step_i,
            "n_steps": len(STEPS),
            "collected": len(self._attempt),
            "needed": self.TARGET_SAMPLES,
            "done_steps": [s.id for s in STEPS if s.id in self._completed],
            "failed_step": self._failed,
        }

    # -- threshold derivation + validation -----------------------------------

    def compute(
        self,
        defaults: tuple[float, float],
        custom_sigs: dict[str, Signature],
    ) -> CalibrationResult:
        """Derive per-finger thresholds from the completed takes and replay
        those takes against them (see module docstring for the why)."""
        if self.state != "done":
            raise ValueError("calibration incomplete: not all steps are done")

        ext_l: dict[str, list[float]] = {f: [] for f in ALL_FINGERS}
        curl_l: dict[str, list[float]] = {f: [] for f in ALL_FINGERS}
        relax_l: dict[str, list[float]] = {f: [] for f in ALL_FINGERS}
        for step in STEPS:
            samples = self._samples[step.id]
            if step.id == _RELAXED_STEP_ID:
                for s in samples:
                    for f in ALL_FINGERS:
                        if f in s:
                            relax_l[f].append(s[f])
                continue
            for f in FINGERS:
                want = step.expected.get(f)
                bucket = ext_l if want == "ext" else curl_l if want == "curl" else None
                if bucket is not None:
                    for s in samples:
                        if f in s:
                            bucket[f].append(s[f])
            thumb_want = THUMB_SOURCES.get(step.id)
            if thumb_want is not None:
                bucket = ext_l if thumb_want == "ext" else curl_l
                for s in samples:
                    if "thumb" in s:
                        bucket["thumb"].append(s["thumb"])

        fingers = {
            f: derive_thresholds(
                ext_l[f], curl_l[f], relax_l[f], defaults, finger=f,
            )
            for f in ALL_FINGERS
        }
        warnings = [
            f"{f}: {fingers[f].note}"
            for f in ALL_FINGERS
            if fingers[f].status != "ok"
        ]

        # Effective per-finger thresholds for the replay: derived values are
        # used even for relaxed_overlap (they ARE what will be persisted);
        # insufficient/overlap fingers fall back to the global defaults.
        eff: dict[str, tuple[float, float]] = {}
        for f in FINGERS:
            r = fingers[f]
            if r.status in ("ok", "relaxed_overlap"):
                assert r.extend is not None and r.curl is not None
                eff[f] = (r.extend, r.curl)
            else:
                eff[f] = (float(defaults[0]), float(defaults[1]))

        validation: dict[str, bool] = {}
        for step in STEPS:
            if step.id == _RELAXED_STEP_ID:
                continue
            validation[step.id] = self._replay_step(step, eff)
        validation["relaxed_matches_nothing"] = (
            self._replay_relaxed(eff, custom_sigs, seed_extended=False) >= 0.95
            and self._replay_relaxed(eff, custom_sigs, seed_extended=True) >= 0.95
        )
        return CalibrationResult(fingers, warnings, validation)

    def _replay_step(self, step: CalibStep, eff: dict[str, tuple[float, float]]) -> bool:
        # Fresh latches start curled; the first 10 samples are warm-up
        # (latch state converging from the arbitrary initial state).
        latches = {f: FingerLatch() for f in FINGERS}
        total = 0
        hits = 0
        for i, s in enumerate(self._samples[step.id]):
            for f in FINGERS:
                if f in s:
                    latches[f].update(s[f], *eff[f])
            if i < 10:
                continue
            total += 1
            ok = True
            for f, want in step.expected.items():
                if want == "ignore":
                    continue
                if latches[f].extended != (want == "ext"):
                    ok = False
                    break
            hits += ok
        return total > 0 and hits / total >= 0.90

    def _replay_relaxed(
        self,
        eff: dict[str, tuple[float, float]],
        custom_sigs: dict[str, Signature],
        seed_extended: bool,
    ) -> float:
        """Fraction of relaxed samples whose latched four-finger state
        equals NO known signature. Run from both latch initializations: a
        relaxed hand sitting inside the hysteresis band never crosses
        either threshold, so it simply keeps its seed state — and if that
        seed state matches a signature (all-extended == open_palm), an idle
        hand would hold a pose forever."""
        latches = {f: FingerLatch() for f in FINGERS}
        if seed_extended:
            for latch in latches.values():
                latch.extended = True
        sigs = list(BUILTINS.values()) + list((custom_sigs or {}).values())
        samples = self._samples[_RELAXED_STEP_ID]
        if not samples:
            return 0.0
        hits = 0
        for s in samples:
            for f in FINGERS:
                if f in s:
                    latches[f].update(s[f], *eff[f])
            state = {f: latches[f].extended for f in FINGERS}
            if not any(_signature_matches(sig, state) for sig in sigs):
                hits += 1
        return hits / len(samples)
