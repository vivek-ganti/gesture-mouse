"""Tests for gesture_mouse.calibration: threshold derivation math (pinned
exactly — it is normative), the sampling session FSM, and the end-to-end
compute() + validation replay on deterministic synthetic angle streams
(random.Random with fixed seeds; no camera, no wall clock)."""
from __future__ import annotations

import dataclasses
import json
import random

import pytest

from gesture_mouse import calibration
from gesture_mouse.calibration import (
    STEPS,
    THUMB_SOURCES,
    CalibrationSession,
    derive_thresholds,
    percentile,
)
from gesture_mouse.signatures import BUILTINS, FINGERS

DEFAULTS = (160.0, 130.0)

# A valid five-finger angle sample for FSM tests where values don't matter.
ANY_ANGLES = {"thumb": 100.0, "index": 170.0, "middle": 60.0, "ring": 60.0, "pinky": 60.0}


# -- percentile ---------------------------------------------------------------

class TestPercentile:
    def test_single_value(self):
        for q in (0.0, 50.0, 100.0):
            assert percentile([42.5], q) == 42.5

    def test_all_equal(self):
        assert percentile([7.0] * 10, 33.3) == 7.0

    def test_interpolation_hand_computed(self):
        vals = [10.0, 20.0, 30.0, 40.0]
        # rank = q/100 * (n-1): q=25 -> 0.75 -> 10 + 0.75*(20-10)
        assert percentile(vals, 25.0) == pytest.approx(17.5)
        assert percentile(vals, 90.0) == pytest.approx(37.0)
        assert percentile(vals, 0.0) == 10.0
        assert percentile(vals, 100.0) == 40.0

    def test_input_order_irrelevant(self):
        assert percentile([40.0, 10.0, 30.0, 20.0], 25.0) == pytest.approx(17.5)

    def test_empty_and_bad_q_raise(self):
        with pytest.raises(ValueError):
            percentile([], 50.0)
        with pytest.raises(ValueError):
            percentile([1.0], 101.0)


# -- derive_thresholds --------------------------------------------------------

class TestDeriveThresholds:
    def test_clean_gaussian_clusters(self):
        rng = random.Random(1)
        ext = [rng.gauss(172.0, 4.0) for _ in range(90)]
        curl = [rng.gauss(60.0, 10.0) for _ in range(90)]
        r = derive_thresholds(ext, curl, [], DEFAULTS, finger="index")
        assert r.status == "ok"
        assert r.finger == "index"
        # Thresholds strictly inside the (curl_high, ext_low) gap.
        assert r.curl_high < r.curl < r.extend < r.ext_low
        assert r.extend - r.curl >= 8.0
        assert r.gap == pytest.approx(r.ext_low - r.curl_high)

    def test_exact_math_min_hyst_band(self):
        # gap 20 -> 0.3*gap = 6 < min_hyst 8 -> band 8, mid 90.
        r = derive_thresholds([100.0] * 45, [80.0] * 45, [], DEFAULTS)
        assert (r.ext_low, r.curl_high, r.gap) == (100.0, 80.0, 20.0)
        assert (r.extend, r.curl) == (94.0, 86.0)
        assert r.status == "ok"

    def test_exact_math_proportional_band(self):
        # gap 100 -> band 30, mid 110.
        r = derive_thresholds([160.0] * 45, [60.0] * 45, [], DEFAULTS)
        assert (r.extend, r.curl) == (125.0, 95.0)
        assert r.status == "ok"

    def test_overlap_clusters(self):
        rng = random.Random(2)
        ext = [rng.gauss(150.0, 15.0) for _ in range(90)]
        curl = [rng.gauss(140.0, 15.0) for _ in range(90)]
        r = derive_thresholds(ext, curl, [], DEFAULTS, finger="ring")
        assert r.status == "overlap"
        assert r.extend is None and r.curl is None
        assert r.gap < 15.0
        # The note carries the numbers in plain English.
        assert f"{r.ext_low:.1f}" in r.note
        assert f"{r.curl_high:.1f}" in r.note
        assert f"{r.gap:.1f}" in r.note

    @pytest.mark.parametrize("n_ext,n_curl", [(39, 90), (90, 39), (0, 0)])
    def test_insufficient_samples(self, n_ext, n_curl):
        r = derive_thresholds(
            [170.0] * n_ext, [60.0] * n_curl, [], DEFAULTS, finger="pinky"
        )
        assert r.status == "insufficient"
        assert r.extend is None and r.curl is None
        assert r.ext_low is None and r.curl_high is None and r.gap is None
        assert str(n_ext) in r.note and str(n_curl) in r.note

    def test_relaxed_nudge_adopted(self):
        # Derived pair 131.5/98.5; relaxed P75=140 >= 131.5 ->
        # candidate = min(145, 167) = 145; 145 - 98.5 >= 8 -> adopt.
        r = derive_thresholds([170.0] * 45, [60.0] * 45, [140.0] * 20, DEFAULTS)
        assert r.status == "ok"
        assert (r.extend, r.curl) == (145.0, 98.5)
        assert "relaxed" in r.note

    def test_relaxed_nudge_rejected(self):
        # min_hyst=12: gap 16 -> band 12 -> extend 98/curl 86. Relaxed
        # P75=99 >= 98 but candidate = min(104, ext_low-3=97) = 97 <= 98:
        # the nudge cannot fit -> derived pair kept, flagged.
        r = derive_thresholds(
            [100.0] * 45, [84.0] * 45, [99.0] * 10, DEFAULTS,
            min_hyst=12.0, finger="middle",
        )
        assert r.status == "relaxed_overlap"
        assert (r.extend, r.curl) == (98.0, 86.0)
        assert "false engages" in r.note
        assert "middle" in r.note

    def test_relaxed_below_extend_no_nudge(self):
        r = derive_thresholds([170.0] * 45, [60.0] * 45, [120.0] * 20, DEFAULTS)
        assert (r.extend, r.curl) == (131.5, 98.5)
        assert r.status == "ok"

    def test_no_relaxed_data_no_nudge(self):
        r = derive_thresholds([170.0] * 45, [60.0] * 45, [], DEFAULTS)
        assert (r.extend, r.curl) == (131.5, 98.5)

    def test_clamps(self):
        # min_hyst > gap forces band == gap, so extend == ext_low (179.9 ->
        # clamp 178.0) and curl == curl_high (0.5 -> clamp 2.0).
        r = derive_thresholds(
            [179.9] * 45, [0.5] * 45, [], DEFAULTS, min_hyst=1000.0
        )
        assert (r.extend, r.curl) == (178.0, 2.0)
        assert r.status == "ok"

    def test_rounding_to_one_decimal(self):
        # mid = (100.37+80.11)/2 = 90.24, band 8 -> 94.24/86.24 -> 94.2/86.2.
        r = derive_thresholds([100.37] * 45, [80.11] * 45, [], DEFAULTS)
        assert (r.extend, r.curl) == (94.2, 86.2)


# -- STEPS / THUMB_SOURCES ----------------------------------------------------

class TestSteps:
    def test_six_steps_in_order(self):
        assert [s.id for s in STEPS] == [
            "pointer", "open_palm", "scroll", "horns", "fist", "relaxed",
        ]

    def test_expected_maps_match_builtin_signatures(self):
        by_id = {s.id: s for s in STEPS}
        for name in ("pointer", "open_palm", "scroll", "horns"):
            assert by_id[name].expected == BUILTINS[name]
        assert by_id["fist"].expected == {f: "curl" for f in FINGERS}
        assert by_id["relaxed"].expected == {f: "ignore" for f in FINGERS}

    def test_labels_and_instructions_present(self):
        for s in STEPS:
            assert s.label and s.instruction

    def test_steps_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            STEPS[0].label = "nope"

    def test_thumb_sources(self):
        assert THUMB_SOURCES == {"open_palm": "ext", "fist": "curl"}


# -- session FSM ----------------------------------------------------------------

def complete_step(sess: CalibrationSession, step_id: str, t0: float) -> float:
    """Feed exactly TARGET_SAMPLES valid post-settle samples; returns the
    next free timestamp."""
    sess.begin_step(step_id, t0)
    ts = t0 + sess.SETTLE_MS
    for i in range(sess.TARGET_SAMPLES):
        sess.add_sample(ts + i * 33.0, dict(ANY_ANGLES))
    return ts + sess.TARGET_SAMPLES * 33.0 + 100.0


class TestSessionFSM:
    def test_initial_progress(self):
        p = CalibrationSession().progress()
        assert p == {
            "state": "await_step", "step": None, "step_i": 0, "n_steps": 6,
            "collected": 0, "needed": 90, "done_steps": [], "failed_step": None,
        }

    def test_begin_unknown_step_raises(self):
        with pytest.raises(ValueError):
            CalibrationSession().begin_step("wave", 0.0)

    def test_settle_window_discards(self):
        sess = CalibrationSession()
        sess.begin_step("pointer", 1000.0)
        for ts in (1000.0, 1500.0, 1749.9):
            sess.add_sample(ts, dict(ANY_ANGLES))
        assert sess.progress()["collected"] == 0
        sess.add_sample(1750.0, dict(ANY_ANGLES))  # exactly SETTLE_MS: counted
        assert sess.progress()["collected"] == 1

    def test_empty_angles_ignored(self):
        sess = CalibrationSession()
        sess.begin_step("pointer", 0.0)
        sess.add_sample(1000.0, {})
        assert sess.progress()["collected"] == 0

    def test_add_sample_noop_when_not_sampling(self):
        sess = CalibrationSession()
        sess.add_sample(1000.0, dict(ANY_ANGLES))
        assert sess.progress() == CalibrationSession().progress()

    def test_auto_complete_at_target(self):
        sess = CalibrationSession()
        complete_step(sess, "pointer", 0.0)
        p = sess.progress()
        assert p["state"] == "await_step"
        assert p["done_steps"] == ["pointer"]
        assert p["step"] is None and p["collected"] == 0
        assert p["step_i"] == 1  # next incomplete step is open_palm

    def test_timeout_with_too_few_samples_fails_step(self):
        sess = CalibrationSession()
        sess.begin_step("open_palm", 0.0)
        for i in range(10):  # sparse: far fewer than MIN_SAMPLES
            sess.add_sample(800.0 + i * 100.0, dict(ANY_ANGLES))
        sess.add_sample(15000.0, dict(ANY_ANGLES))  # past STEP_TIMEOUT_MS
        p = sess.progress()
        assert p["state"] == "await_step"
        assert p["failed_step"] == "open_palm"
        assert p["done_steps"] == []

    def test_retry_after_failure(self):
        sess = CalibrationSession()
        sess.begin_step("open_palm", 0.0)
        sess.add_sample(16000.0, dict(ANY_ANGLES))
        assert sess.progress()["failed_step"] == "open_palm"
        complete_step(sess, "open_palm", 20000.0)
        p = sess.progress()
        assert p["failed_step"] is None
        assert p["done_steps"] == ["open_palm"]

    def test_timeout_with_min_samples_completes(self):
        sess = CalibrationSession()
        sess.begin_step("fist", 0.0)
        for i in range(50):  # >= MIN_SAMPLES but < TARGET_SAMPLES
            sess.add_sample(750.0 + i, dict(ANY_ANGLES))
        sess.add_sample(15000.0, dict(ANY_ANGLES))
        p = sess.progress()
        assert p["done_steps"] == ["fist"]
        assert p["failed_step"] is None

    def test_restart_discards_current_attempt(self):
        sess = CalibrationSession()
        sess.begin_step("pointer", 0.0)
        for i in range(5):
            sess.add_sample(800.0 + i, dict(ANY_ANGLES))
        sess.begin_step("pointer", 2000.0)
        assert sess.progress()["collected"] == 0

    def test_cancel_clears_everything(self):
        sess = CalibrationSession()
        complete_step(sess, "pointer", 0.0)
        sess.begin_step("scroll", 50000.0)
        sess.add_sample(51000.0, dict(ANY_ANGLES))
        sess.cancel()
        assert sess.progress() == CalibrationSession().progress()

    def test_done_only_after_all_six(self):
        sess = CalibrationSession()
        t = 0.0
        for step in STEPS[:-1]:
            t = complete_step(sess, step.id, t)
            assert sess.state == "await_step"
        complete_step(sess, STEPS[-1].id, t)
        p = sess.progress()
        assert p["state"] == "done"
        assert p["done_steps"] == [s.id for s in STEPS]
        assert p["step_i"] is None

    def test_compute_before_done_raises(self):
        sess = CalibrationSession()
        with pytest.raises(ValueError):
            sess.compute(DEFAULTS, {})
        complete_step(sess, "pointer", 0.0)
        with pytest.raises(ValueError):
            sess.compute(DEFAULTS, {})
        sess.begin_step("scroll", 90000.0)  # sampling isn't done either
        with pytest.raises(ValueError):
            sess.compute(DEFAULTS, {})


# -- end-to-end compute() -------------------------------------------------------

EXT_MEAN, EXT_SD = 170.0, 3.0
CURL_MEAN, CURL_SD = 55.0, 5.0
RELAXED_MEAN, RELAXED_SD = 90.0, 3.0


def step_sample(rng: random.Random, step, relaxed_means: dict[str, float]) -> dict:
    a = {}
    for f in FINGERS:
        want = step.expected[f]
        if want == "ext":
            a[f] = rng.gauss(EXT_MEAN, EXT_SD)
        elif want == "curl":
            a[f] = rng.gauss(CURL_MEAN, CURL_SD)
        else:  # relaxed step
            a[f] = rng.gauss(relaxed_means.get(f, RELAXED_MEAN), RELAXED_SD)
    thumb_want = THUMB_SOURCES.get(step.id)
    if thumb_want == "ext":
        a["thumb"] = rng.gauss(EXT_MEAN, EXT_SD)
    elif thumb_want == "curl":
        a["thumb"] = rng.gauss(CURL_MEAN, CURL_SD)
    elif step.id == "relaxed":
        a["thumb"] = rng.gauss(RELAXED_MEAN, RELAXED_SD)
    else:
        # Thumb is sampled on every step but only open_palm/fist feed its
        # clusters; give the other steps a nondescript mid angle.
        a["thumb"] = rng.gauss(120.0, 10.0)
    return a


def run_full_session(relaxed_means: dict[str, float] | None = None,
                     seed: int = 7) -> CalibrationSession:
    rng = random.Random(seed)
    sess = CalibrationSession()
    t = 0.0
    for step in STEPS:
        sess.begin_step(step.id, t)
        ts = t + sess.SETTLE_MS
        for i in range(sess.TARGET_SAMPLES):
            sess.add_sample(ts + i * 33.0, step_sample(rng, step, relaxed_means or {}))
        t = ts + sess.TARGET_SAMPLES * 33.0 + 100.0
    assert sess.state == "done"
    return sess


class TestComputeEndToEnd:
    def test_clean_run_all_ok_all_validated(self):
        res = run_full_session().compute(DEFAULTS, {})
        for f in FINGERS:
            r = res.fingers[f]
            assert r.status == "ok", (f, r.note)
            assert r.curl_high < r.curl < r.extend < r.ext_low
            assert r.extend - r.curl >= 8.0
            assert 100.0 < r.extend < 160.0  # sensible for these clusters
        assert res.fingers["thumb"].status == "ok"  # advisory but derived
        assert res.warnings == []
        assert res.validation == {
            "pointer": True, "open_palm": True, "scroll": True,
            "horns": True, "fist": True, "relaxed_matches_nothing": True,
        }

    def test_relaxed_finger_above_extend_engages_nudge(self):
        # Ring idles at ~140 deg, above the mid-gap extend threshold: the
        # relaxed nudge must raise ring's extend above its relaxed cluster
        # while everything still validates (a lone extended ring matches no
        # built-in signature).
        res = run_full_session(relaxed_means={"ring": 140.0}).compute(DEFAULTS, {})
        ring = res.fingers["ring"]
        assert ring.status == "ok"
        assert "relaxed" in ring.note
        assert ring.extend >= 140.0
        assert ring.extend > res.fingers["index"].extend
        assert all(res.validation.values())

    def test_custom_signature_matching_relaxed_state_fails_validation(self):
        # A custom gesture equal to the latched relaxed state (all curled)
        # means an idle hand would hold that gesture -> replay must flag it.
        custom = {"rest": {f: "curl" for f in FINGERS}}
        res = run_full_session().compute(DEFAULTS, custom)
        assert res.validation["relaxed_matches_nothing"] is False
        assert res.validation["pointer"] is True  # step replays unaffected

    def test_as_dict_shape_is_json_ready(self):
        res = run_full_session().compute(DEFAULTS, {})
        d = res.as_dict()
        assert set(d) == {"fingers", "warnings", "validation"}
        assert set(d["fingers"]) == set(calibration.STEPS[0].expected) | {"thumb"}
        for entry in d["fingers"].values():
            assert set(entry) == {"extend", "curl", "gap", "status", "note"}
        json.dumps(d)  # must not raise
