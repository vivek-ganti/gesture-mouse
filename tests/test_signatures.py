"""Tests for gesture_mouse.signatures: the shared pose-signature vocabulary.

Matching behavior through the engine is exercised via make_bent_frame
(genuine PIP bends at exact angles); the pure helpers (conflicts,
normalization) are tested directly.
"""
from __future__ import annotations

import pytest

from gesture_mouse import signatures
from gesture_mouse.config import Config
from gesture_mouse.engine import GestureEngine
from gesture_mouse.signatures import (
    BUILTINS,
    FINGERS,
    FingerLatch,
    check_conflicts,
    compute_finger_angles,
    normalize_custom_entries,
    normalize_signature,
    signature_from_states,
    signatures_conflict,
)

from pose_fixtures import make_bent_frame


def make_engine() -> GestureEngine:
    return GestureEngine(Config(), lambda name, raw, ts: raw)


EXT, CURL = 180.0, 0.0


class TestMatcherTruthTable:
    @pytest.mark.parametrize(
        "pose,angles",
        [
            ("pointer", {"index": EXT, "middle": CURL, "ring": CURL, "pinky": CURL}),
            ("open_palm", {"index": EXT, "middle": EXT, "ring": EXT, "pinky": EXT}),
            ("horns", {"index": EXT, "middle": CURL, "ring": CURL, "pinky": EXT}),
            ("scroll", {"index": EXT, "middle": EXT, "ring": CURL, "pinky": CURL}),
        ],
    )
    def test_each_builtin_matches_its_own_shape_only(self, pose, angles):
        eng = make_engine()
        lm = eng._smoother.smooth(make_bent_frame(0.0, angles))
        for name, sig in BUILTINS.items():
            assert eng._match(sig, lm) is (name == pose)

    def test_any_finger_never_gates(self):
        eng = make_engine()
        lm = eng._smoother.smooth(
            make_bent_frame(0.0, {"index": EXT, "middle": CURL, "ring": CURL, "pinky": CURL})
        )
        assert eng._match({"index": "ext", "middle": "any"}, lm) is True
        assert eng._match({"index": "ext", "thumb": "ext"}, lm) is True  # thumb ignored

    def test_per_finger_thresholds_override_globals(self):
        cfg = Config()
        cfg.pose.fingers = {"index": {"extend": 100.0, "curl": 80.0}}
        eng = GestureEngine(cfg, lambda name, raw, ts: raw)
        # 120 deg: above index's calibrated extend (100), far below the
        # global extend (160) every other finger still uses.
        lm = eng._smoother.smooth(
            make_bent_frame(0.0, {"index": 120.0, "middle": 120.0, "ring": 120.0, "pinky": 120.0})
        )
        states = {f: eng._ext(f, lm) for f in FINGERS}
        assert states == {"index": True, "middle": False, "ring": False, "pinky": False}


class TestConflicts:
    def test_builtins_are_mutually_exclusive(self):
        names = list(BUILTINS)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert signatures_conflict(BUILTINS[a], BUILTINS[b]) is False

    def test_permissive_signature_conflicts_with_everything(self):
        assert check_conflicts({"index": "ext"}, BUILTINS) == list(BUILTINS)

    def test_identical_signature_conflicts(self):
        assert signatures_conflict(BUILTINS["horns"], dict(BUILTINS["horns"])) is True

    def test_single_separating_finger_resolves(self):
        # pointer vs horns differ only in pinky — that one finger separates.
        assert signatures_conflict(BUILTINS["pointer"], BUILTINS["horns"]) is False

    def test_ring_is_any_in_scroll_and_horns(self):
        # Anatomical: the ring cannot fully curl while middle (scroll) or
        # pinky (horns) is extended — it must never gate those poses.
        assert BUILTINS["scroll"]["ring"] == "any"
        assert BUILTINS["horns"]["ring"] == "any"


class TestNormalizeSignature:
    def test_valid_passthrough(self):
        sig = normalize_signature({"index": "ext", "middle": "curl"})
        assert sig == {"index": "ext", "middle": "curl"}

    def test_thumb_forced_to_any(self):
        sig = normalize_signature({"index": "ext", "thumb": "ext"})
        assert sig == {"index": "ext", "thumb": "any"}

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            {},
            {"indx": "ext"},               # typo'd finger name
            {"index": "extended"},          # invalid state
            {"index": "any"},               # no gating constraint at all
            {"thumb": "ext"},               # thumb-only -> no gating constraint
            "index:ext",                    # not a dict
        ],
    )
    def test_invalid_rejected(self, raw):
        assert normalize_signature(raw) is None


class TestNormalizeCustomEntries:
    ACTION = {"type": "key", "key": "option"}

    def test_legacy_horns_entry_maps_to_builtin_signature(self):
        parsed, skipped = normalize_custom_entries(
            [{"name": "dictate", "pose": "horns", "action": dict(self.ACTION)}]
        )
        assert skipped == []
        assert parsed[0]["name"] == "dictate"
        assert parsed[0]["signature"] == BUILTINS["horns"]
        assert parsed[0]["hold_ms"] == 300.0 and parsed[0]["cooldown_ms"] == 1200.0

    def test_v2_signature_entry(self):
        parsed, skipped = normalize_custom_entries([{
            "name": "two-up", "signature": {"middle": "ext", "ring": "ext"},
            "hold_ms": 500, "action": dict(self.ACTION),
        }])
        assert skipped == []
        assert parsed[0]["signature"] == {"middle": "ext", "ring": "ext"}
        assert parsed[0]["hold_ms"] == 500.0

    def test_junk_surfaces_in_skipped_never_silently(self):
        parsed, skipped = normalize_custom_entries([
            "not-a-dict",
            {"name": "bad-pose", "pose": "nosuch", "action": dict(self.ACTION)},
            {"name": "no-action", "signature": {"index": "ext"}},
            {"name": "bad-sig", "signature": {"indx": "ext"}, "action": dict(self.ACTION)},
        ])
        assert parsed == []
        assert skipped == ["not-a-dict", "bad-pose", "no-action", "bad-sig"]

    def test_malformed_numbers_and_nonfinite_skip_instead_of_raising(self):
        # json.loads accepts NaN/Infinity and "1e999" -> inf; a hand-edited
        # config must never crash the parser (CONTRACTS: skipped, not raised)
        # nor smuggle non-finite floats into config.json / the SSE stream.
        parsed, skipped = normalize_custom_entries([
            {"name": "junk-hold", "signature": {"index": "ext"},
             "hold_ms": "not-a-number", "action": dict(self.ACTION)},
            {"name": "nan-hold", "signature": {"index": "ext"},
             "hold_ms": float("nan"), "action": dict(self.ACTION)},
            {"name": "inf-cool", "signature": {"index": "ext"},
             "cooldown_ms": float("inf"), "action": dict(self.ACTION)},
            {"name": "action-not-dict", "signature": {"index": "ext"},
             "action": "open -a Calculator"},
        ])
        assert parsed == []
        assert skipped == ["junk-hold", "nan-hold", "inf-cool", "action-not-dict"]


class TestSignatureFromStates:
    def test_pins_all_four_gating_fingers(self):
        # Capture always pins all four (no "any" from capture in v1) — so a
        # captured rock sign is STRICTER than the builtin horns signature
        # (whose ring is "any") and correctly conflicts with it.
        sig = signature_from_states({"index": True, "middle": False, "ring": False, "pinky": True})
        assert sig == {"index": "ext", "middle": "curl", "ring": "curl", "pinky": "ext"}
        assert "thumb" not in sig
        assert signatures_conflict(sig, BUILTINS["horns"]) is True


class TestComputeFingerAngles:
    def test_matches_bent_frame_inputs(self):
        frame = make_bent_frame(0.0, {"index": 90.0, "middle": 180.0, "ring": 45.0, "pinky": 0.0})
        angles = compute_finger_angles(frame.landmarks)
        assert angles["index"] == pytest.approx(90.0)
        assert angles["middle"] == pytest.approx(180.0)
        assert angles["ring"] == pytest.approx(45.0)
        assert angles["pinky"] == pytest.approx(0.0)
        assert "thumb" in angles  # sampled even though it never gates


class TestFingerLatchAlias:
    def test_engine_reexports_are_the_signatures_objects(self):
        from gesture_mouse.engine import _FingerState, _pip_angle_deg
        assert _FingerState is FingerLatch
        assert _pip_angle_deg is signatures.pip_angle_deg
