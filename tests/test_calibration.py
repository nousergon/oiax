"""Tests for the operating point and the divergence signal.

The point of this feature is that a shipped constant stops being anonymous. So
the load-bearing tests are the ones about PROVENANCE and about the signal that
fires when the running corpus is not the one the number came from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oiax import build_index
from oiax.calibration import (
    SEPARABILITY_FLOOR,
    SHIPPED,
    SIZE_RATIO_LIMIT,
    CorpusStats,
    OperatingPoint,
    divergence,
    load_operating_point,
    save_operating_point,
)
from oiax.corpus import PolicyDirCorpus
from oiax.router import LEX_FLOOR, RRF_K, SEM_FLOOR, TOP_K

_CORPORA = Path(__file__).resolve().parent.parent / "src" / "oiax" / "eval" / "corpora"
REFERENCE = _CORPORA / "reference-policies"
SYNTHETIC = _CORPORA / "synthetic-policies"


# ── one default, one place ──────────────────────────────────────────────────


def test_router_constants_come_from_the_shipped_operating_point():
    # Re-stating the numbers in router.py would be the second copy that let a
    # recalibration reach the library, its tests and its eval harness while
    # missing the one deployment that existed (#11).
    assert (LEX_FLOOR, SEM_FLOOR, RRF_K, TOP_K) == (
        SHIPPED.lex_floor,
        SHIPPED.sem_floor,
        SHIPPED.rrf_k,
        SHIPPED.top_k,
    )


def test_the_shipped_default_carries_its_provenance():
    # A default presented without its provenance reads as a tuned universal.
    assert SHIPPED.has_provenance
    assert SHIPPED.corpus_size == 15
    assert "reference-policies" in SHIPPED.corpus_id
    assert SHIPPED.model_id and SHIPPED.measured
    assert "15 documents" in SHIPPED.describe()


def test_a_point_without_provenance_says_so_rather_than_looking_universal():
    bare = OperatingPoint(lex_floor=0.1, sem_floor=0.2, rrf_k=60, top_k=2)
    assert not bare.has_provenance
    assert "NO PROVENANCE" in bare.describe()


# ── precedence ──────────────────────────────────────────────────────────────


def test_operating_point_overrides_the_shipped_default():
    point = OperatingPoint(lex_floor=0.42, sem_floor=0.43, rrf_k=7, top_k=3)
    index = build_index(PolicyDirCorpus(REFERENCE), operating_point=point)
    assert index.top_k == 3 and index.rrf_k == 7
    assert index.operating_point is point


def test_an_explicit_kwarg_overrides_the_operating_point():
    # Three layers, one direction: kwarg > operating point > shipped default.
    point = OperatingPoint(lex_floor=0.42, sem_floor=0.43, rrf_k=7, top_k=3)
    index = build_index(PolicyDirCorpus(REFERENCE), operating_point=point, top_k=1)
    assert index.top_k == 1


def test_no_arguments_means_the_shipped_default():
    index = build_index(PolicyDirCorpus(REFERENCE))
    assert index.operating_point is SHIPPED
    assert index.top_k == SHIPPED.top_k


# ── separability, measured once ─────────────────────────────────────────────


def test_index_measures_its_own_separability():
    index = build_index(PolicyDirCorpus(REFERENCE))
    assert index.corpus_separability is not None
    # The shipped provenance claims 0.55; the live measurement must agree, or
    # the number in SHIPPED is a claim about a corpus that no longer exists.
    assert index.corpus_separability == pytest.approx(SHIPPED.corpus_separability, abs=0.05)


def test_the_synthetic_corpus_separates_less_than_the_reference_one():
    ref = build_index(PolicyDirCorpus(REFERENCE)).corpus_separability
    syn = build_index(PolicyDirCorpus(SYNTHETIC)).corpus_separability
    assert ref is not None and syn is not None
    assert syn < ref


# ── divergence ──────────────────────────────────────────────────────────────


def test_the_reference_corpus_does_not_diverge_from_its_own_calibration():
    assert build_index(PolicyDirCorpus(REFERENCE)).divergence() == []


def test_a_much_larger_corpus_diverges():
    stats = CorpusStats(size=int(SHIPPED.corpus_size * SIZE_RATIO_LIMIT) + 1, separability=0.55)
    reasons = divergence(stats, SHIPPED)
    assert any("recalibrate" in r for r in reasons)


def test_a_degenerate_corpus_names_the_routing_surfaces_not_the_floors():
    stats = CorpusStats(size=15, separability=SEPARABILITY_FLOOR - 0.01)
    reasons = divergence(stats, SHIPPED)
    # The fix for a corpus that cannot separate itself is the trigger lines, not
    # a threshold — saying otherwise would send the reader to tune noise.
    assert any("routing surfaces need to differ more" in r for r in reasons)


def test_a_different_model_diverges():
    stats = CorpusStats(size=15, separability=0.55, model_id="some/other-model")
    reasons = divergence(stats, SHIPPED)
    assert any("not comparable between models" in r for r in reasons)


def test_a_point_without_provenance_diverges_by_definition():
    bare = OperatingPoint(lex_floor=0.1, sem_floor=0.2, rrf_k=60, top_k=2)
    reasons = divergence(CorpusStats(size=15), bare)
    assert reasons and "nothing can check" in reasons[0]


def test_divergence_never_raises_on_an_unmeasured_corpus():
    assert divergence(CorpusStats(size=0), SHIPPED) == []


# ── persistence ─────────────────────────────────────────────────────────────


def test_round_trip(tmp_path):
    path = tmp_path / "op.json"
    save_operating_point(SHIPPED, path)
    assert load_operating_point(path) == SHIPPED


def test_loading_a_missing_file_raises_rather_than_falling_back(tmp_path):
    # A caller who passed a path meant to use it. Silently running the shipped
    # numbers instead is the failure this module exists to prevent.
    with pytest.raises(FileNotFoundError):
        load_operating_point(tmp_path / "nope.json")


def test_an_unknown_field_raises(tmp_path):
    path = tmp_path / "op.json"
    path.write_text(json.dumps({**SHIPPED.to_dict(), "lex_flor": 0.9}))
    with pytest.raises(ValueError):
        load_operating_point(path)


def test_a_missing_required_field_raises(tmp_path):
    path = tmp_path / "op.json"
    path.write_text(json.dumps({"lex_floor": 0.1}))
    with pytest.raises(ValueError):
        load_operating_point(path)


# ── the calibrate command ───────────────────────────────────────────────────


def test_calibrate_rediscovers_the_shipped_point_on_the_shipped_corpus(tmp_path, capsys):
    """The procedure and the shipped default must agree.

    If they did not, one of them would be wrong and nobody could say which — the
    exact state that "exposed rather than performed once by the maintainer" is
    supposed to end.
    """
    from oiax.eval.route_eval import calibrate, load_labelled

    labels = (_CORPORA / "reference_labelled.jsonl").read_text(encoding="utf-8").splitlines()
    items = load_labelled(labels)
    out = tmp_path / "op.json"
    rc = calibrate(str(REFERENCE), items, out=str(out), corpus_id="reference-policies")
    assert rc == 0
    captured = capsys.readouterr().out
    # The losing rows are printed, or the next reader re-runs the sweep to find
    # out what the winner was chosen over.
    assert "0.15  0.35" in captured or "0.15 0.35" in captured.replace("  ", " ")

    point = load_operating_point(out)
    assert (point.lex_floor, point.sem_floor) == (SHIPPED.lex_floor, SHIPPED.sem_floor)
    assert point.metrics["false_alarm_rate"] == 0.0
    assert point.corpus_size == 15


def test_calibrate_warns_when_the_labelled_set_has_no_negatives(tmp_path, capsys):
    from oiax.eval.route_eval import calibrate

    items = [{"prompt": "how do I roll back a bad deploy", "expected": ["deployment-policy"]}]
    calibrate(str(REFERENCE), items, out=str(tmp_path / "op.json"))
    err = capsys.readouterr().err
    # Without negatives the zero-false-alarm gate passes vacuously, and a gate
    # that cannot fail is not a gate.
    assert "NO negative prompts" in err
