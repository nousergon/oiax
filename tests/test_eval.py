"""Tests for oiax eval harness."""

import json
import tempfile
from io import StringIO
from pathlib import Path

import pytest

from oiax import build_index, route
from oiax.corpus import PolicyDirCorpus
from oiax.eval.route_eval import EvalResult, aggregate, evaluate, load_labelled, score_labelled


def test_eval_result_recall_perfect():
    """All expected documents found = 1.0 recall."""
    r = EvalResult(
        prompt="test",
        expected=["a", "b"],
        routed=["a", "b", "c"],
    )
    assert r.recall == 1.0


def test_eval_result_recall_partial():
    """Half expected documents found = 0.5 recall."""
    r = EvalResult(
        prompt="test",
        expected=["a", "b", "c", "d"],
        routed=["a", "b"],
    )
    assert r.recall == 0.5


def test_eval_result_recall_empty_expected():
    """Empty expected = 1.0 recall (nothing to miss)."""
    r = EvalResult(prompt="test", expected=[], routed=["a"])
    assert r.recall == 1.0


def test_eval_result_precision_perfect():
    """All routed documents are expected = 1.0 precision."""
    r = EvalResult(
        prompt="test",
        expected=["a", "b"],
        routed=["a", "b"],
    )
    assert r.precision == 1.0


def test_eval_result_precision_with_false_positives():
    """Some routed documents are not expected."""
    r = EvalResult(
        prompt="test",
        expected=["a"],
        routed=["a", "b", "c"],  # b and c are false positives
    )
    assert r.precision == 1.0 / 3.0


def test_eval_result_precision_empty_routed():
    """Nothing routed = 1.0 precision (no false positives)."""
    r = EvalResult(prompt="test", expected=["a"], routed=[])
    assert r.precision == 1.0


def test_score_labelled_with_real_corpus():
    """Score labelled examples against a real corpus."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "deploy-policy.md").write_text(
            "# Deploy\n\n**Agent-trigger:** deploying to production\n\nBody.\n"
        )
        (Path(tmp) / "security-policy.md").write_text(
            "# Security\n\n**Agent-trigger:** security vulnerability scanning\n\nBody.\n"
        )

        labelled = StringIO(
            json.dumps({"prompt": "how do I deploy to prod?", "expected": ["deploy-policy"]})
            + "\n"
            + json.dumps({"prompt": "what's for lunch?", "expected": []})
            + "\n"
        )
        import sys
        old_stdin = sys.stdin
        sys.stdin = labelled
        try:
            results = score_labelled(tmp)
        finally:
            sys.stdin = old_stdin

        assert len(results) == 2
        deploy_result = results[0]
        assert deploy_result.expected == ["deploy-policy"]
        # The deploy-related prompt should route deploy-policy
        if deploy_result.routed:
            assert "deploy-policy" in deploy_result.routed


def test_score_labelled_ignores_invalid_json():
    """Malformed JSON lines are skipped, not fatal."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "p.md").write_text("**Agent-trigger:** test\n\nBody.\n")
        labelled = StringIO(
            '{"prompt": "valid", "expected": []}\n'
            'not json\n'
            '{"prompt": "also valid", "expected": []}\n'
        )
        import sys
        old_stdin = sys.stdin
        sys.stdin = labelled
        try:
            results = score_labelled(tmp)
        finally:
            sys.stdin = old_stdin
        assert len(results) == 2


# ── quality guards over the reference corpus (I5) ───────────────────────────
#
# These three exercise the PRIMARY path end to end: they download the embedding
# model (~90 MB, cached by CI's pip/HF cache after the first run) and route real
# prompts. That cost is deliberate. The 0.1.1 model-id defect and the 0.1.2 inert
# threshold both survived a green suite precisely because every existing test
# either mocked the embedder away or asserted only a shape.

CORPORA = Path(__file__).resolve().parent.parent / "src" / "oiax" / "eval" / "corpora"
REFERENCE_DIR = CORPORA / "reference-policies"
REFERENCE_LABELS = CORPORA / "reference_labelled.jsonl"

# Measured 2026-08-03 at the shipped defaults: recall 0.648, top-1 0.673, false
# alarms 0.000 (see corpora/README.md). The floors sit below the measurement with
# room for embedding-model jitter, and RATCHET: raise them when a change earns it,
# never lower them to make a regression pass.
RECALL_FLOOR = 0.60
TOP1_FLOOR = 0.60
SEPARATION_FLOOR = 0.15


@pytest.fixture(scope="module")
def labelled():
    with REFERENCE_LABELS.open(encoding="utf-8") as fh:
        return load_labelled(fh)


def test_reference_corpus_documents_are_separable():
    """Distinct documents must embed to distinct vectors.

    The shipped synthetic corpus fails this — its five trigger lines are one
    templated sentence, pairwise cosine spread ~0.04 — which is why its recall is
    flat at every threshold and why it cannot calibrate a threshold. A corpus that
    cannot separate its own documents makes any number measured on it meaningless.
    """
    index = build_index(PolicyDirCorpus(str(REFERENCE_DIR)))
    # Reads the ONE implementation (`Index.corpus_separability`), which the
    # divergence signal also uses. Recomputing it here would be a second copy of
    # the metric, free to disagree with the number the layer actually acts on.
    spread = index.corpus_separability
    assert spread is not None, "embedding model did not load — cannot judge separation"
    assert spread > SEPARATION_FLOOR, f"corpus is degenerate: cosine spread {spread:.3f}"


def test_shipped_defaults_meet_the_recall_floor(labelled):
    """Routing quality at the SHIPPED defaults, on a real corpus. The ratchet.

    Fails against the 0.1.2 defaults (lex 0.15 / sem 0.55 with union selection),
    which measure recall 0.185 on this corpus.
    """
    m = aggregate(evaluate(str(REFERENCE_DIR), labelled))
    assert m.recall >= RECALL_FLOOR, f"recall@2 regressed to {m.recall:.3f}"
    assert m.top1_accuracy >= TOP1_FLOOR, f"top-1 regressed to {m.top1_accuracy:.3f}"
    assert m.false_alarm_rate == 0.0, "a negative prompt routed a policy"


def test_semantic_scorer_actually_contributes_at_shipped_defaults(labelled):
    """At least one labelled prompt must route on SEMANTIC evidence.

    The inertness guard. Through 0.1.2 the semantic floor (0.55) sat above every
    correct match this corpus produces (0.40-0.48), so the semantic half could not
    fire and the "hybrid" router was lexical-only — with nothing in the suite, the
    logs, or the output saying so.
    """
    index = build_index(PolicyDirCorpus(str(REFERENCE_DIR)))
    semantic_hits = sum(
        1
        for item in labelled
        for hit in route(item["prompt"], index)
        if "semantic match" in hit.why
    )
    assert semantic_hits > 0, "no prompt routed on semantic evidence — the floor is inert"


# ── harmful siblings ────────────────────────────────────────────────────────

SIBLING_LABELS = REFERENCE_DIR.parent / "reference_siblings.jsonl"

#: The ratchet, not the target. Published work reaches HSR 0 with within-family
#: representative selection; oiax has none, so this floor holds the line while
#: that stays true. A change that raises recall by returning more same-family
#: siblings is a REGRESSION, and no other metric here can see it.
HSR_CEILING = 0.10


@pytest.fixture(scope="module")
def sibling_labelled():
    with SIBLING_LABELS.open(encoding="utf-8") as fh:
        return load_labelled(fh)


def test_the_sibling_set_actually_carries_sibling_labels(sibling_labelled):
    # A sibling file with no sibling labels would make HSR vacuously 0 — a gate
    # that cannot fail is not a gate.
    labelled = [i for i in sibling_labelled if i.get("sibling")]
    assert len(labelled) >= 10


def test_every_sibling_is_a_real_document(sibling_labelled):
    names = {d.name for d in PolicyDirCorpus(str(REFERENCE_DIR)).documents()}
    for item in sibling_labelled:
        for s in item.get("sibling", []):
            assert s in names, f"sibling {s!r} is not in the corpus"
        for e in item["expected"]:
            assert e in names, f"gold {e!r} is not in the corpus"


def test_a_sibling_is_never_also_gold(sibling_labelled):
    # The label means "same family and WRONG for this prompt". A document that
    # is both would make the metric incoherent.
    for item in sibling_labelled:
        assert not (set(item.get("sibling", [])) & set(item["expected"]))


def test_the_set_keeps_a_both_govern_case(sibling_labelled):
    # Two members of one family may legitimately both govern. Without this row,
    # a representative selector that collapses every family to one document
    # would look like a pure improvement.
    assert any(len(i["expected"]) > 1 for i in sibling_labelled)


def test_harmful_sibling_rate_does_not_regress(sibling_labelled):
    m = aggregate(evaluate(str(REFERENCE_DIR), sibling_labelled))
    assert m.n_sibling >= 10, "HSR needs a denominator"
    assert m.hsr <= HSR_CEILING, f"HSR@2 regressed to {m.hsr:.3f}"


# ── known misses ─────────────────────────────────────────────────────────────

KNOWN_MISSES = CORPORA / "known_misses.jsonl"


@pytest.fixture(scope="module")
def known():
    with KNOWN_MISSES.open(encoding="utf-8") as fh:
        return load_labelled(fh)


def test_known_misses_file_exists():
    assert KNOWN_MISSES.exists(), "known_misses.jsonl must be in the eval package"


def test_known_misses_has_entries(known):
    assert len(known) >= 1, "at least one known miss must be on file"


def test_known_misses_every_entry_has_expected(known):
    for item in known:
        assert item.get("prompt"), "every known miss must have a prompt"
        assert item.get("expected"), "every known miss must have expected labels"
        assert item.get("recorded"), "every known miss must name when it was recorded"


def test_known_misses_the_current_operative_miss_is_still_active(known):
    results = evaluate(str(REFERENCE_DIR), known)
    active = [r for r in results if r.missed]
    assert len(active) >= 1, (
        "the standing known miss has recovered — this is worth naming in the "
        "changelog and in the issue that added the missing document"
    )


def test_known_misses_recovery_is_detectable():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "test-policy.md").write_text(
            "# Test\n\n**Agent-trigger:** deploying to production\n\nBody.\n"
        )
        item = {"prompt": "how do I deploy to prod?", "expected": ["test-policy"],
                "recorded": "2026-08-03", "note": "test miss"}
        results = evaluate(tmp, [item])
        recovered = [r for r in results if not r.missed]
        active = [r for r in results if r.missed]
        assert len(recovered) == 1
        assert len(active) == 0


def test_known_misses_eval_result_carries_raw_item():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "p.md").write_text("**Agent-trigger:** deploying to production\n\nBody.\n")
        item = {"prompt": "how do I deploy to prod?", "expected": ["p"], "recorded": "2026-01-01",
                "note": "test note"}
        results = evaluate(tmp, [item])
        assert results[0]._item == item
        assert results[0]._item["note"] == "test note"


# ── arms record ──────────────────────────────────────────────────────────────

from oiax.eval.route_eval import _load_arms, _append_arm_entry, _supersede_arm_entry, print_arms
from io import StringIO


def test_arms_file_exists():
    assert (CORPORA / "ARMS.jsonl").exists(), "ARMS.jsonl must be in the eval package"


def test_arms_has_entry_for_shipped():
    arms = _load_arms(str(CORPORA / "ARMS.jsonl"))
    active = [a for a in arms if not a.get("superseded_by")]
    assert len(active) >= 1, "at least one active arm must exist"


def test_shipped_arm_id_is_in_the_arms_record():
    from oiax.calibration import SHIPPED
    arms = _load_arms(str(CORPORA / "ARMS.jsonl"))
    arm_ids = {a["arm_id"] for a in arms}
    assert SHIPPED.arm_id in arm_ids, (
        f"SHIPPED.arm_id={SHIPPED.arm_id!r} not in arms record"
    )


def test_shipped_supersedes_the_entry_it_claims():
    from oiax.calibration import SHIPPED
    arms = _load_arms(str(CORPORA / "ARMS.jsonl"))
    if SHIPPED.superseded_id:
        superseded = [a for a in arms if a["arm_id"] == SHIPPED.superseded_id]
        assert len(superseded) == 1, f"superseded arm {SHIPPED.superseded_id} not found"
        assert superseded[0].get("superseded_by") == SHIPPED.arm_id, (
            f"superseded entry not marked with superseded_by={SHIPPED.arm_id}"
        )


def test_arms_append_and_supersede():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "arms.jsonl"

        _append_arm_entry(str(path), {
            "arm_id": "arm-a", "superseded_id": "", "superseded_by": "",
            "lex_floor": 0.10, "sem_floor": 0.25, "rrf_k": 60, "top_k": 2,
            "model_id": "m", "corpus_id": "c", "corpus_size": 10,
            "measured": "2026-01-01", "metrics": {"recall@2": 0.5},
            "metrics_n": 20, "metrics_negatives": 3,
        })

        _append_arm_entry(str(path), {
            "arm_id": "arm-b", "superseded_id": "arm-a", "superseded_by": "",
            "lex_floor": 0.05, "sem_floor": 0.30, "rrf_k": 60, "top_k": 2,
            "model_id": "m", "corpus_id": "c", "corpus_size": 10,
            "measured": "2026-02-01", "metrics": {"recall@2": 0.6},
            "metrics_n": 20, "metrics_negatives": 3,
        })

        _supersede_arm_entry(str(path), "arm-a", "arm-b")

        arms = _load_arms(str(path))
        assert len(arms) == 2

        active = [a for a in arms if not a.get("superseded_by")]
        assert len(active) == 1
        assert active[0]["arm_id"] == "arm-b"

        superseded = [a for a in arms if a.get("superseded_by")]
        assert len(superseded) == 1
        assert superseded[0]["arm_id"] == "arm-a"
        assert superseded[0]["superseded_by"] == "arm-b"


def test_arms_print_shows_active():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "arms.jsonl"
        _append_arm_entry(str(path), {
            "arm_id": "arm-z", "superseded_id": "", "superseded_by": "",
            "lex_floor": 0.10, "sem_floor": 0.25, "rrf_k": 60, "top_k": 2,
            "model_id": "m", "corpus_id": "c", "corpus_size": 10,
            "measured": "2026-01-01", "metrics": {"recall@2": 0.5},
            "metrics_n": 20, "metrics_negatives": 3,
        })
        buf = StringIO()
        import sys
        old = sys.stdout
        sys.stdout = buf
        try:
            print_arms(str(path))
        finally:
            sys.stdout = old
        output = buf.getvalue()
        assert "arm-z" in output
        assert "← active" in output


def test_arms_empty():
    buf = StringIO()
    import sys
    old = sys.stdout
    sys.stdout = buf
    try:
        print_arms("/nonexistent/path/arms.jsonl")
    finally:
        sys.stdout = old
    assert "No arms on file" in buf.getvalue()
