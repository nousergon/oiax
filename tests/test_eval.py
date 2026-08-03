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
