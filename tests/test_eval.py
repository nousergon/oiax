"""Tests for oiax eval harness."""

import json
import tempfile
from io import StringIO
from pathlib import Path

from oiax.eval.route_eval import EvalResult, score_labelled


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
