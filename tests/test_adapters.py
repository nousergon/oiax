"""Tests for oiax adapters."""

import json
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path

from oiax.router import RouteHit


def test_stdout_adapter_renders_hits():
    """stdout adapter renders route hits as text."""
    from oiax.adapters.stdout import deliver

    hits = [
        RouteHit(name="security-policy", score=0.85, why=["vulnerability", "report"]),
        RouteHit(name="deploy-policy", score=0.72, why=["deploy"]),
    ]
    output = deliver(hits, "how do I deploy?")
    assert "security-policy" in output
    assert "deploy-policy" in output
    assert "vulnerability" in output
    assert "0.85" in output or "0.72" in output


def test_stdout_adapter_empty_hits():
    """Empty hits produce empty string."""
    from oiax.adapters.stdout import deliver

    output = deliver([], "anything")
    assert output == ""


def test_claude_code_adapter_render():
    """Claude Code adapter renders hits as markdown, names only — never body."""
    from oiax.adapters.claude_code import _render

    hits = [
        RouteHit(name="test-policy", score=0.9, why=["test", "routing"]),
    ]
    rendered = _render(hits)
    assert "**`test-policy`**" in rendered
    assert "`test`" in rendered
    assert "`routing`" in rendered
    # Surface names only — never inlined body text
    assert "Body of" not in rendered


def test_claude_code_adapter_emits_valid_hook_json():
    """Adapter emits valid UserPromptSubmit hook JSON."""

    # Create a tiny policy corpus
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "test-policy.md").write_text(
            "# Test Policy\n\n**Agent-trigger:** governs testing deployment secrets scanning\n\n"
        )

        # Run the adapter with stdin payload — use prompt matching trigger terms
        input_payload = json.dumps({
            "prompt": "how do I scan for secrets during deployment?",
            "session_id": "test",
        })
        result = subprocess.run(
            [sys.executable, "-m", "oiax.adapters.claude_code", tmp],
            input=input_payload,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

        output = json.loads(result.stdout)
        assert "hookSpecificOutput" in output
        assert output["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        ctx = output["hookSpecificOutput"]["additionalContext"]
        # Should mention the matched policy by name
        assert "test-policy" in ctx
        # Should NOT contain raw body text
        assert "Body of" not in ctx


def test_claude_code_adapter_short_prompt_exits_early():
    """Short prompts (< MIN_PROMPT_CHARS) produce no output."""

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "p.md").write_text(
            "**Agent-trigger:** something\n\nBody.\n"
        )
        result = subprocess.run(
            [sys.executable, "-m", "oiax.adapters.claude_code", tmp],
            input=json.dumps({"prompt": "yes"}),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""


def test_claude_code_adapter_unparseable_input_exits_cleanly():
    """Invalid JSON stdin exits 0 (never blocks on confusion)."""
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "-m", "oiax.adapters.claude_code", tmp],
            input="not json",
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0


def test_claude_code_adapter_flags_lexical_only_routing():
    """A degraded (no-embeddings) route SAYS SO in the delivered context.

    The reference settings.json invokes this hook with ``2>/dev/null``, so a
    warning on stderr reaches no one. The context paragraph is the only surface
    that survives, so the degradation has to be rendered into it.
    """
    from oiax.adapters.claude_code import DEGRADED_NOTICE, _render

    hits = [RouteHit(name="deploy-policy", score=0.4, why=["deploy"])]
    assert DEGRADED_NOTICE in _render(hits, degraded=True)
    assert DEGRADED_NOTICE not in _render(hits, degraded=False)
    assert DEGRADED_NOTICE not in _render(hits)


def test_claude_code_adapter_uses_the_calibrated_library_defaults():
    """The adapter must not carry its own copy of the selection thresholds.

    Through 0.1.3 it hardcoded `--lex-threshold 0.15 --sem-threshold 0.55` as argparse
    DEFAULTS — the pre-calibration values. So the 0.1.3 recalibration reached the
    library, its tests and its eval harness, and did not reach the only deployment
    that exists: the hook kept routing at the old operating point (recall 0.185)
    while every measurement said 0.648. A default duplicated into a caller is not a
    default.
    """
    import argparse
    from unittest import mock

    from oiax.adapters import claude_code
    from oiax.router import LEX_FLOOR, SEM_FLOOR

    parser_defaults = {}

    real_parse = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        parsed = real_parse(self, args, namespace)
        parser_defaults.update(vars(parsed))
        return parsed

    captured = {}

    def fake_build_index(corpus, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after build_index")  # nothing else to exercise

    payload = json.dumps({"prompt": "how do I deploy this to production safely"})
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "deploy-policy.md").write_text(
            "# deploy\n\n**Agent-trigger:** deploying to production\n\nbody\n", encoding="utf-8"
        )
        with mock.patch.object(argparse.ArgumentParser, "parse_args", capture), \
             mock.patch.object(claude_code, "build_index", fake_build_index), \
             mock.patch("sys.stdin", StringIO(payload)):
            claude_code.main([tmp])

    assert parser_defaults["lex_threshold"] is None, "adapter pins its own lexical floor"
    assert parser_defaults["sem_threshold"] is None, "adapter pins its own semantic floor"
    assert "lex_threshold" not in captured, "adapter passed a threshold the caller never set"
    assert "sem_threshold" not in captured
    # And the library defaults it falls through to are the calibrated ones.
    assert (LEX_FLOOR, SEM_FLOOR) == (0.10, 0.25)


def test_claude_code_adapter_still_honours_an_explicit_threshold():
    """An explicitly passed flag must reach build_index — the flags stay usable."""
    from unittest import mock

    from oiax.adapters import claude_code

    captured = {}

    def fake_build_index(corpus, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after build_index")

    payload = json.dumps({"prompt": "how do I deploy this to production safely"})
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "deploy-policy.md").write_text(
            "# deploy\n\n**Agent-trigger:** deploying to production\n\nbody\n", encoding="utf-8"
        )
        with mock.patch.object(claude_code, "build_index", fake_build_index), \
             mock.patch("sys.stdin", StringIO(payload)):
            claude_code.main([tmp, "--sem-threshold", "0.42"])

    assert captured.get("sem_threshold") == 0.42
    assert "lex_threshold" not in captured
