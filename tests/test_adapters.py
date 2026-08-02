"""Tests for oiax adapters."""

import json
import subprocess
import sys
import tempfile
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
