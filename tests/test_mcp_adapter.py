"""Contract tests for the MCP server adapter.

These drive the server through a real MCP **client** over the protocol rather than
calling the handler functions directly. That distinction is the point: the defects
this package shipped in its first week were all cases where each half worked and
the pairing was never exercised (a model id nothing loaded, a threshold nothing
could clear, an adapter default that shadowed the library's). A test that imports
`route_policies` and calls it proves nothing about what an agent receives.

`mcp` is an optional extra (`pip install oiax[mcp]`), so the whole module skips
when it is absent — but it IS in the dev extra, so CI runs it.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import anyio
import pytest

pytest.importorskip("mcp", reason="oiax[mcp] extra not installed")

from mcp import Client  # noqa: E402

from oiax.adapters.mcp import build_server  # noqa: E402

DEPLOY = """\
# deploy-policy

**Agent-trigger:** deploying to production, canary rollouts, rollback criteria, release windows

Deploys go out behind a canary. Roll back on any regression; rolling forward past a
failed canary needs a named approver.
"""

PULL_REQUEST = """\
# pull-request-policy

**Agent-trigger:** opening a pull request, merging a branch, review approvals, linked issues

Every pull request states what changed and why. Two approvals for auth, billing, or a
public interface.
"""

INCIDENT = """\
# incident-response-policy

**Agent-trigger:** production outages, severity levels, paging, postmortems

Severity one is a total loss of a customer-facing capability. Every incident above
severity three gets a written postmortem.
"""


@pytest.fixture(scope="module")
def corpus_dir():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "deploy-policy.md").write_text(DEPLOY, encoding="utf-8")
        (root / "pull-request-policy.md").write_text(PULL_REQUEST, encoding="utf-8")
        (root / "incident-response-policy.md").write_text(INCIDENT, encoding="utf-8")
        yield str(root)


@pytest.fixture(scope="module")
def server(corpus_dir):
    return build_server(corpus_dir)


def _call(server, tool: str, args: dict):
    """Drive one tool call through the MCP client protocol; return the result."""

    async def run():
        async with Client(server) as client:
            return await client.call_tool(tool, args)

    return anyio.run(run)


def _list_tools(server):
    async def run():
        async with Client(server) as client:
            return await client.list_tools()

    return anyio.run(run)


# ── surface ─────────────────────────────────────────────────────────────────


def test_server_exposes_exactly_the_two_documented_tools(server):
    names = {tool.name for tool in _list_tools(server).tools}
    assert names == {"route_policies", "get_policy"}


def test_route_tool_description_tells_the_agent_when_to_call_it(server):
    """A tool description that only says what a tool does under-triggers.

    The agent reads this string and nothing else before deciding whether routing
    is relevant, so the trigger condition has to be in it.
    """
    (route_tool,) = [t for t in _list_tools(server).tools if t.name == "route_policies"]
    assert "before acting" in (route_tool.description or "").lower()


# ── routing ─────────────────────────────────────────────────────────────────


def test_route_returns_surface_names_and_evidence(server):
    result = _call(server, "route_policies", {"prompt": "I want to ship this to production today"})
    assert result.is_error is False
    hits = result.structured_content["result"]
    assert hits, "no policy routed for an obviously deploy-shaped prompt"
    assert hits[0]["name"] == "deploy-policy"
    assert hits[0]["why"], "a hit with no evidence is not dismissible"
    assert 0.0 <= hits[0]["score"] <= 1.0


def test_route_never_returns_rule_text(server):
    """Product decision 3: surface names, never rules.

    A route is probabilistic. Returning the body would assert a governing
    relationship the router cannot establish — and the agent has `get_policy`
    for when it decides the route is relevant.
    """
    result = _call(server, "route_policies", {"prompt": "rolling back a bad deploy"})
    serialized = json.dumps(result.structured_content)
    assert "canary" not in serialized
    assert "named approver" not in serialized
    for hit in result.structured_content["result"]:
        assert set(hit) <= {"name", "score", "why", "degraded"}


def test_route_caps_at_two_hits(server):
    result = _call(
        server, "route_policies", {"prompt": "deploy a pull request during an outage"}
    )
    assert len(result.structured_content["result"]) <= 2


def test_route_abstains_on_an_unrelated_prompt(server):
    result = _call(server, "route_policies", {"prompt": "what time is the all hands on thursday"})
    assert result.structured_content["result"] == []


# ── document fetch ──────────────────────────────────────────────────────────


def test_get_policy_returns_the_whole_document(server):
    """Whole document, never a chunk — a rule without its carve-out inverts it."""
    result = _call(server, "get_policy", {"name": "deploy-policy"})
    assert result.is_error is False
    text = result.content[0].text
    assert "**Agent-trigger:**" in text
    assert "Roll back on any regression" in text
    assert "named approver" in text  # the carve-out, not just the rule


def test_get_policy_names_the_available_documents_when_asked_for_an_unknown_one(server):
    """The agent's recovery path is the error text — so it has to be actionable."""
    result = _call(server, "get_policy", {"name": "does-not-exist"})
    assert result.is_error is True
    message = result.content[0].text
    assert "deploy-policy" in message and "pull-request-policy" in message


def test_a_routed_name_is_always_fetchable(server):
    """The two tools must agree on names, or the handoff between them breaks."""
    routed = _call(server, "route_policies", {"prompt": "who approves a merge"})
    for hit in routed.structured_content["result"]:
        assert _call(server, "get_policy", {"name": hit["name"]}).is_error is False


# ── the reason this adapter exists ──────────────────────────────────────────


def test_build_server_refuses_an_empty_corpus():
    """A server that routes nothing reads to an agent as 'no policy applies'."""
    with tempfile.TemporaryDirectory() as empty:
        with pytest.raises(ValueError, match="no policy documents"):
            build_server(empty)


def test_warm_calls_are_fast(server):
    """The resident index is the whole point (oiax#9).

    A Claude Code turn pays ~1.26 s: 817 ms of imports, 328 ms of model load,
    110 ms of index build, 6 ms of routing. Here all but the routing happen once
    at server start. The 50 ms bar is loose enough for CI jitter and still fails
    by two orders of magnitude if the index is ever rebuilt per call.
    """

    async def run():
        async with Client(server) as client:
            await client.call_tool("route_policies", {"prompt": "warm up"})
            start = time.perf_counter()
            for _ in range(5):
                await client.call_tool("route_policies", {"prompt": "deploying to production"})
            return (time.perf_counter() - start) / 5

    per_call = anyio.run(run)
    assert per_call < 0.050, f"warm route_policies took {per_call * 1000:.0f} ms per call"
