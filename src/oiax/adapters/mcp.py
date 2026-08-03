"""MCP server adapter — routing as a tool any MCP-capable harness can call.

The Claude Code adapter delivers routes by *pushing* them into a per-turn hook.
Most harnesses have no such hook: Cursor's rules are model-judged with no
system-injection point, and Codex's are static. MCP is the injection point they
do have, so this adapter exposes the router as two tools an agent can call:

    route_policies(prompt)  -> [{name, score, why}]   surface names + evidence
    get_policy(name)        -> the document's full text

**Two tools, deliberately.** ``route_policies`` never returns a rule body — a
route is probabilistic, and inlining the text would assert a governing
relationship the router cannot establish (product decision 3). The agent decides
a route is relevant and then asks for the body. That also keeps the routing
response small enough to be worth calling on every turn.

**Whole documents, never chunks** (decision 1). ``get_policy`` returns the entire
file: a rule and its carve-out are semantically distant and logically
inseparable, so a chunk of a policy can invert the policy.

**The index lives in memory for the life of the process.** That is the point of a
server rather than a per-turn subprocess: the ~1.26 s a Claude Code turn pays is
817 ms of imports plus 328 ms of model load plus 110 ms of index build, all of
which happen once here, leaving ~6 ms per call (``oiax#9``).

Run it::

    oiax-mcp <corpus-dir>

and point a harness at that command over stdio.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from oiax import build_index, route, semantic_ready
from oiax.corpus import Document, PolicyDirCorpus

logger = logging.getLogger(__name__)

SERVER_NAME = "oiax"

INSTRUCTIONS = """\
oiax routes a prompt against a governance corpus by meaning and returns the
documents that bear on it.

Call `route_policies` with the user's request before acting on anything that a
standing rule might govern. It returns at most two SURFACE NAMES with the terms
that matched — not rule text — so a bad match is dismissible at a glance. When a
returned name looks relevant, call `get_policy` with that name to read the whole
document. Routing is advisory: a name that does not fit the task can be ignored,
and silence does not prove no policy applies.\
"""


def _route_tool_description(degraded: bool) -> str:
    """Description for ``route_policies``, naming the degradation when present.

    A lexical-only router answers the same call with materially worse recall. The
    tool description is the one surface an agent always reads before calling, so
    it is where the degradation has to appear — the same reasoning that put the
    notice in the Claude Code adapter's context paragraph rather than in a log.
    """
    base = (
        "Route a prompt against the governance corpus. Returns at most two "
        "matching policy documents as {name, score, why} — surface names and the "
        "evidence that matched, never rule text. Call `get_policy` to read a "
        "document. Call this before acting on any request a standing rule, "
        "convention, or policy might govern."
    )
    if degraded:
        return (
            base + " WARNING: the embedding model did not load, so this server is "
            "matching on keywords only and recall is materially worse than usual."
        )
    return base


def build_server(
    corpus_dir: str,
    **index_kwargs: Any,
) -> Any:
    """Build the MCP server over ``corpus_dir``, with the index already loaded.

    Index construction happens here, at server start, not per call. Raises if the
    corpus directory yields no documents — a server that answers every route with
    an empty list is worse than one that fails to start, because the agent reads
    "no policy governs this" from it.
    """
    from mcp.server.mcpserver import MCPServer

    corpus = PolicyDirCorpus(corpus_dir)
    documents = {doc.name: doc for doc in corpus.documents()}
    if not documents:
        raise ValueError(f"no policy documents found in {corpus_dir!r}")

    index = build_index(corpus, **index_kwargs)
    degraded = not semantic_ready()
    if degraded:
        logger.warning(
            "embedding model unavailable — serving lexical-only routes from %s", corpus_dir
        )

    server = MCPServer(name=SERVER_NAME, instructions=INSTRUCTIONS)

    @server.tool(name="route_policies", description=_route_tool_description(degraded))
    def route_policies(prompt: str) -> list[dict[str, Any]]:
        hits = route(prompt, index)
        return [
            {
                "name": hit.name,
                "score": round(hit.score, 3),
                "why": list(hit.why),
                # Stated per response, not only in the description: an agent that
                # cached the tool list at connect time would otherwise never see it.
                **({"degraded": True} if degraded else {}),
            }
            for hit in hits
        ]

    @server.tool(
        name="get_policy",
        description=(
            "Return the full text of one policy document by name, as given by "
            "`route_policies`. Whole document — a rule separated from its "
            "carve-out inverts the rule."
        ),
    )
    def get_policy(name: str) -> str:
        document: Document | None = documents.get(name)
        if document is None:
            known = ", ".join(sorted(documents)) or "(none)"
            raise ValueError(f"unknown policy {name!r}. Available: {known}")
        return f"# {document.name}\n\n**Agent-trigger:** {document.trigger_line}\n\n{document.body}"

    return server


def main(argv: list[str] | None = None) -> int:
    """Console entry point (``oiax-mcp``): serve the corpus over MCP stdio."""
    parser = argparse.ArgumentParser(
        prog="oiax-mcp",
        description="Serve an oiax policy corpus as an MCP server over stdio.",
    )
    parser.add_argument("corpus_dir")
    parser.add_argument("--lex-threshold", type=float, default=None)
    parser.add_argument("--sem-threshold", type=float, default=None)
    parsed = parser.parse_args(sys.argv[1:] if argv is None else argv)

    overrides = {
        key: value
        for key, value in (
            ("lex_threshold", parsed.lex_threshold),
            ("sem_threshold", parsed.sem_threshold),
        )
        if value is not None
    }
    server = build_server(
        parsed.corpus_dir,
        **overrides,
    )
    # Blocks until the client disconnects. stdio is the transport every MCP
    # harness supports; a URL transport would need auth this package does not own.
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
