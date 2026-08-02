"""Claude Code ``UserPromptSubmit`` hook adapter.

Delivers oiax route hits into the Claude Code hook context paragraph.
Registered in ``~/.claude/settings.json`` under ``hooks.UserPromptSubmit``.

**Invocation (settings.json):** ``python3 -m oiax.adapters.claude_code``
``<corpus-dir> [--expansions PATH]``

**Design invariants:**

- **Surface names, never rules.** A route is probabilistic; showing matched
  terms makes a bad match dismissible (oiax positioning doc §4 decision 3).
- **Never blocks, never fails closed.** Any error exits 0 silently.
- **Pin what the prompt cannot reveal; route what it can.** Always-resident
  policies live in the resident context layer, not here (decision 4).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from oiax import build_index, route
from oiax.corpus import PolicyDirCorpus
from oiax.router import RouteHit

logger = logging.getLogger(__name__)

MIN_PROMPT_CHARS = 12

HEADER = (
    "Policy check — this prompt matches the standing policies below. "
    "They may or may not apply; the matched terms are shown so you can dismiss "
    "a bad match immediately. If one applies, load its skill BEFORE answering."
)

FOOTER = (
    "Matching is advisory. A policy not listed here may still "
    "govern — the skill descriptions remain the primary routing surface."
)


def _render(hits: list[RouteHit]) -> str:
    """Render route hits as Claude Code additionalContext markdown."""
    lines = [HEADER, ""]
    for hit in hits:
        terms = ", ".join(f"`{t}`" for t in hit.why)
        lines.append(f"- **`{hit.name}`** — matched {terms}")
    lines.append("")
    lines.append(FOOTER)
    return "\n".join(lines)


def _load_expansions(path: str | None) -> dict[str, str] | None:
    """Load optional query-expansion phrases from a JSON file."""
    if path is None:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data
    except Exception as exc:
        logger.warning("could not load expansions from %s: %s", path, exc)
        return None


def main(argv: list[str] | None = None) -> int:
    """Read ``UserPromptSubmit`` stdin, route, emit hook JSON.
    CLI: ``<corpus-dir> [--expansions PATH] [--lex-threshold F] [--sem-threshold F]``
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_dir")
    parser.add_argument("--expansions", default=None)
    parser.add_argument("--lex-threshold", type=float, default=0.15)
    parser.add_argument("--sem-threshold", type=float, default=0.55)
    parsed = parser.parse_args(argv)

    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or len(prompt.strip()) < MIN_PROMPT_CHARS:
        return 0

    try:
        corpus = PolicyDirCorpus(parsed.corpus_dir)
        expansions = _load_expansions(parsed.expansions)
        index = build_index(
            corpus,
            expansions=expansions,
            lex_threshold=parsed.lex_threshold,
            sem_threshold=parsed.sem_threshold,
        )
        hits = route(prompt, index)
    except Exception as exc:
        logger.warning("route failed: %s", exc)
        return 0

    if not hits:
        return 0

    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": _render(hits),
            }
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
