"""Claude Code ``UserPromptSubmit`` hook adapter.

Delivers oiax route hits into the Claude Code hook context paragraph.
Registered in ``~/.claude/settings.json`` under ``hooks.UserPromptSubmit``.

**Invocation (settings.json):** ``python3 -m oiax.adapters.claude_code``
``<corpus-dir>``

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

from oiax import build_index, route, semantic_ready
from oiax.calibration import load_operating_point
from oiax.corpus import PolicyDirCorpus
from oiax.router import RouteHit
from oiax.telemetry import RouteEvent, emit, install_env_sink, timer

logger = logging.getLogger(__name__)

MIN_PROMPT_CHARS = 12

ADAPTER_NAME = "claude_code"

HEADER = (
    "Policy check — this prompt matches the standing policies below. "
    "They may or may not apply; the matched terms are shown so you can dismiss "
    "a bad match immediately. If one applies, load its skill BEFORE answering."
)

FOOTER = (
    "Matching is advisory. A policy not listed here may still "
    "govern — the skill descriptions remain the primary routing surface."
)


DIVERGENCE_HEADER = (
    "⚠ oiax is running **far from its calibration** — the selection floors in "
    "force were measured on a different corpus, so recall and precision here are "
    "unmeasured. Reasons:"
)


DEGRADED_NOTICE = (
    "⚠ oiax is routing **lexical-only** — the embedding model did not load, so these "
    "are keyword matches with no semantic scoring. Treat recall as materially worse "
    "than usual and rely on the skill descriptions."
)


def _render(
    hits: list[RouteHit], degraded: bool = False, divergence: list[str] | None = None
) -> str:
    """Render route hits as Claude Code additionalContext markdown.

    ``degraded`` (no embedding model — lexical-only scoring) is rendered INTO the
    context paragraph rather than logged. This hook is invoked with ``2>/dev/null``
    in the reference settings.json, so stderr is the one surface guaranteed not to
    reach anyone; a silently lexical-only router is indistinguishable from a healthy
    one at the read side, which is exactly how the 0.1.1 model-id defect survived.
    """
    lines = [HEADER, ""]
    if degraded:
        lines.append(DEGRADED_NOTICE)
        lines.append("")
    if divergence:
        # Rendered INTO the paragraph for the same reason as the degradation
        # notice: the layer is running, and its numbers do not mean what its
        # documentation says. A log would reach nobody.
        lines.append(DIVERGENCE_HEADER)
        for reason in divergence:
            lines.append(f"  - {reason}")
        lines.append("")
    for hit in hits:
        terms = ", ".join(f"`{t}`" for t in hit.why)
        lines.append(f"- **`{hit.name}`** — matched {terms}")
    lines.append("")
    lines.append(FOOTER)
    return "\n".join(lines)



def main(argv: list[str] | None = None) -> int:
    """Read ``UserPromptSubmit`` stdin, route, emit hook JSON.
    CLI: ``<corpus-dir> [--operating-point PATH] [--lex-threshold F] [--sem-threshold F]``

    The two threshold flags are OVERRIDES. Omitted, the calibrated library defaults
    apply; passing one pins that admission floor for this invocation.
    """
    if argv is None:
        argv = sys.argv[1:]

    # The delivered path starts HERE, not at route(). This process pays imports,
    # model load and index build every turn; timing only the inner call would
    # describe a fraction of a percent of what the turn costs.
    delivered = timer()
    install_env_sink()

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_dir")
    # Default None, NOT a literal. These were hardcoded 0.15 / 0.55 — the values
    # `build_index` shipped through 0.1.2 — so when 0.1.3 recalibrated the library
    # defaults to 0.10 / 0.25, the calibration reached every caller EXCEPT the one
    # deployment that exists. The adapter silently kept routing at the old operating
    # point (recall 0.185) while the library, its tests and its eval harness all
    # measured 0.648. One default, one place.
    parser.add_argument(
        "--operating-point",
        default=None,
        help="path to a JSON operating point from `route_eval calibrate`; "
             "omitted, the shipped reference-corpus defaults apply",
    )
    parser.add_argument("--lex-threshold", type=float, default=None)
    parser.add_argument("--sem-threshold", type=float, default=None)
    parsed = parser.parse_args(argv)

    overrides = {
        k: v
        for k, v in (
            ("lex_threshold", parsed.lex_threshold),
            ("sem_threshold", parsed.sem_threshold),
        )
        if v is not None
    }

    def _fail(kind: str, exc: object = None) -> int:
        """Record a classified failure and exit clean.

        Every early return in this function goes through here. Before, each of
        them was a bare `return 0` — so a crashed router and a quiet one were
        the same observation, which is the defect this wiring closes.
        """
        if exc is not None:
            logger.warning("%s: %s", kind, exc)
        delivered.__exit__()
        emit(
            RouteEvent(
                outcome="failed",
                failure=kind,  # type: ignore[arg-type]
                adapter=ADAPTER_NAME,
                elapsed_ms=delivered.ms,
                operating_point=overrides,
            )
        )
        return 0

    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        return _fail("input", exc)

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or len(prompt.strip()) < MIN_PROMPT_CHARS:
        # Not an error — a prompt too short to route is a legitimate skip. It is
        # still recorded, as an abstention, so the denominator stays whole.
        delivered.__exit__()
        emit(
            RouteEvent(
                outcome="abstained",
                adapter=ADAPTER_NAME,
                elapsed_ms=delivered.ms,
                operating_point=overrides,
            )
        )
        return 0

    try:
        corpus = PolicyDirCorpus(parsed.corpus_dir)
    except Exception as exc:
        return _fail("corpus_load", exc)

    try:
        point = (
            load_operating_point(parsed.operating_point) if parsed.operating_point else None
        )
    except Exception as exc:
        # NOT a soft fallback to the shipped defaults: a caller who passed a path
        # meant to use it, and silently routing on someone else's numbers is the
        # failure the operating point exists to prevent.
        return _fail("corpus_load", exc)

    try:
        index = build_index(corpus, operating_point=point, **overrides)
    except Exception as exc:
        return _fail("index_build", exc)

    try:
        inner = timer()
        hits = route(prompt, index)
        inner.__exit__()
    except Exception as exc:
        return _fail("route", exc)

    degraded = not semantic_ready()

    if not hits:
        delivered.__exit__()
        emit(
            RouteEvent(
                outcome="abstained",
                corpus_size=index.doc_count,
                degraded=degraded,
                adapter=ADAPTER_NAME,
                elapsed_ms=delivered.ms,
                route_ms=inner.ms,
                operating_point=overrides,
            )
        )
        return 0

    try:
        rendered = _render(hits, degraded=degraded, divergence=index.divergence())
    except Exception as exc:
        return _fail("render", exc)

    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": rendered,
            }
        },
        sys.stdout,
    )
    delivered.__exit__()
    emit(
        RouteEvent(
            outcome="routed",
            returned=tuple(h.name for h in hits),
            corpus_size=index.doc_count,
            degraded=degraded,
            adapter=ADAPTER_NAME,
            elapsed_ms=delivered.ms,
            route_ms=inner.ms,
            operating_point=overrides,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
