"""Read a telemetry log and say whether the layer is working.

Turns the append-only stream of :class:`oiax.telemetry.RouteEvent` records into
the four questions the emission exists to answer:

1. **Is it failing?** Failure rate, broken down by class. This is the number
   that did not exist before — a crashed router and a quiet one used to be the
   same observation.
2. **Is it degraded?** The share of attempts running lexical-only because the
   embedder did not load. A silently lexical-only router shipped in this
   package for four days.
3. **What does it actually cost?** Delivered latency percentiles, and the warm
   route beside them so the gap between the advertised number and the true one
   is visible rather than quotable in isolation.
4. **Which documents never route?** A document with zero routes over a period
   has a routing surface that is not discriminating. It is reported BY NAME,
   because that list is the actionable half of routing quality.

Run it::

    python -m oiax.eval.telemetry_report events.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TelemetrySummary:
    """Aggregate over a telemetry log."""

    total: int = 0
    routed: int = 0
    abstained: int = 0
    failed: int = 0
    degraded: int = 0
    failures: Counter[str] = field(default_factory=Counter)
    per_document: Counter[str] = field(default_factory=Counter)
    delivered_ms: list[float] = field(default_factory=list)
    route_ms: list[float] = field(default_factory=list)
    corpus_size: int = 0

    @property
    def failure_rate(self) -> float:
        return self.failed / self.total if self.total else 0.0

    @property
    def degraded_rate(self) -> float:
        return self.degraded / self.total if self.total else 0.0

    @property
    def abstention_rate(self) -> float:
        return self.abstained / self.total if self.total else 0.0


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def load_events(stream: Iterable[str]) -> list[dict[str, Any]]:
    """Parse events from JSONL, skipping unparseable lines.

    Skipping is correct HERE and wrong in the eval harnesses: a telemetry log is
    appended to by concurrent short-lived processes, so a torn final line is an
    expected artifact of collection rather than a defect in the data. A skipped
    line costs one sample; refusing to read the file costs the whole surface.
    """
    events: list[dict[str, Any]] = []
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "outcome" in obj:
            events.append(obj)
    return events


def summarize(events: Iterable[dict[str, Any]]) -> TelemetrySummary:
    s = TelemetrySummary()
    for e in events:
        s.total += 1
        outcome = e.get("outcome")
        if outcome == "routed":
            s.routed += 1
            for name in e.get("returned") or []:
                s.per_document[str(name)] += 1
        elif outcome == "abstained":
            s.abstained += 1
        elif outcome == "failed":
            s.failed += 1
            s.failures[str(e.get("failure") or "unclassified")] += 1
        if e.get("degraded"):
            s.degraded += 1
        if isinstance(e.get("elapsed_ms"), (int, float)):
            s.delivered_ms.append(float(e["elapsed_ms"]))
        if isinstance(e.get("route_ms"), (int, float)):
            s.route_ms.append(float(e["route_ms"]))
        if isinstance(e.get("corpus_size"), int):
            s.corpus_size = max(s.corpus_size, e["corpus_size"])
    return s


def format_summary(s: TelemetrySummary, known_documents: set[str] | None = None) -> str:
    lines = [f"oiax telemetry — {s.total} route attempt(s)", ""]
    if not s.total:
        # An empty log is NOT a healthy one. A component emitting nothing is
        # unobserved, and the report says which of the two it is looking at.
        lines.append(
            "No events. That is not the same as healthy: either nothing routed, "
            "or the sink was never installed (set OIAX_TELEMETRY_PATH)."
        )
        return "\n".join(lines)

    lines.append(f"  routed     {s.routed:>6}  ({s.routed / s.total:.1%})")
    lines.append(f"  abstained  {s.abstained:>6}  ({s.abstention_rate:.1%})")
    lines.append(f"  failed     {s.failed:>6}  ({s.failure_rate:.1%})")
    if s.failures:
        for kind, n in s.failures.most_common():
            lines.append(f"      {kind:<14} {n}")
    lines.append(f"  degraded   {s.degraded:>6}  ({s.degraded_rate:.1%})  lexical-only")
    lines.append("")

    p50, p99 = _pct(s.delivered_ms, 0.50), _pct(s.delivered_ms, 0.99)
    r50 = _pct(s.route_ms, 0.50)
    if p50 is not None:
        lines.append(f"  delivered  p50 {p50:.0f} ms   p99 {p99:.0f} ms")
    if r50 is not None:
        lines.append(f"  warm route p50 {r50:.0f} ms")
        if p50 is not None and r50 > 0:
            # Printed as a ratio because the two numbers are routinely quoted
            # apart, and the whole point is that one is a small fraction of the
            # other.
            lines.append(f"             warm route is {r50 / p50:.1%} of delivered")
    lines.append("")

    if known_documents:
        never = sorted(known_documents - set(s.per_document))
        seen, total = len(s.per_document), len(known_documents)
        lines.append(f"  documents routed at least once: {seen}/{total}")
        if never:
            lines.append("  NEVER ROUTED — these routing surfaces are not discriminating:")
            for name in never:
                lines.append(f"      {name}")
    elif s.per_document:
        lines.append("  routes per document (top 10):")
        for name, n in s.per_document.most_common(10):
            lines.append(f"      {name:<40} {n}")
        lines.append(
            "  (pass the corpus directory to list documents that NEVER routed — "
            "the actionable half)"
        )

    lines.append("")
    lines.append(
        "Bound: this measures DELIVERY. A routed document reached the context "
        "window; nothing here shows a rule was read or followed."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m oiax.eval.telemetry_report")
    parser.add_argument("events", help="path to the JSONL telemetry log, or - for stdin")
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="corpus directory; enables the NEVER-ROUTED list, which is the actionable half",
    )
    args = parser.parse_args(argv)

    if args.events == "-":
        events = load_events(sys.stdin)
    else:
        path = Path(args.events)
        if not path.exists():
            print(f"no telemetry log at {path}", file=sys.stderr)
            return 2
        events = load_events(path.read_text(encoding="utf-8").splitlines())

    known: set[str] | None = None
    if args.corpus_dir:
        from oiax.corpus import PolicyDirCorpus

        known = {d.name for d in PolicyDirCorpus(args.corpus_dir).documents()}

    print(format_summary(summarize(events), known_documents=known))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
