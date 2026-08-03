"""Measure routing quality against labelled ground truth.

Usage:
    python -m oiax.eval.route_eval score <corpus-dir> < labelled.jsonl
    python -m oiax.eval.route_eval sweep <corpus-dir> < labelled.jsonl

``score`` reports the shipped configuration. ``sweep`` re-runs the calibration
that produced the shipped defaults across a grid of admission floors, so the
numbers in ``corpora/README.md`` are reproducible rather than asserted.

Labelled input is JSONL, one object per line::

    {"prompt": "...", "expected": ["policy-a", "policy-b"]}

An example with ``"expected": []`` is a NEGATIVE — a prompt no policy should
govern. Negatives are the only way to measure the false-alarm rate, and a suite
without them can be gamed to recall 1.0 by routing everything.

**Metrics, and why these ones.**

- ``recall@k`` — of all expected labels, how many appear in the (at most ``k``)
  names shown. This is the metric that matters most: a policy that never
  surfaces is silently absent, and the reader has no way to know.
- ``top-1 accuracy`` — how often the FIRST name shown is a correct one.
- ``precision`` — correct names over names shown. Read it against the top_k cap:
  with ``top_k=2`` and one expected label, precision is capped at 0.5 for that
  prompt no matter how good the router is, so precision below 0.5 is the signal
  to watch, not precision below 1.0.
- ``false-alarm rate`` — share of NEGATIVE prompts that routed anything.

Judge labels are evidence, not proof. Hand-check a slice before treating any
rate as authoritative. A judge that silently mislabels produces a confident
wrong number, which is worse than no number.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from oiax import build_index, route
from oiax.corpus import PolicyDirCorpus
from oiax.router import LEX_FLOOR, SEM_FLOOR, TOP_K


@dataclass
class EvalResult:
    """Outcome of scoring one labelled example."""

    prompt: str
    expected: list[str]
    routed: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)

    @property
    def is_negative(self) -> bool:
        """True when no policy should govern this prompt."""
        return not self.expected

    @property
    def top1_correct(self) -> bool:
        """True when the first name shown is one of the expected labels."""
        return bool(self.routed) and self.routed[0] in self.expected

    @property
    def recall(self) -> float:
        if not self.expected:
            return 1.0
        found = len(set(self.expected) & set(self.routed))
        return found / len(self.expected)

    @property
    def precision(self) -> float:
        if not self.routed:
            return 1.0  # nothing routed = no false positives
        correct = len(set(self.expected) & set(self.routed))
        return correct / len(self.routed)


@dataclass
class Metrics:
    """Aggregate scores over a labelled set."""

    n: int
    n_negative: int
    expected: int
    routed: int
    found: int
    recall: float
    precision: float
    f1: float
    top1_accuracy: float
    false_alarm_rate: float


def load_labelled(stream: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Parse labelled JSONL from a stream (default stdin), skipping bad lines."""
    src = stream if stream is not None else sys.stdin
    items: list[dict[str, Any]] = []
    for line in src:
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def evaluate(
    corpus_dir: str, items: list[dict[str, Any]], **index_kwargs: Any
) -> list[EvalResult]:
    """Route every labelled prompt and return per-example results."""
    index = build_index(PolicyDirCorpus(corpus_dir), **index_kwargs)

    results: list[EvalResult] = []
    for item in items:
        prompt = item.get("prompt", "")
        expected = item.get("expected", [])
        routed_names = [h.name for h in route(prompt, index)]
        results.append(
            EvalResult(
                prompt=prompt,
                expected=expected,
                routed=routed_names,
                missed=[e for e in expected if e not in routed_names],
                extras=[r for r in routed_names if r not in expected],
            )
        )
    return results


def aggregate(results: list[EvalResult]) -> Metrics:
    """Aggregate per-example results into the reported metrics."""
    expected = sum(len(r.expected) for r in results)
    routed = sum(len(r.routed) for r in results)
    found = sum(len(set(r.expected) & set(r.routed)) for r in results)
    positives = [r for r in results if not r.is_negative]
    negatives = [r for r in results if r.is_negative]

    recall = found / expected if expected else 1.0
    precision = found / routed if routed else 1.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    top1 = (
        sum(1 for r in positives if r.top1_correct) / len(positives) if positives else 1.0
    )
    false_alarm = (
        sum(1 for r in negatives if r.routed) / len(negatives) if negatives else 0.0
    )
    return Metrics(
        n=len(results),
        n_negative=len(negatives),
        expected=expected,
        routed=routed,
        found=found,
        recall=recall,
        precision=precision,
        f1=f1,
        top1_accuracy=top1,
        false_alarm_rate=false_alarm,
    )


# Backwards-compatible entry point kept for callers of the 0.1.x API.
def score_labelled(corpus_dir: str, threshold: float | None = None) -> list[EvalResult]:
    """Score labelled examples read from stdin against ``corpus_dir``.

    ``threshold`` is accepted and ignored: selection is reciprocal-rank fusion
    over admission floors, so a single post-hoc score cutoff no longer describes
    the selection rule. Pass floors to ``evaluate`` instead.
    """
    return evaluate(corpus_dir, load_labelled())


def print_summary(results: list[EvalResult]) -> None:
    """Print the metric block plus every example that missed."""
    if not results:
        print("No labelled examples to score.")
        return

    m = aggregate(results)
    print(f"Examples: {m.n} ({m.n_negative} negative)")
    print(f"Expected labels: {m.expected}   Routed: {m.routed}   Correct: {m.found}")
    print(f"Recall@{TOP_K}:        {m.recall:.3f}")
    print(
        f"Precision:       {m.precision:.3f}"
        f"  (cap is 0.5 per single-label prompt at top_k={TOP_K})"
    )
    print(f"F1:              {m.f1:.3f}")
    print(f"Top-1 accuracy:  {m.top1_accuracy:.3f}")
    print(
        f"False alarms:    {m.false_alarm_rate:.3f}"
        f"  (share of negative prompts that routed anything)"
    )

    failed = [r for r in results if r.missed]
    if failed:
        print(f"\n{len(failed)} examples with misses:")
        for r in failed:
            print(f"  prompt: {r.prompt[:88]}")
            print(f"    expected: {r.expected}")
            print(f"    routed:   {r.routed}")


SWEEP_LEX = (0.03, 0.05, 0.08, 0.10, 0.15)
SWEEP_SEM = (0.20, 0.25, 0.30, 0.35, 0.45, 0.55)


def print_sweep(corpus_dir: str, items: list[dict[str, Any]]) -> None:
    """Grid the admission floors and print one row per configuration.

    Reproduces the calibration behind the shipped defaults. The shipped point is
    marked, so a reader can see both what was chosen and what it was chosen over
    — a default with no visible runners-up is indistinguishable from a guess.
    """
    print(f"{'lex':>6} {'sem':>6} {'recall':>8} {'prec':>7} {'F1':>7} {'top-1':>7} {'falarm':>7}")
    for lex in SWEEP_LEX:
        for sem in SWEEP_SEM:
            m = aggregate(evaluate(corpus_dir, items, lex_threshold=lex, sem_threshold=sem))
            shipped = " <- shipped" if (lex, sem) == (LEX_FLOOR, SEM_FLOOR) else ""
            print(
                f"{lex:6.2f} {sem:6.2f} {m.recall:8.3f} {m.precision:7.3f} "
                f"{m.f1:7.3f} {m.top1_accuracy:7.3f} {m.false_alarm_rate:7.3f}{shipped}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m oiax.eval.route_eval")
    parser.add_argument("command", choices=["score", "sweep"])
    parser.add_argument("corpus_dir")
    parsed = parser.parse_args(sys.argv[1:] if argv is None else argv)

    items = load_labelled()
    if not items:
        print("No labelled examples on stdin.", file=sys.stderr)
        return 1

    if parsed.command == "sweep":
        print_sweep(parsed.corpus_dir, items)
    else:
        print_summary(evaluate(parsed.corpus_dir, items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
