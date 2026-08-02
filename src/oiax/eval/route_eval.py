"""Measure router miss rate against labelled ground truth.

Usage:
    python -m oiax.eval.route_eval score <corpus-dir> < labelled.jsonl

Takes a JSONL file of labelled examples (one JSON object per line:
``{"prompt": "...", "expected": ["policy-a", "policy-b"]}``) and scores
the router against them. Reports recall and precision.

Judge labels are evidence, not proof. Hand-check a slice before treating
the rate as authoritative. A judge that silently mislabels produces a
confident wrong number, worse than no number.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from oiax import build_index, route
from oiax.corpus import PolicyDirCorpus


@dataclass
class EvalResult:
    """Outcome of scoring one labelled example."""
    prompt: str
    expected: list[str]
    routed: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)

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


def score_labelled(corpus_dir: str, threshold: float = 0.15) -> list[EvalResult]:
    """Score the router against labelled examples from stdin.

    Returns one EvalResult per labelled example.
    """
    corpus = PolicyDirCorpus(corpus_dir)
    index = build_index(corpus)

    results: list[EvalResult] = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        prompt = item.get("prompt", "")
        expected = item.get("expected", [])

        hits = route(prompt, index)
        routed_names = [h.name for h in hits if h.score >= threshold]

        missed = [e for e in expected if e not in routed_names]
        extras = [r for r in routed_names if r not in expected]

        results.append(EvalResult(
            prompt=prompt,
            expected=expected,
            routed=routed_names,
            missed=missed,
            extras=extras,
        ))

    return results


def print_summary(results: list[EvalResult]) -> None:
    """Print recall/precision summary."""
    if not results:
        print("No labelled examples to score.")
        return

    total_expected = sum(len(r.expected) for r in results)
    total_found = sum(
        len(set(r.expected) & set(r.routed)) for r in results
    )
    total_routed = sum(len(r.routed) for r in results)
    total_missed = sum(len(r.missed) for r in results)
    total_extras = sum(len(r.extras) for r in results)

    recall = total_found / total_expected if total_expected else 1.0
    precision = total_found / total_routed if total_routed else 1.0

    print(f"Examples: {len(results)}")
    print(f"Expected (total): {total_expected}")
    print(f"Routed (total): {total_routed}")
    print(f"Found: {total_found}")
    print(f"Missed: {total_missed}")
    print(f"False positives: {total_extras}")
    print(f"Recall: {recall:.3f}")
    print(f"Precision: {precision:.3f}")

    failed = [r for r in results if r.missed]
    if failed:
        print(f"\n{len(failed)} examples with misses:")
        for r in failed[:5]:  # Show first 5
            print(f"  prompt: {r.prompt[:80]}...")
            print(f"    expected: {r.expected}")
            print(f"    missed:   {r.missed}")
            if r.extras:
                print(f"    extras:   {r.extras}")


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] != "score":
        prog = "python -m oiax.eval.route_eval"
        print(f"Usage: {prog} score <corpus-dir> < labelled.jsonl", file=sys.stderr)
        return 1

    corpus_dir = sys.argv[2]
    results = score_labelled(corpus_dir)
    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
