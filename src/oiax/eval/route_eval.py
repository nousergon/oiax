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
from oiax.calibration import SHIPPED, OperatingPoint, save_operating_point
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
    #: Documents from the same family as the gold that are WRONG for this
    #: prompt. Empty when the example carries no sibling label — which is most
    #: of them, and why HSR has its own denominator.
    sibling: list[str] = field(default_factory=list)

    @property
    def has_sibling_label(self) -> bool:
        return bool(self.sibling)

    @property
    def harmful(self) -> bool:
        """True when a labelled wrong sibling was returned.

        This is the failure a relevance-only scorer creates while improving
        recall: both members of a family score highly, both land in the top-k,
        and the reader is handed a plausible wrong answer with convincing
        matched terms. It is not caught by recall, by precision, or by the
        false-alarm rate — the sibling IS relevant-looking, and the prompt IS
        governed by something.
        """
        return any(s in self.routed for s in self.sibling)

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
    n_sibling: int = 0
    hsr: float = 0.0


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
                sibling=item.get("sibling") or [],
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
    # HSR has its OWN denominator — only examples carrying a sibling label. A
    # rate computed over every example would fall simply by adding unlabelled
    # prompts, which is improvement by dilution.
    sibling_labelled = [r for r in results if r.has_sibling_label]
    hsr = (
        sum(1 for r in sibling_labelled if r.harmful) / len(sibling_labelled)
        if sibling_labelled
        else 0.0
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
        n_sibling=len(sibling_labelled),
        hsr=hsr,
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
    if m.n_sibling:
        print(
            f"HSR@{TOP_K}:           {m.hsr:.3f}"
            f"  (share of {m.n_sibling} sibling-labelled prompts whose WRONG"
            f" same-family document was returned)"
        )
    else:
        # Silence here would read as HSR 0. It is not measured.
        print(
            "HSR@k:           not measured — no example carries a `sibling` label. "
            "Recall, precision and the false-alarm rate are all blind to a "
            "same-family wrong answer."
        )

    harmful = [r for r in results if r.harmful]
    if harmful:
        print(f"\nHARMFUL SIBLINGS ({len(harmful)}):")
        for r in harmful:
            got = [s for s in r.sibling if s in r.routed]
            print(f"  {r.prompt[:66]}")
            print(f"    wanted {r.expected}, also returned {got}")

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


# Grid searched by `calibrate`. Same rules the shipped default was chosen from,
# exposed rather than performed once by the maintainer and frozen — that freezing
# is what handed every adopter a 15-document corpus's answer.
_LEX_GRID = (0.05, 0.10, 0.15, 0.20)
_SEM_GRID = (0.20, 0.25, 0.30, 0.35)


def calibrate(
    corpus_dir: str,
    items: list[dict[str, Any]],
    *,
    out: str | None = None,
    corpus_id: str | None = None,
) -> int:
    """Find this corpus's operating point and record what it was measured on.

    Selection rule, in order and stated because a grid search with an unstated
    objective is a number nobody can argue with:

    1. **Zero false alarms is a hard gate**, not a tiebreak. A configuration that
       routes a negative is excluded no matter what it scores — a false positive
       gets the layer tuned out permanently, and a tuned-out layer costs tokens
       forever while changing nothing.
    2. Among survivors, highest F1.
    3. Ties broken toward the QUIETER point (higher floors), for the same reason
       as (1).

    Prints the full grid, including the rows that lost, because an
    operating-point table showing only the winner hides what it was chosen over
    and the next person re-runs the sweep to find out.
    """
    from oiax.corpus import PolicyDirCorpus
    from oiax.router import _MODEL_NAME

    index_probe = build_index(PolicyDirCorpus(corpus_dir))
    if index_probe.doc_count == 0:
        print(f"no documents in {corpus_dir}", file=sys.stderr)
        return 1

    rows: list[tuple[float, float, Metrics]] = []
    for lex in _LEX_GRID:
        for sem in _SEM_GRID:
            m = aggregate(evaluate(corpus_dir, items, lex_threshold=lex, sem_threshold=sem))
            rows.append((lex, sem, m))

    print(f"{'lex':>5} {'sem':>5} {'recall':>7} {'prec':>7} {'F1':>7} {'top1':>7} {'falarm':>7}")
    for lex, sem, m in rows:
        print(
            f"{lex:>5.2f} {sem:>5.2f} {m.recall:>7.3f} {m.precision:>7.3f} "
            f"{m.f1:>7.3f} {m.top1_accuracy:>7.3f} {m.false_alarm_rate:>7.3f}"
        )

    clean = [r for r in rows if r[2].false_alarm_rate == 0.0]
    if not clean:
        print(
            "\nNO CONFIGURATION CLEARS THE ZERO-FALSE-ALARM BAR on this corpus. "
            "That is a finding, not a failure to calibrate: either the negatives "
            "are mislabelled, or the routing surfaces overlap too much to "
            "separate. Fix the corpus, not the floors.",
            file=sys.stderr,
        )
        return 1

    lex, sem, best = max(clean, key=lambda r: (r[2].f1, r[0], r[1]))
    point = OperatingPoint(
        lex_floor=lex,
        sem_floor=sem,
        rrf_k=SHIPPED.rrf_k,
        top_k=SHIPPED.top_k,
        corpus_id=corpus_id or corpus_dir,
        corpus_size=index_probe.doc_count,
        corpus_separability=index_probe.corpus_separability,
        model_id=_MODEL_NAME,
        measured="",  # stamped by the caller; this module has no clock by design
        metrics={
            "recall@2": round(best.recall, 3),
            "precision": round(best.precision, 3),
            "f1": round(best.f1, 3),
            "top1_accuracy": round(best.top1_accuracy, 3),
            "false_alarm_rate": best.false_alarm_rate,
            "labelled_prompts": best.n,
            "negatives": best.n_negative,
        },
    )

    print("\nWINNER (zero false alarms, then highest F1, then quieter):")
    print("  " + point.describe())
    if not best.n_negative:
        print(
            "\n  WARNING: this labelled set contains NO negative prompts, so the "
            "zero-false-alarm gate passed vacuously. The winning floors are the "
            "highest-F1 point and nothing has tested whether they stay quiet.",
            file=sys.stderr,
        )

    if out:
        save_operating_point(point, out)
        print(f"\nwrote {out}")
    else:
        print("\n(pass --out PATH to write this as a loadable operating point)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m oiax.eval.route_eval")
    parser.add_argument("command", choices=["score", "sweep", "calibrate"])
    parser.add_argument("corpus_dir")
    parser.add_argument(
        "--out",
        default=None,
        help="calibrate only: write the winning operating point here as JSON",
    )
    parser.add_argument(
        "--corpus-id",
        default=None,
        help="calibrate only: a name for this corpus, recorded as provenance",
    )
    parsed = parser.parse_args(sys.argv[1:] if argv is None else argv)

    items = load_labelled()
    if not items:
        print("No labelled examples on stdin.", file=sys.stderr)
        return 1

    if parsed.command == "sweep":
        print_sweep(parsed.corpus_dir, items)
    elif parsed.command == "calibrate":
        return calibrate(parsed.corpus_dir, items, out=parsed.out, corpus_id=parsed.corpus_id)
    else:
        print_summary(evaluate(parsed.corpus_dir, items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
