"""The operating point belongs to a corpus.

`LEX_FLOOR`, `SEM_FLOOR`, `RRF_K` and `TOP_K` were module constants measured
against one 15-document corpus and shipped to every adopter. They are **not**
universal properties of the algorithm: they are the point where that corpus's
score distribution separated signal from noise. An adopter with 300 documents,
different writing conventions and a different trigger-line style was getting a
number nobody measured for them, with nothing in the package saying so.

This module makes three things true instead:

- **The shipped defaults carry their provenance.** :data:`SHIPPED` names the
  corpus, its size, the embedding model and the date — so a default cannot be
  read as a tuned universal.
- **Calibration is an operation, not a release-time constant.**
  ``python -m oiax.eval.route_eval calibrate`` runs the same sweep that
  produced :data:`SHIPPED` against the adopter's own corpus and labels, and
  emits an :class:`OperatingPoint` they can load.
- **The package notices when it is far from its calibration.**
  :func:`divergence` compares the running corpus against the values the
  operating point was measured under, and the delivery surfaces render the
  result where the reader sees it — the same reasoning that put the
  lexical-only notice in the context paragraph rather than in a log. The layer
  is running, and its numbers do not mean what its documentation says.

**One default, one place.** :data:`SHIPPED` is the single definition; the
module-level constants in :mod:`oiax.router` read from it. Defining the numbers
twice is the defect that let a recalibration reach the library, its tests and
its eval harness while missing the one deployment that existed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "OperatingPoint",
    "SHIPPED",
    "CorpusStats",
    "divergence",
    "load_operating_point",
    "save_operating_point",
]


@dataclass(frozen=True)
class OperatingPoint:
    """A selection configuration together with what it was measured on.

    The provenance fields are not decoration. A configuration without them
    cannot be checked against the corpus it is running on, which is the whole
    mechanism :func:`divergence` implements.
    """

    lex_floor: float
    sem_floor: float
    rrf_k: int
    top_k: int

    # provenance — what this was measured against
    corpus_id: str = ""
    corpus_size: int = 0
    corpus_separability: float | None = None  # mean pairwise cosine spread
    model_id: str = ""
    measured: str = ""  # ISO date
    metrics: dict[str, Any] = field(default_factory=dict)

    # arms-record identity
    arm_id: str = ""
    superseded_id: str = ""

    def selection_kwargs(self) -> dict[str, Any]:
        """The four numbers, as ``build_index`` keyword arguments."""
        return {
            "lex_threshold": self.lex_floor,
            "sem_threshold": self.sem_floor,
            "rrf_k": self.rrf_k,
            "top_k": self.top_k,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatingPoint:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            # Loud, because a typo'd key would otherwise silently leave the
            # shipped default in place while the caller believed they had
            # loaded their own calibration.
            raise ValueError(f"unknown operating-point fields: {sorted(unknown)}")
        missing = {"lex_floor", "sem_floor", "rrf_k", "top_k"} - set(data)
        if missing:
            raise ValueError(f"operating point is missing {sorted(missing)}")
        return cls(**data)

    @property
    def has_provenance(self) -> bool:
        return bool(self.corpus_id and self.corpus_size and self.model_id)

    def describe(self) -> str:
        if not self.has_provenance:
            return (
                f"lex={self.lex_floor} sem={self.sem_floor} rrf_k={self.rrf_k} "
                f"top_k={self.top_k} — NO PROVENANCE: nothing records which corpus "
                "these were measured on, so nothing can tell you when they stop applying"
            )
        return (
            f"lex={self.lex_floor} sem={self.sem_floor} rrf_k={self.rrf_k} "
            f"top_k={self.top_k} — measured {self.measured} on {self.corpus_id} "
            f"({self.corpus_size} documents) under {self.model_id}"
        )


# ── the shipped default, with its provenance ────────────────────────────────
#
# Calibrated 2026-08-03 against `eval/corpora/reference-policies` (15 documents)
# and `reference_labelled.jsonl` (52 prompts, 3 negatives). The full sweep,
# including the rules that lost, is in `eval/corpora/README.md`.
#
# Chosen over lex=0.05 (recall 0.704, F1 0.623), which scored marginally better
# on recall and WORSE on false alarms: it routed a policy for 1 of the 3
# negative prompts, where 0.10 routes nothing for any of them.
#
# THIS IS 15 DOCUMENTS' ANSWER. It is the honest starting point for a corpus
# nobody has calibrated, and it is not a claim about yours.
SHIPPED = OperatingPoint(
    lex_floor=0.10,
    sem_floor=0.25,
    rrf_k=60,
    top_k=2,
    corpus_id="oiax reference-policies",
    corpus_size=15,
    corpus_separability=0.55,
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    measured="2026-08-03",
    metrics={
        "recall@2": 0.648,
        "precision": 0.603,
        "f1": 0.625,
        "top1_accuracy": 0.673,
        "false_alarm_rate": 0.0,
        "labelled_prompts": 52,
        "negatives": 3,
    },
    arm_id="arm-20260803-rrf060",
    superseded_id="arm-20260731-default",
)


# ── divergence ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CorpusStats:
    """What the running corpus looks like, measured at index build."""

    size: int
    separability: float | None = None
    model_id: str = ""


#: A corpus this many times larger or smaller than the calibration corpus is
#: far enough that the floors are unlikely to transfer. Deliberately crude and
#: wide: a divergence detector that fires constantly gets ignored exactly like
#: a false-positive route does, and the cost of a missed warning here is a
#: suboptimal operating point, not a wrong answer.
SIZE_RATIO_LIMIT = 4.0

#: Separability below this is the failure mode where every document embeds to
#: nearly the same vector — the first eval corpus measured ~0.04 against 0.55
#: on the one that replaced it, and any threshold measured on it was measured
#: on noise.
SEPARABILITY_FLOOR = 0.15


def divergence(stats: CorpusStats, point: OperatingPoint = SHIPPED) -> list[str]:
    """Reasons the running corpus is unlike the one ``point`` was measured on.

    Returns an empty list when nothing is out of range. Each entry is a
    complete sentence, because it is rendered to a reader rather than parsed.

    This never raises and never blocks. An unusual corpus is not an error — the
    layer still routes, and the caller is told the numbers may not mean what the
    documentation says.
    """
    out: list[str] = []

    if not point.has_provenance:
        out.append(
            "The operating point in force records no corpus, so nothing can "
            "check whether it applies here."
        )
        return out

    if stats.size and point.corpus_size:
        ratio = max(stats.size, point.corpus_size) / min(stats.size, point.corpus_size)
        if ratio >= SIZE_RATIO_LIMIT:
            bigger = "larger" if stats.size > point.corpus_size else "smaller"
            out.append(
                f"This corpus has {stats.size} documents; the operating point was "
                f"calibrated on {point.corpus_size} ({ratio:.0f}x {bigger}). "
                "Near-neighbour pressure rises with corpus size, so the floors "
                "are unlikely to transfer — recalibrate."
            )

    if stats.separability is not None:
        if stats.separability < SEPARABILITY_FLOOR:
            out.append(
                f"This corpus separates its own documents at only "
                f"{stats.separability:.2f} (floor {SEPARABILITY_FLOOR}). Documents "
                "that embed to nearly the same vector cannot be separated by any "
                "threshold — the routing surfaces need to differ more, not the "
                "floors."
            )
        elif (
            point.corpus_separability is not None
            and abs(stats.separability - point.corpus_separability) > 0.25
        ):
            out.append(
                f"This corpus separates at {stats.separability:.2f}; the operating "
                f"point was calibrated at {point.corpus_separability:.2f}. The "
                "score distribution here is materially different."
            )

    if stats.model_id and point.model_id and stats.model_id != point.model_id:
        out.append(
            f"Running under {stats.model_id}; the operating point was calibrated "
            f"under {point.model_id}. Cosine distributions are not comparable "
            "between models — these floors were measured on a system that is not "
            "this one."
        )

    return out


# ── persistence ─────────────────────────────────────────────────────────────


def save_operating_point(point: OperatingPoint, path: str | Path) -> None:
    """Write an operating point as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(point.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_operating_point(path: str | Path) -> OperatingPoint:
    """Read an operating point from JSON.

    Raises on a missing file or a malformed one. **Not** a soft fallback to
    :data:`SHIPPED`: a caller who passed a path meant to use it, and silently
    running someone else's numbers instead is precisely the failure this module
    exists to prevent.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no operating point at {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p}: expected a JSON object")
    return OperatingPoint.from_dict(data)
