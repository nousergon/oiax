"""Scoring oiax against corpora its authors did not write.

Every quality number this package published before now came from
``corpora/reference-policies``: 15 documents, 52 prompts, one author — the
author of the design those numbers are cited to support. That is a regression
surface. It cannot establish that a *design* is better than the alternatives,
because it shares an author with the design.

This module maps public benchmarks onto the oiax interfaces so a score lands on
somebody else's comparison table instead of in a vacuum.

**Nothing is vendored.** Benchmark corpora are third-party artifacts under
their own licences and are hundreds of megabytes. :func:`fetch` downloads them
to a cache directory the caller names; the repo carries the adapter and the
published numbers, never the data.

## SkillRet

`ThakiCloud/SKILLRET <https://huggingface.co/datasets/ThakiCloud/SKILLRET>`_ —
a skill-retrieval benchmark over real skills scraped from public GitHub
repositories, in classic IR shape: a corpus, queries, and qrels.

The **substitution this adapter makes, stated because it is material**: oiax
scores an authored *routing surface* — a one-line statement of when a document
applies — and SkillRet skills have no such field. Their ``description`` is used
in its place. That is the closest analogue and it is not the same thing: a
description says what a skill *is*, a routing surface says when it *applies*.
oiax's design assumes the second, so this benchmark scores it on the first, and
a lower number here is partly a measure of that gap rather than of the router.
Reported as measured; not adjusted for.

**Documents are keyed by skill id, never by name.** 6,006 skills carry only
5,801 distinct names, and the qrels key on id. Keying by name would silently
merge 205 distinct skills into their neighbours — and those collisions are
themselves the same-family population that harmful-sibling work is about.
"""

from __future__ import annotations

import json
import random
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oiax.corpus import Document

__all__ = [
    "SKILLRET_FILES",
    "SkillRetCorpus",
    "fetch",
    "load_skillret_labelled",
    "subsample",
]

_HF_BASE = "https://huggingface.co/datasets/ThakiCloud/SKILLRET/resolve/main"

#: The three files the test split needs. The full ``data/skills.jsonl`` (335 MB)
#: is the train+test corpus and is deliberately not used: scoring against the
#: split the benchmark designates for testing is the only honest option.
SKILLRET_FILES: dict[str, str] = {
    "skills": "data/skills/test.jsonl",
    "queries": "data/queries/test.jsonl",
    "qrels": "data/qrels/test.jsonl",
}


def fetch(cache_dir: str | Path, *, files: dict[str, str] | None = None) -> dict[str, Path]:
    """Download the benchmark into ``cache_dir``. Returns local paths by key.

    Skips a file that already exists — these are ~112 MB and immutable. A
    partial download from an interrupted run would be silently reused, so
    delete the cache directory rather than trusting a suspicious file.

    Raises on any failure. A benchmark that half-downloaded and scored anyway
    would produce a number against a corpus nobody can reconstruct, which is
    worse than no number.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for key, rel in (files or SKILLRET_FILES).items():
        dest = cache / Path(rel).name if key != "skills" else cache / "skills.jsonl"
        if key == "queries":
            dest = cache / "queries.jsonl"
        elif key == "qrels":
            dest = cache / "qrels.jsonl"
        if not dest.exists() or dest.stat().st_size == 0:
            urllib.request.urlretrieve(f"{_HF_BASE}/{rel}", dest)  # noqa: S310
        out[key] = dest
    return out


@dataclass
class SkillRetCorpus:
    """SkillRet skills as an oiax :class:`~oiax.corpus.Corpus`.

    ``limit`` truncates the corpus, for the corpus-size curve. Truncation is
    deterministic when ``seed`` is given, because a size curve measured on a
    different random subset at each point is measuring the subset.
    """

    path: str | Path
    limit: int | None = None
    seed: int | None = None
    _ids: set[str] | None = None

    def __post_init__(self) -> None:
        if self.limit is not None:
            all_ids = [json.loads(line)["id"] for line in self._lines()]
            rng = random.Random(self.seed)
            rng.shuffle(all_ids)
            self._ids = set(all_ids[: self.limit])

    def _lines(self) -> Iterator[str]:
        with Path(self.path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line

    def ids(self) -> set[str]:
        """The skill ids this corpus contains — the denominator for filtering
        queries whose gold document was truncated away."""
        if self._ids is not None:
            return set(self._ids)
        return {json.loads(line)["id"] for line in self._lines()}

    def documents(self) -> Iterator[Document]:
        for line in self._lines():
            rec = json.loads(line)
            if self._ids is not None and rec["id"] not in self._ids:
                continue
            description = (rec.get("description") or "").strip()
            if not description:
                # A skill with no description has no routing surface at all.
                # Skipping it silently would shrink the corpus without shrinking
                # the qrels, inflating recall — so this is a hard skip and the
                # caller's `ids()` reflects it only if it also parses.
                continue
            yield Document(
                name=rec["id"],  # id, NEVER name: 6,006 skills, 5,801 names
                trigger_line=description,
                body=rec.get("body") or rec.get("skill_md") or description,
            )


def load_skillret_labelled(
    queries_path: str | Path,
    qrels_path: str | Path,
    *,
    restrict_to: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Queries + qrels as oiax labelled items: ``{"prompt", "expected"}``.

    ``restrict_to`` drops queries whose gold skills are not all present — used
    by the corpus-size curve, where truncating the corpus would otherwise leave
    queries that are **unanswerable by construction** and depress recall for a
    reason that has nothing to do with the router.

    Note there are **no negatives**: every SkillRet query has at least one gold
    skill. So a false-alarm rate is not measurable on this benchmark, and the
    zero-false-alarm bar oiax calibrates against cannot be checked here. That is
    a property of the benchmark, and any result quoted from it carries the gap.
    """
    gold: dict[str, list[str]] = {}
    with Path(qrels_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if int(rec.get("relevance", 0)) <= 0:
                continue
            gold.setdefault(str(rec["query_id"]), []).append(str(rec["skill_id"]))

    items: list[dict[str, Any]] = []
    with Path(queries_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            expected = gold.get(str(rec["id"]), [])
            if not expected:
                continue
            if restrict_to is not None:
                if not all(e in restrict_to for e in expected):
                    continue
            prompt = (rec.get("query") or "").strip()
            if prompt:
                items.append({"prompt": prompt, "expected": expected})
    return items


def subsample(items: list[dict[str, Any]], n: int, *, seed: int = 0) -> list[dict[str, Any]]:
    """A deterministic subset, for keeping a long run bounded.

    Deterministic on purpose: a benchmark number that moves between runs because
    the sample moved cannot be compared to itself, let alone to a baseline.
    """
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n)
