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

## SRA-Bench

`WeihangSu/SRA-Bench <https://huggingface.co/datasets/WeihangSu/SRA-Bench>`_
(MIT, arXiv:2604.24594) — a skill-retrieval-augmentation benchmark: a shared
26,262-skill pool plus six task-domain instance files, each instance a real
question with the gold skill id(s) needed to answer it.

**Second out-of-org corpus, at a different order of magnitude from SkillRet
(policy §7.2 items 6-8).** SkillRet is 6,006 documents; SkillRet alone cannot
show whether a design holds at a different scale, only whether it holds on one
large corpus. This adapter scores the **MedCalc-Bench domain slice**: the
55 skills whose id starts with ``medcalcbench_`` (medical risk-score and
dosage calculators, e.g. "CHA2DS2-VASc Score", "Body Mass Index (BMI)") and
the 1,100 instances that name one of them as gold. 55 documents against
SkillRet's 6,006 is a ~109x gap — just over two orders of magnitude
(log10 ≈ 2.04) — and lands squarely in the "tens" the policy calls for.

**The corpus is a documented subset of a larger public release, not an
independent download.** SRA-Bench ships one corpus file shared by all six
domains; there is no per-domain corpus artifact upstream. Restricting to the
``medcalcbench_`` prefix is this adapter's choice, stated because it is
material: the 25,626 non-medcalcbench entries in the shared file are never
scored against and never enter the index. Every kept document and every kept
query is verbatim upstream data — nothing here is authored or relabelled.

**The substitution this adapter makes, same shape as SkillRet's:** SRA-Bench
skills carry a ``description`` field ("Compute BMI from weight (kg) and height
(cm)."), used as the routing surface in place of an authored trigger line. It
reads closer to "when this applies" than SkillRet's free-text description
does, because SRA-Bench skills are single-purpose calculators rather than
general-purpose tools — but it is still a description standing in for a
trigger line, and the gap is reported as measured, not adjusted for.

**No negatives, same bound as SkillRet.** Every MedCalc-Bench instance names
at least one gold skill, so the false-alarm rate — the number oiax actually
calibrates against — is not measurable on this benchmark either.

**Queries are real clinical case reports, not engineering prose.** Each
question is a multi-paragraph patient note (165-11,297 characters, mean
~3,000) drawn from published case studies, structurally unlike both
SkillRet's short GitHub-derived queries and the reference corpus's
engineer-voice prompts — a third register, for whatever that is worth against
a router tuned on the other two.
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
    "SRABENCH_DOMAIN",
    "SRABENCH_FILES",
    "SRABenchCorpus",
    "SkillRetCorpus",
    "family_confusion",
    "fetch",
    "fetch_srabench",
    "load_skillret_labelled",
    "load_srabench_labelled",
    "subsample",
]

_HF_BASE = "https://huggingface.co/datasets/ThakiCloud/SKILLRET/resolve/main"
_SRABENCH_HF_BASE = "https://huggingface.co/datasets/WeihangSu/SRA-Bench/resolve/main"

#: The domain slice this adapter scores — tens-of-documents end of the size
#: curve. Chosen over champ (89 skills) and bigcodebench (139 skills) because
#: it clears a full two orders of magnitude against SkillRet's 6,006
#: (log10(6006/55) ≈ 2.04) and every instance names exactly one gold skill,
#: the same single-label IR shape SkillRet has.
SRABENCH_DOMAIN = "medcalcbench"

#: The two files this adapter needs. ``corpus.json`` (~232 MB) is the full
#: 26,262-skill pool shared by all six SRA-Bench domains — there is no
#: per-domain corpus artifact upstream, so the whole file is fetched and
#: :class:`SRABenchCorpus` filters it down at read time.
SRABENCH_FILES: dict[str, str] = {
    "corpus": "corpus/corpus.json",
    "instances": f"instances/{SRABENCH_DOMAIN}.json",
}

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


def fetch_srabench(
    cache_dir: str | Path, *, files: dict[str, str] | None = None
) -> dict[str, Path]:
    """Download SRA-Bench's shared corpus and one domain's instances.

    Same contract as :func:`fetch`: skips a file that already exists (the
    corpus is ~232 MB and immutable), raises on any failure rather than
    scoring against a partial download.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for key, rel in (files or SRABENCH_FILES).items():
        dest = cache / ("corpus.json" if key == "corpus" else "instances.json")
        if not dest.exists() or dest.stat().st_size == 0:
            urllib.request.urlretrieve(f"{_SRABENCH_HF_BASE}/{rel}", dest)  # noqa: S310
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

    def fingerprint(self) -> str:
        """Hash of file path + mtime + first/last skill ids.

        A full content hash of a 112 MB file is too slow; this catches
        the common cases (file replaced, re-downloaded, different split).
        """
        import hashlib as _hashlib

        h = _hashlib.sha256()
        h.update(str(self.path).encode("utf-8"))
        try:
            h.update(str(Path(self.path).stat().st_mtime_ns).encode("utf-8"))
        except OSError:
            pass
        return h.hexdigest()

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


@dataclass
class SRABenchCorpus:
    """One SRA-Bench domain slice as an oiax :class:`~oiax.corpus.Corpus`.

    ``corpus.json`` is the full 26,262-skill pool shared by every SRA-Bench
    domain. This class filters it to the skills whose id starts with
    ``f"{domain}_"`` — :data:`SRABENCH_DOMAIN` (``medcalcbench``, 55 skills)
    by default. Filtering happens in :meth:`documents`/:meth:`ids`, never at
    fetch time, so the cached ``corpus.json`` stays reusable across domains.
    """

    path: str | Path
    domain: str = SRABENCH_DOMAIN

    def _records(self) -> Iterator[dict[str, Any]]:
        with Path(self.path).open(encoding="utf-8") as fh:
            for rec in json.load(fh):
                if rec.get("skill_id", "").startswith(f"{self.domain}_"):
                    yield rec

    def fingerprint(self) -> str:
        """Hash of file path + mtime + domain — see :meth:`SkillRetCorpus.fingerprint`."""
        import hashlib as _hashlib

        h = _hashlib.sha256()
        h.update(str(self.path).encode("utf-8"))
        h.update(self.domain.encode("utf-8"))
        try:
            h.update(str(Path(self.path).stat().st_mtime_ns).encode("utf-8"))
        except OSError:
            pass
        return h.hexdigest()

    def ids(self) -> set[str]:
        """The skill ids in this domain slice."""
        return {rec["skill_id"] for rec in self._records()}

    def documents(self) -> Iterator[Document]:
        for rec in self._records():
            description = (rec.get("description") or "").strip()
            if not description:
                # Same rule as SkillRetCorpus: no routing surface, no document.
                continue
            yield Document(
                name=rec["skill_id"],
                trigger_line=description,
                body=rec.get("content") or description,
            )


def load_srabench_labelled(
    instances_path: str | Path,
    *,
    domain: str = SRABENCH_DOMAIN,
    restrict_to: set[str] | None = None,
) -> list[dict[str, Any]]:
    """SRA-Bench instances as oiax labelled items: ``{"prompt", "expected"}``.

    Every MedCalc-Bench instance names exactly one gold skill and every named
    skill is single-domain (measured: 0 cross-domain references, 0 multi-skill
    instances in the 1,100-instance file) — the same single-label IR shape
    SkillRet has, so both corpora are scored the same way.

    ``restrict_to`` mirrors :func:`load_skillret_labelled`: drops instances
    whose gold skill was filtered out of the corpus slice.

    Note there are **no negatives** here either: every instance names a gold
    skill, so the false-alarm rate cannot be measured on this benchmark.
    """
    items: list[dict[str, Any]] = []
    with Path(instances_path).open(encoding="utf-8") as fh:
        for rec in json.load(fh):
            if rec.get("dataset") != domain:
                continue
            expected = [str(s) for s in (rec.get("skill_annotations") or [])]
            if not expected:
                continue
            if restrict_to is not None and not all(e in restrict_to for e in expected):
                continue
            prompt = (rec.get("question") or "").strip()
            if prompt:
                items.append({"prompt": prompt, "expected": expected})
    return items


def family_confusion(
    corpus_path: str | Path,
    labelled: list[dict[str, Any]],
    *,
    top_k: int = 2,
    **index_kwargs: Any,
) -> dict[str, Any]:
    """How often do same-family skills crowd the top-k?

    Family is derived from SkillRet's taxonomy: two skills are in the same
    family when they share a ``major`` taxonomy category or the same skill
    name (205 collisions across 6,006 skills). A query labelled for skill A
    whose top-k also contains skill B (same family, not the gold) is a
    **family-confusion hit**.

    This gates `#24` (selection stage). It is NOT the HSR metric — HSR
    requires query-specific sibling labels, and SkillResolve-Bench (the only
    source) is unavailable. Family-confusion is coarser but measured rather
    than asserted.
    """
    from oiax.router import build_index

    limit: int | None = index_kwargs.pop("limit", None)
    index_kwargs.pop("seed", None)

    corpus = SkillRetCorpus(corpus_path, limit=limit)
    index = build_index(corpus, **index_kwargs)

    id_to_family: dict[str, str] = {}
    _build_family_map(corpus_path, id_to_family)

    n_confused = 0
    n_queries = 0
    examples: list[dict[str, Any]] = []

    for item in labelled:
        prompt = item["prompt"]
        expected = set(item["expected"])
        hits = index.route(prompt)
        hit_names = [h.name for h in hits[:top_k]]

        for h in hit_names:
            if h in expected:
                continue
            h_family = id_to_family.get(h, "")
            if not h_family:
                continue
            for gold in expected:
                gold_family = id_to_family.get(gold, "")
                if gold_family and gold_family == h_family:
                    n_confused += 1
                    if len(examples) < 5:
                        examples.append({
                            "prompt": prompt[:120],
                            "expected": list(expected),
                            "confused_with": h,
                            "family": h_family,
                        })
                    break
            else:
                continue
            break

        n_queries += 1

    rate = n_confused / n_queries if n_queries else 0.0
    return {
        "n_queries": n_queries,
        "n_confused": n_confused,
        "rate": round(rate, 4),
        "top_k": top_k,
        "examples": examples[:3],
    }


def _build_family_map(path: str | Path, out: dict[str, str]) -> None:
    """Populate id->family from SkillRet taxonomy and name collisions."""
    p = Path(path)
    if not p.is_file():
        return

    name_to_ids: dict[str, list[str]] = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("id", "")
            if not sid:
                continue
            name = (rec.get("name") or "").strip().lower()
            if name:
                name_to_ids.setdefault(name, []).append(sid)

            major = (rec.get("major") or "").strip()
            sub = (rec.get("sub") or "").strip()
            if major:
                family = f"major:{major}"
                if sub:
                    family += f"/sub:{sub}"
                out[sid] = family

    for name, ids in name_to_ids.items():
        if len(ids) >= 2:
            for sid in ids:
                existing = out.get(sid, "")
                out[sid] = (existing + " " if existing else "") + f"name:{name}"


def subsample(items: list[dict[str, Any]], n: int, *, seed: int = 0) -> list[dict[str, Any]]:
    """A deterministic subset, for keeping a long run bounded.

    Deterministic on purpose: a benchmark number that moves between runs because
    the sample moved cannot be compared to itself, let alone to a baseline.
    """
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n)
