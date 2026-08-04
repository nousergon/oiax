"""Semantic policy router — the transferable core.

`route(prompt) -> list[RouteHit]` scores a free-text prompt against a
governance corpus using hybrid retrieval — semantic cosine via local ONNX
embeddings plus lexical TF-IDF — combined by reciprocal-rank fusion and
capped at the top two names.

The secret sauce is the retrieval design for normative text, not the
algorithm. The four load-bearing decisions (see oiax positioning doc §4):

1. **Whole-document delivery, never chunks.** A rule and its carve-out
   are semantically distant but logically inseparable.
2. **Precision over recall, asymmetric errors.** A miss degrades to the
   status quo; a false positive actively degrades the layer. Implemented as
   the `top_k=2` cap and the zero-false-alarm bar on the negative prompts,
   NOT as a high score threshold — a threshold high enough to guarantee
   precision silences the semantic scorer entirely, which is how 0.1.2
   reached precision 0.769 at recall 0.185. Precision bought by not
   answering is not precision.
3. **Surface names, never rules.** A route is probabilistic; showing
   matched terms makes a bad match dismissible at a glance.
4. **Pin what the prompt cannot reveal; route what it can.** The
   always-resident / on-demand split.

Measured 2026-08-03 on the shipped reference corpus (15 documents, 52
labelled prompts — `eval/corpora/README.md` records the sweep): recall@2
0.648, top-1 accuracy 0.673, precision 0.603, zero false alarms on the
negative prompts. Warm route ~6ms; cold index build ~700ms, cached
process-wide.

**Superseded claim, kept visible on purpose.** Through 0.1.2 this docstring
read "the hybrid scorer holds 15/15 positives, 0/10 negatives". No shipped
version ever ran the hybrid: 0.1.0-0.1.1 named an embedding model fastembed
does not publish, and 0.1.2 fixed the name but kept a semantic floor (0.55)
above every correct match a real corpus produces (0.40-0.48). Both were
lexical-only in practice, which is what the 19% recall measured over 120
judge-labelled prompts on 2026-07-29 was actually measuring.

This module is a port of `nous-ergon-ops/scripts/policy_router.py`
(505 lines, 2026-07-28), stripped of every NE-specific path and
refactored against the oiax product contract.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from oiax.calibration import SHIPPED, CorpusStats, OperatingPoint, divergence
from oiax.corpus import Corpus, Document
from oiax.embedding import get_embedder

logger = logging.getLogger(__name__)

# ── embedding ───────────────────────────────────────────────────────────────
# The provider lives in `oiax.embedding` and is named NOWHERE in this module.
# Substituting it is one adapter plus a recalibration (the floors are
# model-specific), never a change here.


def semantic_ready() -> bool:
    """True when the embedder loaded and routing is semantic + lexical.

    False means the scorer is lexical-only. Callers that render routes to a user
    MUST surface that — a lexical-only route looks identical to a semantic one at
    the call site, which is how the 0.1.1 model-id defect survived: the warning
    went to stderr and the Claude Code hook discards stderr. Loading is lazy, so
    this triggers the load on first call.
    """
    return get_embedder().ready()


# ── selection defaults ──────────────────────────────────────────────────────
# Calibrated 2026-08-03 against `eval/corpora/reference-policies` (15 documents)
# and `reference_labelled.jsonl` (52 prompts). The full sweep, including the rules
# that lost, is recorded in `eval/corpora/README.md` and is reproducible with
# `python -m oiax.eval.route_eval sweep <corpus-dir> < labelled.jsonl`.
#
# The floors are ADMISSION thresholds (is this document a candidate?), not the
# selection rule — `Index.route` selects by reciprocal-rank fusion. Through 0.1.2
# these were 0.15 / 0.55 and were the selection rule; that scored recall 0.185,
# with the semantic floor sitting ABOVE every correct match the corpus produces.
#
# Chosen over lex=0.05 (recall 0.704, F1 0.623), which scored marginally better on
# recall and WORSE on false alarms: it routed a policy for 1 of the 3 negative
# prompts, where 0.10 routes nothing for any of them. Decision 2 of the product
# contract — asymmetric errors, a false positive degrades the layer more than a
# miss — settles a tie this close toward the quieter operating point.
# ONE DEFAULT, ONE PLACE. These read from `calibration.SHIPPED`, which is the
# single definition and the only thing carrying the provenance — which corpus,
# how many documents, which model, what date. Re-stating the numbers here would
# be the second copy that let a recalibration reach the library, its tests and
# its eval harness while missing the one deployment that existed (#11).
LEX_FLOOR: float = SHIPPED.lex_floor
SEM_FLOOR: float = SHIPPED.sem_floor
RRF_K: int = SHIPPED.rrf_k
TOP_K: int = SHIPPED.top_k


# ── route result ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RouteHit:
    """One routed document with its score and explanation.

    `name` is a SURFACE NAME only — never inlined rule text. A route is
    probabilistic; showing matched terms makes a bad match dismissible.
    """

    name: str
    # The best RAW scorer score that admitted this document — a [0, 1] number a
    # reader can interpret. Hits are ORDERED by reciprocal-rank fusion (see
    # ``Index.route``), not by this field, because the two scorers' raw scores are
    # not on a common scale and sorting a union by them compares apples to oranges.
    score: float
    why: list[str] = field(default_factory=list)  # matched terms/segments


# ── lexical scorer ──────────────────────────────────────────────────────────


class _LexicalScorer:
    """TF-IDF over trigger lines, threshold-gated.

    Ported from policy_router.py:312 — the lexical half of the hybrid.
    """

    def __init__(self, threshold: float = 0.15, max_ngram: int = 3) -> None:
        self._threshold = threshold
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, max_ngram),
            stop_words="english",
        )
        self._doc_names: list[str] = []
        self._doc_terms: list[str] = []  # raw trigger text per doc
        self._matrix: Any = None

    def build(self, docs: list[Document]) -> None:
        """Build the TF-IDF matrix over trigger lines, and nothing else.

        There is deliberately no expansion parameter. A per-document term list
        keyed alongside the corpus is a second copy of the document's own
        metadata: it must be regenerated whenever a routing surface changes,
        nothing fails when it is not, and an inert one is indistinguishable
        from an intentionally disabled one. The measured instance — 32 keys,
        keyed by skill name against a router keying by file stem, 0 matching,
        live and doing nothing for the life of the feature — is why this
        signature has no room for one.
        """
        self._doc_names = [d.name for d in docs]
        self._doc_terms = [d.trigger_line for d in docs]
        if not self._doc_terms:
            self._matrix = None
            return
        self._matrix = self._vectorizer.fit_transform(self._doc_terms)

    def query(self, prompt: str) -> list[tuple[str, float, list[str]]]:
        """Score prompt against the built index. Returns hits above threshold."""
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        try:
            q_vec = self._vectorizer.transform([prompt])
            scores = sklearn_cosine(q_vec, self._matrix).flatten()
        except Exception:
            return []

        hits: list[tuple[str, float, list[str]]] = []
        for i, score in enumerate(scores):
            if score < self._threshold:
                continue
            # Extract matching terms for transparency
            terms = self._extract_terms(prompt, i)
            hits.append((self._doc_names[i], float(score), terms))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits

    def _extract_terms(self, prompt: str, doc_idx: int) -> list[str]:
        """Find terms from the prompt that appear in the doc's trigger text."""
        prompt_lower = prompt.lower()
        doc_text = self._doc_terms[doc_idx].lower()
        # Simple word-overlap extraction
        prompt_words = set(re.findall(r"[a-z]{3,}", prompt_lower))
        doc_words = set(re.findall(r"[a-z]{3,}", doc_text))
        common = prompt_words & doc_words
        return sorted(common)[:10]


# ── semantic scorer ─────────────────────────────────────────────────────────


class _SemanticScorer:
    """Cosine similarity over local ONNX embeddings, threshold-gated.

    Ported from policy_router.py:357 — the semantic half.
    """

    def __init__(self, threshold: float = 0.55) -> None:
        self._threshold = threshold
        self._doc_names: list[str] = []
        self._doc_embeddings: np.ndarray | None = None

    def build(self, docs: list[Document]) -> None:
        """Build embedding matrix for all doc trigger lines."""
        embedder = get_embedder()
        if not embedder.ready():
            self._doc_embeddings = None
            return

        self._doc_names = [d.name for d in docs]
        trigger_lines = [d.trigger_line for d in docs]
        try:
            matrix = embedder.embed(trigger_lines)
            self._doc_embeddings = matrix if len(matrix) else None
        except Exception as exc:
            logger.warning("embedding build failed (%s) — semantic routing disabled", exc)
            self._doc_embeddings = None

    def separability(self) -> float | None:
        """Mean pairwise cosine SPREAD across document vectors.

        The precondition on believing any number derived from a corpus: a
        corpus whose documents all embed to nearly the same vector cannot be
        separated by any threshold, so recall measures flat and a threshold
        chosen on it was chosen on noise. Measured ~0.04 on the first eval
        corpus against 0.55 on the one that replaced it.
        """
        if self._doc_embeddings is None or len(self._doc_embeddings) < 2:
            return None
        sims = sklearn_cosine(self._doc_embeddings, self._doc_embeddings)
        n = len(sims)
        off_diagonal = [sims[i][j] for i in range(n) for j in range(n) if i != j]
        if not off_diagonal:
            return None
        return float(max(off_diagonal) - min(off_diagonal))

    def query(self, prompt: str) -> list[tuple[str, float, list[str]]]:
        """Score prompt against doc embeddings."""
        if self._doc_embeddings is None:
            return []
        embedder = get_embedder()
        if not embedder.ready():
            return []
        try:
            q_vec = embedder.embed([prompt])
            if not len(q_vec):
                return []
            scores = sklearn_cosine(q_vec, self._doc_embeddings).flatten()
        except Exception as exc:
            logger.warning("semantic query failed: %s", exc)
            return []

        hits: list[tuple[str, float, list[str]]] = []
        for i, score in enumerate(scores):
            fscore = float(score)
            if fscore < self._threshold:
                continue
            hits.append((self._doc_names[i], fscore, ["semantic match"]))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits


# ── body scorer ──────────────────────────────────────────────────────────────


class _BodyScorer:
    """Cosine similarity over full document bodies, threshold-gated.

    The semantic scorer (§ "Routing surface only") embeds only trigger lines —
    the short "when this applies" statement. That carries the stability argument
    (a body edit does not move the vector, so retrieval does not drift on
    unrelated changes) and an untested quality claim (body prose carries no
    signal beyond the routing surface). This scorer measures that claim.

    Defaults OFF — it is a measurement arm, not a selection change, until the
    evidence says otherwise. When enabled, its results rank-fuse alongside the
    lexical and semantic scorers.
    """

    def __init__(self, threshold: float = 0.55) -> None:
        self._threshold = threshold
        self._doc_names: list[str] = []
        self._doc_embeddings: np.ndarray | None = None

    def build(self, docs: list[Document]) -> None:
        """Build embedding matrix over full document bodies."""
        embedder = get_embedder()
        if not embedder.ready():
            self._doc_embeddings = None
            return

        self._doc_names = [d.name for d in docs]
        try:
            matrix = embedder.embed([d.body for d in docs])
            self._doc_embeddings = matrix if len(matrix) else None
        except Exception as exc:
            logger.warning("body embed build failed (%s) — body scorer disabled", exc)
            self._doc_embeddings = None

    def query(self, prompt: str) -> list[tuple[str, float, list[str]]]:
        """Score prompt against body embeddings."""
        if self._doc_embeddings is None:
            return []
        embedder = get_embedder()
        if not embedder.ready():
            return []
        try:
            q_vec = embedder.embed([prompt])
            if not len(q_vec):
                return []
            scores = sklearn_cosine(q_vec, self._doc_embeddings).flatten()
        except Exception as exc:
            logger.warning("body query failed: %s", exc)
            return []

        hits: list[tuple[str, float, list[str]]] = []
        for i, score in enumerate(scores):
            fscore = float(score)
            if fscore < self._threshold:
                continue
            hits.append((self._doc_names[i], fscore, ["body match"]))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits


# ── index ───────────────────────────────────────────────────────────────────


@dataclass
class Index:
    """A built index ready for routing.

    Holds the two scorers (lexical + semantic), the fusion parameters, and
    metadata. Either scorer may return nothing — a lexical-only or semantic-only
    result set fuses to itself, so neither half is load-bearing on its own.
    """

    lexical: _LexicalScorer
    semantic: _SemanticScorer
    doc_count: int
    build_time_ms: float
    rrf_k: int = RRF_K
    top_k: int = TOP_K
    operating_point: OperatingPoint = SHIPPED
    corpus_separability: float | None = None
    body: _BodyScorer | None = None
    #: Adjacency: doc → docs it depends on. Populated from ``Document.depends_on``.
    #: Keyed by name so expansion costs a key lookup, never a scan.
    _dep_graph: dict[str, list[str]] = field(default_factory=dict)
    expand_deps: bool = False
    expand_budget: int = 4
    #: name -> family_id. Empty family = singleton.
    _family_map: dict[str, str] = field(default_factory=dict)
    representative: bool = False

    @property
    def stats(self) -> CorpusStats:
        """What this corpus looks like, for comparison against the calibration."""
        return CorpusStats(
            size=self.doc_count,
            separability=self.corpus_separability,
            model_id=get_embedder().model_id() if self.corpus_separability is not None else "",
        )

    def divergence(self) -> list[str]:
        """Reasons this corpus is unlike the one the operating point came from.

        Empty when nothing is out of range. Callers rendering routes to a reader
        MUST surface a non-empty result — the layer is running and its numbers do
        not mean what its documentation says, which is the same character as the
        lexical-only degradation and belongs on the same surface.
        """
        return divergence(self.stats, self.operating_point)

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, cache_dir: str | Path, fingerprint: str = "") -> None:
        """Persist this index to a directory, together with a corpus fingerprint.

        A cache hit serves the pre-built index instantly (~6 ms cold per turn
        instead of ~1.26 s). The fingerprint guards staleness: a cache built
        against a different corpus must never be served silently.
        """
        import json as _json

        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)

        # Fingerprint
        (d / "fingerprint.txt").write_text(fingerprint.strip() or "none", encoding="utf-8")

        # Lexical — joblib for the sklearn TfidfVectorizer (handles stop_words_
        # etc), scipy sparse for the matrix
        try:
            import joblib as _joblib
        except ImportError:
            import pickle as _joblib  # fallback — pickle works for sklearn
        _joblib.dump(self.lexical._vectorizer, d / "lexical_vectorizer.joblib")
        from scipy.sparse import save_npz as _save_sparse
        if self.lexical._matrix is not None:
            _save_sparse(d / "lexical_matrix.npz", self.lexical._matrix)
        _json.dump(self.lexical._doc_names, (d / "lexical_doc_names.json").open("w"))
        _json.dump(self.lexical._doc_terms, (d / "lexical_doc_terms.json").open("w"))

        # Semantic — numpy for the embedding matrix
        import numpy as _np
        if self.semantic._doc_embeddings is not None:
            _np.save(d / "semantic_embeddings.npy", self.semantic._doc_embeddings)
        _json.dump(self.semantic._doc_names, (d / "semantic_doc_names.json").open("w"))

        # Body scorer (optional)
        if self.body is not None and self.body._doc_embeddings is not None:
            _np.save(d / "body_embeddings.npy", self.body._doc_embeddings)
        _json.dump(
            self.body._doc_names if self.body is not None else [],
            (d / "body_doc_names.json").open("w"),
        )

        # Metadata
        _json.dump(
            {
                "doc_count": self.doc_count,
                "build_time_ms": self.build_time_ms,
                "rrf_k": self.rrf_k,
                "top_k": self.top_k,
                "operating_point": self.operating_point.to_dict(),
                "corpus_separability": self.corpus_separability,
                "has_body_scorer": self.body is not None,
                "dep_graph": self._dep_graph,
                "expand_deps": self.expand_deps,
                "expand_budget": self.expand_budget,
                "family_map": self._family_map,
                "representative": self.representative,
            },
            (d / "meta.json").open("w"),
        )

    @classmethod
    def load(cls, cache_dir: str | Path) -> Index | None:
        """Load a persisted index, or None if the directory is missing/corrupt.

        A caller that checks the fingerprint before loading is guarding
        staleness. This method loads whatever is there — the fingerprint
        gate belongs to the caller.
        """
        import json as _json

        d = Path(cache_dir)
        if not d.is_dir():
            return None

        required = [
            "lexical_vectorizer.joblib",
            "lexical_doc_names.json",
            "semantic_doc_names.json",
            "meta.json",
        ]
        missing = [f for f in required if not (d / f).exists()]
        if missing:
            logger.warning("cache dir %s is missing %s — rebuilding", d, missing)
            return None

        try:
            try:
                import joblib as _joblib
            except ImportError:
                import pickle as _joblib
            vectorizer = _joblib.load(d / "lexical_vectorizer.joblib")
            lexical_doc_names: list[str] = _json.load((d / "lexical_doc_names.json").open())
            lexical_doc_terms: list[str] = _json.load((d / "lexical_doc_terms.json").open())

            from scipy.sparse import load_npz as _load_sparse
            sparse_path = d / "lexical_matrix.npz"
            if sparse_path.exists():
                matrix = _load_sparse(str(sparse_path))
            else:
                matrix = None

            import numpy as _np
            emb_path = d / "semantic_embeddings.npy"
            embeddings = _np.load(emb_path) if emb_path.exists() else None
            semantic_doc_names: list[str] = _json.load((d / "semantic_doc_names.json").open())

            meta = _json.load((d / "meta.json").open())
            op_dict = meta.pop("operating_point", {})
            from oiax.calibration import OperatingPoint
            operating_point = OperatingPoint.from_dict(op_dict)
        except Exception as exc:
            logger.warning("failed to load cached index from %s: %s — rebuilding", d, exc)
            return None

        lexical = _LexicalScorer()
        lexical._vectorizer = vectorizer
        lexical._doc_names = lexical_doc_names
        lexical._doc_terms = lexical_doc_terms
        lexical._matrix = matrix

        semantic = _SemanticScorer()
        semantic._doc_names = semantic_doc_names
        semantic._doc_embeddings = embeddings

        # Body scorer — optional, may not exist in caches built before 0.4.0
        body: _BodyScorer | None = None
        if meta.get("has_body_scorer"):
            body_doc_names: list[str] = _json.load(
                (d / "body_doc_names.json").open()
            )
            body_emb_path = d / "body_embeddings.npy"
            body_embeddings = _np.load(body_emb_path) if body_emb_path.exists() else None
            body = _BodyScorer()
            body._doc_names = body_doc_names
            body._doc_embeddings = body_embeddings

        return cls(
            lexical=lexical,
            semantic=semantic,
            doc_count=meta.get("doc_count", 0),
            build_time_ms=meta.get("build_time_ms", 0.0),
            rrf_k=meta.get("rrf_k", RRF_K),
            top_k=meta.get("top_k", TOP_K),
            operating_point=operating_point,
            corpus_separability=meta.get("corpus_separability"),
            body=body,
            _dep_graph=meta.get("dep_graph", {}),
            expand_deps=meta.get("expand_deps", False),
            expand_budget=meta.get("expand_budget", 4),
            _family_map=meta.get("family_map", {}),
            representative=meta.get("representative", False),
        )

    def route(self, prompt: str) -> list[RouteHit]:
        """Route a prompt against the index, returning at most ``top_k`` hits.

        **Reciprocal-rank fusion**, not a union sorted by raw score. Each scorer
        contributes ``1 / (rrf_k + rank)`` per document it ranks; the fused scores
        are summed and the top ``top_k`` documents are returned. A document ranked
        modestly by BOTH scorers therefore beats one ranked top by a single scorer,
        which is the behaviour hybrid retrieval is chosen for.

        Why fusion rather than comparing scores directly: TF-IDF cosine and
        embedding cosine are not on a common scale, and an absolute cutoff on
        either is corpus-dependent. Shipped through 0.1.2 the defaults were
        ``lex 0.15 / sem 0.55``; measured on the reference corpus that yielded
        recall 0.185 — and NO semantic hit could ever clear 0.55, so the hybrid
        was lexical-only in practice on any corpus resembling it. Fusion plus low
        admission floors measures recall 0.704 on the same corpus
        (``src/oiax/eval/corpora/README.md`` records the sweep). Rank fusion is
        scale-free: it survives a corpus whose absolute scores sit anywhere.

        The per-scorer thresholds still apply, but as ADMISSION FLOORS — "is this
        document a candidate at all" — rather than as the selection rule. That is
        what keeps abstention possible: a prompt no scorer admits routes to
        nothing, which a pure ranking cannot express.
        """
        lex_hits = self.lexical.query(prompt)
        sem_hits = self.semantic.query(prompt)
        body_hits = self.body.query(prompt) if self.body is not None else []

        scorer_sets: list[list[tuple[str, float, list[str]]]] = [lex_hits, sem_hits]
        if body_hits:
            scorer_sets.append(body_hits)

        fused: dict[str, float] = {}
        best: dict[str, float] = {}
        why: dict[str, list[str]] = {}
        for hits in scorer_sets:
            for rank, (name, score, reasons) in enumerate(hits):
                fused[name] = fused.get(name, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                if score > best.get(name, 0.0):
                    best[name] = score
                for reason in reasons:
                    if reason not in why.setdefault(name, []):
                        why[name].append(reason)

        ranked = sorted(fused, key=lambda n: (fused[n], best[n]), reverse=True)

        # ── representative selection ──────────────────────────────────
        if self.representative and self._family_map:
            seen_families: set[str] = set()
            deduped: list[str] = []
            for name in ranked:
                fam = self._family_map.get(name, "")
                if fam and fam in seen_families:
                    continue
                if fam:
                    seen_families.add(fam)
                deduped.append(name)
            seeds: list[str] = deduped[: self.top_k]
        else:
            seeds: list[str] = ranked[: self.top_k]

        # ── dependency expansion ──────────────────────────────────────
        if not self.expand_deps or not self._dep_graph or not seeds:
            def _make_hit(name: str) -> RouteHit:
                why_list = list(why.get(name, []))
                fam_label = self._family_map.get(name, "")
                if self.representative and fam_label:
                    why_list.append(f"family: {fam_label}")
                return RouteHit(name=name, score=best[name], why=why_list)
            return [_make_hit(name) for name in seeds]

        expanded: list[str] = list(seeds)
        seen: set[str] = set(seeds)
        for seed in seeds:
            deps = self._dep_graph.get(seed, [])
            for dep in deps:
                if dep not in seen and len(expanded) < self.expand_budget:
                    expanded.append(dep)
                    seen.add(dep)
        # Dependency hits carry the score and why of their referrer, marked
        # so a consumer can distinguish seeds from expansions.
        dep_hits: list[RouteHit] = []
        for name in expanded:
            if name in best:
                dep_hits.append(RouteHit(
                    name=name, score=best[name],
                    why=why.get(name, []) + (
                        ["dependency of: " + ", ".join(seeds)]
                        if name not in seeds else []
                    ),
                ))
            else:
                # Expanded-only — no direct match score; carry which seed
                # pulled it in at score 0.0 so the reader can see it's an
                # expansion, not a direct hit.
                referrers = [s for s in seeds if name in self._dep_graph.get(s, [])]
                dep_hits.append(RouteHit(
                    name=name, score=0.0,
                    why=["dependency of: " + ", ".join(referrers)],
                ))
        return dep_hits


# ── public API ──────────────────────────────────────────────────────────────


def build_index(
    corpus: Corpus,
    *,
    operating_point: OperatingPoint | None = None,
    lex_threshold: float | None = None,
    sem_threshold: float | None = None,
    rrf_k: int | None = None,
    top_k: int | None = None,
    cache_dir: str | Path | None = None,
    body_scorer: bool = False,
    expand_deps: bool = False,
    expand_budget: int = 4,
    representative: bool = False,
) -> Index:
    """Build a routing index from a corpus.

    Args:
        corpus: Any object satisfying the `Corpus` Protocol — provides
                documents via `.documents() -> Iterator[Document]`.
        lex_threshold: ADMISSION FLOOR for the lexical scorer — the minimum
                       TF-IDF cosine at which a document becomes a candidate.
                       Not a selection gate; see `Index.route`.
        sem_threshold: Admission floor for the semantic scorer (embedding cosine).
        rrf_k: Reciprocal-rank-fusion constant. 60 is the value from Cormack et
               al. (2009) and the one Elasticsearch/OpenSearch use; measured
               here, results are flat for k between 10 and 60.
        top_k: Maximum hits returned. Two, because a route is a hint the reader
               must be able to dismiss at a glance — a list of six is not read.
        cache_dir: Directory to persist and read the built index. A cache hit
                   serves the pre-built index instantly; a fingerprint mismatch
                   triggers a rebuild. A stale cache must never be served
                   silently, so the fingerprint is the only trigger — never an
                   mtime heuristic on the cache file.

    Returns:
        An `Index` ready for `route()` calls. Cold build ~700ms; the
        embedding model is cached process-wide after first load.
    """
    point = operating_point or SHIPPED
    if operating_point is not None:
        # An EXPLICITLY supplied operating point whose model disagrees with the
        # installed embedder is a caller error, not a runtime condition: they
        # passed both, so they meant both, and cosine distributions are not
        # comparable between models. Carrying floors across a model change means
        # running with a number measured on a system that no longer exists —
        # silently, because the router still returns results.
        #
        # The SHIPPED default against a swapped embedder is NOT an error. That
        # is the ordinary adopter case and it goes to the divergence signal
        # (§4.5a), which reports rather than refuses.
        running = get_embedder().model_id()
        if point.model_id and running and point.model_id != running:
            raise ValueError(
                f"operating point was calibrated under {point.model_id!r} but the "
                f"installed embedder is {running!r}. Cosine distributions are not "
                f"comparable between models — recalibrate with "
                f"`route_eval calibrate`, or drop the operating point to fall back "
                f"to the shipped defaults and the divergence signal."
            )
    # Explicit kwargs override the operating point; the operating point overrides
    # the shipped default. Three layers, one direction, no silent precedence.
    lex_threshold = point.lex_floor if lex_threshold is None else lex_threshold
    sem_threshold = point.sem_floor if sem_threshold is None else sem_threshold
    rrf_k = point.rrf_k if rrf_k is None else rrf_k
    top_k = point.top_k if top_k is None else top_k

    # ── cache hit? ──────────────────────────────────────────────────────
    # The fingerprint must be computed AFTER the thresholds above are resolved
    # (not the raw, possibly-None, kwargs) — it has to cover the config that
    # will actually be active, model_id/dimension/thresholds included, or a
    # config change is invisible to the comparison below (#45).
    if cache_dir is not None:
        cache_d = Path(cache_dir)
        fp = _index_fingerprint(
            corpus,
            lex_threshold=lex_threshold,
            sem_threshold=sem_threshold,
            rrf_k=rrf_k,
            top_k=top_k,
        )
        fp_path = cache_d / "fingerprint.txt"
        known = ""
        try:
            if fp_path.exists():
                known = fp_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        if known and known != "none" and known == fp:
            cached = Index.load(cache_dir)
            if cached is not None:
                # A cache built without body_scorer cannot serve a caller
                # asking for it — the body embeddings don't exist on disk.
                if body_scorer and cached.body is None:
                    logger.info("cache hit but body scorer not in cache — rebuilding")
                else:
                    logger.info("index cache hit: %s (%d docs)", cache_d, cached.doc_count)
                    return cached
            # Corrupt cache — fingerprint matched but load failed.
            # Fall through to rebuild.

    t0 = time.perf_counter()
    docs = list(corpus.documents())
    if not docs:
        logger.warning("empty corpus — index will return no hits")

    lexical = _LexicalScorer(threshold=lex_threshold)
    lexical.build(docs)

    semantic = _SemanticScorer(threshold=sem_threshold)
    semantic.build(docs)

    body: _BodyScorer | None = None
    if body_scorer:
        body = _BodyScorer(threshold=sem_threshold)
        # A body embed failure is non-fatal — the body scorer degrades out,
        # same as the semantic scorer degrades out on model-load failure.
        body.build(docs)

    # ── dependency graph ──────────────────────────────────────────────
    dep_graph: dict[str, list[str]] = {}
    if expand_deps:
        for d in docs:
            if d.depends_on:
                dep_graph[d.name] = list(d.depends_on)
    family_map: dict[str, str] = {}
    if representative:
        for d in docs:
            if d.family:
                family_map[d.name] = d.family

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("index built: %d docs in %.0fms", len(docs), elapsed)

    idx = Index(
        lexical=lexical,
        semantic=semantic,
        doc_count=len(docs),
        build_time_ms=elapsed,
        rrf_k=rrf_k,
        top_k=top_k,
        operating_point=point,
        corpus_separability=semantic.separability(),
        body=body,
        _dep_graph=dep_graph,
        expand_deps=expand_deps,
        expand_budget=expand_budget,
        _family_map=family_map,
        representative=representative,
    )

    # ── persist to cache ────────────────────────────────────────────────
    if cache_dir is not None:
        fp = _index_fingerprint(
            corpus,
            lex_threshold=lex_threshold,
            sem_threshold=sem_threshold,
            rrf_k=rrf_k,
            top_k=top_k,
        )
        try:
            idx.save(cache_dir, fingerprint=fp)
            logger.info("index cached: %s (%d docs)", cache_dir, idx.doc_count)
        except Exception as exc:
            logger.warning("failed to persist index cache to %s: %s", cache_dir, exc)

    return idx


def _corpus_fingerprint(corpus: Corpus) -> str:
    """Compute a fingerprint for any corpus that lacks one.

    Hashes all documents (name + trigger_line), which is correct for any
    in-memory corpus. Filesystem-based corpora should implement their own
    ``fingerprint()`` that includes mtimes.
    """
    if hasattr(corpus, "fingerprint"):
        try:
            fp = corpus.fingerprint()
            if fp:
                return str(fp)
        except Exception:
            pass
    # Fallback for corpora without fingerprint()
    import hashlib as _hashlib
    h = _hashlib.sha256()
    for doc in sorted(corpus.documents(), key=lambda d: d.name):
        h.update(doc.name.encode())
        h.update(doc.trigger_line.encode())
    return h.hexdigest()


def _index_fingerprint(
    corpus: Corpus,
    *,
    lex_threshold: float,
    sem_threshold: float,
    rrf_k: int,
    top_k: int,
) -> str:
    """Fingerprint over corpus content AND the configuration the index is
    built under: embedding model id, embedding dimension, and the active
    selection thresholds (lexical/semantic admission floors, RRF constant,
    top-k cap).

    `_corpus_fingerprint()` alone only guards against a *corpus* change — a
    persisted cache built under one embedding model or one set of thresholds
    was served, silently, to a caller running a different model or different
    thresholds, because the cache-hit check compared corpus content only
    (semantic-context-routing-policy.md §3.6; oiax#45). A model swap or a
    threshold change is exactly the kind of change a fingerprint exists to
    catch — cosine distributions and admission floors are not comparable
    across either axis, the same reasoning `build_index`'s explicit
    operating-point check above already applies to a caller-supplied point.
    """
    embedder = get_embedder()
    h = hashlib.sha256()
    h.update(_corpus_fingerprint(corpus).encode())
    h.update(b"\x00model_id=")
    h.update(embedder.model_id().encode())
    h.update(b"\x00dimension=")
    h.update(str(embedder.dimension()).encode())
    h.update(b"\x00lex_threshold=")
    h.update(repr(float(lex_threshold)).encode())
    h.update(b"\x00sem_threshold=")
    h.update(repr(float(sem_threshold)).encode())
    h.update(b"\x00rrf_k=")
    h.update(str(int(rrf_k)).encode())
    h.update(b"\x00top_k=")
    h.update(str(int(top_k)).encode())
    return h.hexdigest()


def route(prompt: str, index: Index | None = None) -> list[RouteHit]:
    """Route a prompt against a pre-built index.

    This is the convenience entry point for the common case where the
    caller holds a built index. For repeated routing, pre-build once
    and pass the same index.

    Args:
        prompt: The free-text prompt to route.
        index: A pre-built `Index`. If None, returns an empty list
               (convenience — callers should pre-build).

    Returns:
        List of `RouteHit` objects, sorted by score descending.
    """
    if index is None:
        return []
    return index.route(prompt)
