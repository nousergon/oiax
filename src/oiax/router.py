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

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from oiax.calibration import SHIPPED, CorpusStats, OperatingPoint, divergence
from oiax.corpus import Corpus, Document

logger = logging.getLogger(__name__)

# ── embedding model ─────────────────────────────────────────────────────────
# fastembed provides local ONNX inference — no network call after initial
# model download. Model is cached per-process; cold first-build downloads
# and converts (~700ms), warm route is ~6ms.

_EMBEDDER: Any = None  # fastembed.TextEmbedding, lazy-loaded
_EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2
# The id must be one fastembed publishes in `TextEmbedding.list_supported_models()`.
# Through 0.1.1 this read "fastembed/all-MiniLM-L6-v2", which fastembed does not
# recognise: EVERY install raised at load, took the lexical-only branch below, and
# routed with no semantic scorer at all — the whole point of the package — while
# reporting nothing beyond one warning on stderr. `test_model_name_is_supported`
# pins the id against fastembed's own registry so a rename cannot repeat it.
_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

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


def _load_embedder() -> Any:
    """Lazy-load the embedding model once per process."""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        from fastembed import TextEmbedding

        _EMBEDDER = TextEmbedding(model_name=_MODEL_NAME)
        logger.info("fastembed model loaded: %s", _MODEL_NAME)
    except Exception as exc:
        logger.warning("fastembed unavailable (%s) — routing lexical-only", exc)
        _EMBEDDER = False  # sentinel: tried and failed
    return _EMBEDDER


def semantic_ready() -> bool:
    """True when the embedding model loaded and routing is semantic + lexical.

    False means the scorer is lexical-only. Callers that render routes to a user
    MUST surface that — a lexical-only route looks identical to a semantic one at
    the call site, which is how the 0.1.1 model-id defect survived: the warning
    went to stderr and the Claude Code hook discards stderr. Loading is lazy, so
    this triggers the load on first call.
    """
    return bool(_load_embedder())


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
    """TF-IDF over trigger lines + expansions, threshold-gated.

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

    def build(self, docs: list[Document], expansions: dict[str, str] | None = None) -> None:
        """Build TF-IDF matrix over trigger lines + optional expansions."""
        if expansions is None:
            expansions = {}
        self._doc_names = [d.name for d in docs]
        self._doc_terms = [
            d.trigger_line + " " + expansions.get(d.name, "") for d in docs
        ]
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
        embedder = _load_embedder()
        if embedder is False or embedder is None:
            self._doc_embeddings = None
            return

        self._doc_names = [d.name for d in docs]
        trigger_lines = [d.trigger_line for d in docs]
        try:
            embeddings = list(embedder.embed(trigger_lines, batch_size=len(trigger_lines)))
            self._doc_embeddings = np.array(embeddings, dtype=np.float32)
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
        embedder = _load_embedder()
        if embedder is False or embedder is None:
            return []
        try:
            q_embedding = list(embedder.embed([prompt], batch_size=1))
            q_vec = np.array(q_embedding, dtype=np.float32)
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

    @property
    def stats(self) -> CorpusStats:
        """What this corpus looks like, for comparison against the calibration."""
        return CorpusStats(
            size=self.doc_count,
            separability=self.corpus_separability,
            model_id=_MODEL_NAME if self.corpus_separability is not None else "",
        )

    def divergence(self) -> list[str]:
        """Reasons this corpus is unlike the one the operating point came from.

        Empty when nothing is out of range. Callers rendering routes to a reader
        MUST surface a non-empty result — the layer is running and its numbers do
        not mean what its documentation says, which is the same character as the
        lexical-only degradation and belongs on the same surface.
        """
        return divergence(self.stats, self.operating_point)

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

        fused: dict[str, float] = {}
        best: dict[str, float] = {}
        why: dict[str, list[str]] = {}
        for hits in (lex_hits, sem_hits):
            for rank, (name, score, reasons) in enumerate(hits):
                fused[name] = fused.get(name, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                if score > best.get(name, 0.0):
                    best[name] = score
                for reason in reasons:
                    if reason not in why.setdefault(name, []):
                        why[name].append(reason)

        ranked = sorted(fused, key=lambda n: (fused[n], best[n]), reverse=True)
        return [
            RouteHit(name=name, score=best[name], why=why.get(name, []))
            for name in ranked[: self.top_k]
        ]


# ── public API ──────────────────────────────────────────────────────────────


def build_index(
    corpus: Corpus,
    *,
    expansions: dict[str, str] | None = None,
    operating_point: OperatingPoint | None = None,
    lex_threshold: float | None = None,
    sem_threshold: float | None = None,
    rrf_k: int | None = None,
    top_k: int | None = None,
) -> Index:
    """Build a routing index from a corpus.

    Args:
        corpus: Any object satisfying the `Corpus` Protocol — provides
                documents via `.documents() -> Iterator[Document]`.
        expansions: Optional per-document query-expansion phrases. Keys
                    are document names; values are additional text
                    appended to the trigger line for lexical matching.
        lex_threshold: ADMISSION FLOOR for the lexical scorer — the minimum
                       TF-IDF cosine at which a document becomes a candidate.
                       Not a selection gate; see `Index.route`.
        sem_threshold: Admission floor for the semantic scorer (embedding cosine).
        rrf_k: Reciprocal-rank-fusion constant. 60 is the value from Cormack et
               al. (2009) and the one Elasticsearch/OpenSearch use; measured
               here, results are flat for k between 10 and 60.
        top_k: Maximum hits returned. Two, because a route is a hint the reader
               must be able to dismiss at a glance — a list of six is not read.

    Returns:
        An `Index` ready for `route()` calls. Cold build ~700ms; the
        embedding model is cached process-wide after first load.
    """
    point = operating_point or SHIPPED
    # Explicit kwargs override the operating point; the operating point overrides
    # the shipped default. Three layers, one direction, no silent precedence.
    lex_threshold = point.lex_floor if lex_threshold is None else lex_threshold
    sem_threshold = point.sem_floor if sem_threshold is None else sem_threshold
    rrf_k = point.rrf_k if rrf_k is None else rrf_k
    top_k = point.top_k if top_k is None else top_k

    t0 = time.perf_counter()
    docs = list(corpus.documents())
    if not docs:
        logger.warning("empty corpus — index will return no hits")

    lexical = _LexicalScorer(threshold=lex_threshold)
    lexical.build(docs, expansions)

    semantic = _SemanticScorer(threshold=sem_threshold)
    semantic.build(docs)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("index built: %d docs in %.0fms", len(docs), elapsed)

    return Index(
        lexical=lexical,
        semantic=semantic,
        doc_count=len(docs),
        build_time_ms=elapsed,
        rrf_k=rrf_k,
        top_k=top_k,
        operating_point=point,
        corpus_separability=semantic.separability(),
    )


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
