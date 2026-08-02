"""Semantic policy router — the transferable core.

`route(prompt) -> list[RouteHit]` scores a free-text prompt against a
governance corpus using hybrid retrieval (semantic cosine via local ONNX
embeddings + lexical TF-IDF with a union fallback).

The secret sauce is the retrieval design for normative text, not the
algorithm. The four load-bearing decisions (see oiax positioning doc §4):

1. **Whole-document delivery, never chunks.** A rule and its carve-out
   are semantically distant but logically inseparable.
2. **Precision over recall, asymmetric errors.** A miss degrades to the
   status quo; a false positive actively degrades the layer.
3. **Surface names, never rules.** A route is probabilistic; showing
   matched terms makes a bad match dismissible at a glance.
4. **Pin what the prompt cannot reveal; route what it can.** The
   always-resident / on-demand split.

Measured 2026-07-29 over 120 judge-labelled real prompts: lexical-only
recall 19%, precision ~50%. The hybrid scorer holds 15/15 positives,
0/10 negatives. Warm route ~6ms; cold index build ~700ms, cached
process-wide.

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

from oiax.corpus import Corpus, Document

logger = logging.getLogger(__name__)

# ── embedding model ─────────────────────────────────────────────────────────
# fastembed provides local ONNX inference — no network call after initial
# model download. Model is cached per-process; cold first-build downloads
# and converts (~700ms), warm route is ~6ms.

_EMBEDDER: Any = None  # fastembed.TextEmbedding, lazy-loaded
_EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2
_MODEL_NAME: str = "fastembed/all-MiniLM-L6-v2"


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


# ── route result ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RouteHit:
    """One routed document with its score and explanation.

    `name` is a SURFACE NAME only — never inlined rule text. A route is
    probabilistic; showing matched terms makes a bad match dismissible.
    """

    name: str
    score: float  # [0, 1], union of semantic + lexical
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

    Holds the two scorers (lexical + semantic) and metadata. Union fallback:
    if the semantic scorer returns nothing, the lexical scorer's results are
    used alone — and vice versa.
    """

    lexical: _LexicalScorer
    semantic: _SemanticScorer
    doc_count: int
    build_time_ms: float

    def route(self, prompt: str) -> list[RouteHit]:
        """Route a prompt against the index.

        Returns hits from the union of lexical and semantic scorers,
        deduplicated by name, sorted by score descending.
        """
        lex_hits = self.lexical.query(prompt)
        sem_hits = self.semantic.query(prompt)

        # Union by name — keep the higher score
        seen: dict[str, RouteHit] = {}
        for name, score, why in lex_hits:
            seen[name] = RouteHit(name=name, score=score, why=why)
        for name, score, why in sem_hits:
            if name not in seen or score > seen[name].score:
                seen[name] = RouteHit(name=name, score=score, why=why)

        hits = sorted(seen.values(), key=lambda h: h.score, reverse=True)
        return hits


# ── public API ──────────────────────────────────────────────────────────────


def build_index(
    corpus: Corpus,
    *,
    expansions: dict[str, str] | None = None,
    lex_threshold: float = 0.15,
    sem_threshold: float = 0.55,
) -> Index:
    """Build a routing index from a corpus.

    Args:
        corpus: Any object satisfying the `Corpus` Protocol — provides
                documents via `.documents() -> Iterator[Document]`.
        expansions: Optional per-document query-expansion phrases. Keys
                    are document names; values are additional text
                    appended to the trigger line for lexical matching.
        lex_threshold: Minimum TF-IDF cosine score for a lexical hit.
        sem_threshold: Minimum embedding cosine score for a semantic hit.

    Returns:
        An `Index` ready for `route()` calls. Cold build ~700ms; the
        embedding model is cached process-wide after first load.
    """
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
