"""The embedding provider, behind one seam.

The router used to name the provider directly: ``_MODEL_NAME``, ``from fastembed
import TextEmbedding`` inside a lazy loader, and a bare ``_EMBEDDING_DIM = 384``.
Substituting the embedding runtime therefore meant editing the module that is
supposed to know nothing about where anything comes from — while the corpus,
which has the same shape of dependency, was already a Protocol with adapters.

Three properties were blocked by that coupling, none of which needed a second
provider to matter:

- **Testability.** Every test touching the semantic scorer either loaded a real
  ~90 MB ONNX model or asserted the fallback path. That is why the 0.1.1
  model-id defect survived: the suite could only see the branch that was
  accidentally always taken.
- **Normalisation was undeclared.** Nothing said vectors were L2-normalised;
  ``sklearn``'s cosine normalises internally, so an adapter returning
  unnormalised vectors would work here and break any consumer assuming dot
  product.
- **The dimension was a bare constant** with no assertion that the loaded model
  produces it.

**One module names a provider. This one.** A provider import anywhere else is a
lint failure (``tests/test_embedding.py``), because a boundary erodes one
convenience import at a time.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MODEL_ID",
    "Embedder",
    "FastEmbedEmbedder",
    "get_embedder",
    "set_embedder",
]

#: The shipped model id. It must be one the provider publishes in its own
#: registry — ``tests/test_embedding.py::test_model_id_is_one_the_provider_publishes``
#: pins it there. Through 0.1.1 this read ``fastembed/all-MiniLM-L6-v2``, which
#: the provider does not recognise: EVERY install raised at load, took the
#: lexical-only branch, and routed with no semantic scorer at all — the whole
#: point of the package — while reporting nothing beyond one stderr warning on a
#: path the reference deployment discards.
DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_DIMENSION = 384


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. The whole surface the router needs, and no more.

    **Vectors are L2-normalised**, so cosine similarity is a dot product. This
    is part of the contract rather than an implementation detail: a consumer
    that assumes it and an adapter that does not honour it fail silently and
    only on some corpora.
    """

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts``. Returns ``float32[len(texts), dimension()]``."""
        ...

    def model_id(self) -> str:
        """The identity that enters an index fingerprint and a calibration."""
        ...

    def dimension(self) -> int:
        ...

    def ready(self) -> bool:
        """Whether embedding will actually work, load-triggering and honest.

        A lexical-only route looks identical to a semantic one at the call site,
        which is how the 0.1.1 defect survived. Every caller that renders routes
        to a reader depends on this being the truth rather than an optimistic
        default.
        """
        ...


class FastEmbedEmbedder:
    """Local ONNX inference via ``fastembed``. No network call after first use.

    Behaviour is unchanged from the router's original loader, deliberately: the
    lazy singleton, the tried-and-failed sentinel, and degradation to
    lexical-only rather than raising. This is a seam, not a substitution — a
    recalibration in the same change would make it impossible to tell which
    edit moved the numbers.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self._model_id = model_id
        self._model: Any = None  # None = not tried, False = tried and failed

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding  # the ONE provider import

            self._model = TextEmbedding(model_name=self._model_id)
            logger.info("fastembed model loaded: %s", self._model_id)
        except Exception as exc:
            logger.warning("fastembed unavailable (%s) — routing lexical-only", exc)
            self._model = False
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        if not model or not texts:
            return np.empty((0, self.dimension()), dtype=np.float32)
        vectors = list(model.embed(texts, batch_size=len(texts)))
        arr = np.asarray(vectors, dtype=np.float32)
        # fastembed returns L2-normalised vectors; asserting rather than
        # trusting keeps the contract true if that ever changes upstream.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalised: np.ndarray = (arr / norms).astype(np.float32)
        return normalised

    def model_id(self) -> str:
        return self._model_id

    def dimension(self) -> int:
        return DEFAULT_DIMENSION

    def ready(self) -> bool:
        return bool(self._load())


_EMBEDDER: Embedder = FastEmbedEmbedder()


def set_embedder(embedder: Embedder | None) -> None:
    """Install the process-wide embedder. ``None`` restores the default.

    This is the substitution point. Swapping providers is one adapter and a
    recalibration (the floors are model-specific), never a change to the
    router, the corpus loader or any harness adapter.
    """
    global _EMBEDDER
    _EMBEDDER = embedder if embedder is not None else FastEmbedEmbedder()


def get_embedder() -> Embedder:
    return _EMBEDDER
