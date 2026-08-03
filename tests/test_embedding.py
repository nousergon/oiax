"""Tests for the embedding seam.

Two of these are the point of the feature:

- :func:`test_routing_works_end_to_end_with_no_provider_installed` — impossible
  before the extraction, because the provider import lived inside the router.
  Every test touching the semantic scorer had to load a real ~90 MB model or
  assert the fallback path, which is why the 0.1.1 model-id defect survived: the
  suite could only see the branch that was accidentally always taken.
- :func:`test_no_module_outside_the_adapter_imports_the_provider` — the boundary
  guard. A seam erodes one convenience import at a time.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from oiax import build_index, route, semantic_ready
from oiax.corpus import Document
from oiax.embedding import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL_ID,
    Embedder,
    FastEmbedEmbedder,
    UnknownEmbeddingModel,
    get_embedder,
    set_embedder,
)

SRC = Path(__file__).resolve().parent.parent / "src" / "oiax"
ADAPTER_MODULE = SRC / "embedding.py"

#: Every provider package the router must never name directly.
PROVIDER_IMPORT = re.compile(r"^\s*(?:from|import)\s+(fastembed|sentence_transformers)\b", re.M)


@pytest.fixture(autouse=True)
def _restore_default_embedder():
    yield
    set_embedder(None)


class StubEmbedder:
    """A deterministic embedder with no model and no provider.

    Two documents whose text starts with the same character embed close; others
    embed orthogonally. Enough structure to drive real routing.
    """

    def __init__(self, dim: int = 4, ready: bool = True) -> None:
        self._dim = dim
        self._ready = ready
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i][ord((text or " ")[0]) % self._dim] = 1.0
        return out

    def model_id(self) -> str:
        return "stub/deterministic-v1"

    def dimension(self) -> int:
        return self._dim

    def ready(self) -> bool:
        return self._ready


class _Corpus:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def documents(self):
        return iter(self._docs)


DOCS = [
    Document(name="alpha", trigger_line="alpha things about deployment", body="A"),
    Document(name="beta", trigger_line="beta things about testing", body="B"),
]


# ── the contract ────────────────────────────────────────────────────────────


def test_the_stub_satisfies_the_protocol():
    assert isinstance(StubEmbedder(), Embedder)


def test_the_shipped_adapter_satisfies_the_protocol():
    assert isinstance(FastEmbedEmbedder(), Embedder)


def test_the_default_embedder_is_the_shipped_adapter():
    assert isinstance(get_embedder(), FastEmbedEmbedder)


def test_vectors_are_l2_normalised():
    # Part of the contract, not an implementation detail: a consumer that
    # assumes it and an adapter that does not honour it fail silently, and only
    # on some corpora.
    vectors = FastEmbedEmbedder().embed(["deploying to production", "writing tests"])
    if not len(vectors):
        pytest.skip("embedding model unavailable in this environment")
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_dimension_is_declared_and_matches_what_the_model_returns():
    embedder = FastEmbedEmbedder()
    vectors = embedder.embed(["anything"])
    if not len(vectors):
        pytest.skip("embedding model unavailable in this environment")
    assert vectors.shape[1] == embedder.dimension() == DEFAULT_DIMENSION


def test_model_id_is_one_the_provider_publishes():
    """The id must exist in the provider's own registry.

    Through 0.1.1 it read ``fastembed/all-MiniLM-L6-v2``, which the provider
    rejects — every install raised at load and fell back to lexical-only, for
    four days, with nothing red. The old suite exercised only the fallback path,
    so CI stayed green against a dead feature.
    """
    fastembed = pytest.importorskip("fastembed")

    supported = {m["model"] for m in fastembed.TextEmbedding.list_supported_models()}
    assert DEFAULT_MODEL_ID in supported, (
        f"{DEFAULT_MODEL_ID!r} is not a published model id — routing would "
        "silently degrade to lexical-only on every install"
    )


def test_an_unpublished_id_raises_from_embed_too_not_only_from_ready():
    """Corrected in #19.

    This test used to assert that an unpublished model id produced an empty
    result — i.e. that it DEGRADED. That was the defect, not the contract: it
    made a configuration error look exactly like a machine without the model
    cache. Both entry points now raise, because a caller who never checks
    `ready()` must not get silence either.
    """
    pytest.importorskip("fastembed")
    embedder = FastEmbedEmbedder(model_id="not-a-real-model-id-at-all")
    with pytest.raises(UnknownEmbeddingModel):
        embedder.embed(["x"])


def test_embed_of_nothing_is_empty():
    assert len(StubEmbedder().embed([])) == 0


# ── the point of the extraction ─────────────────────────────────────────────


def test_routing_works_end_to_end_with_no_provider_installed():
    """Impossible before this seam existed.

    The provider import lived inside the router, so a test could not drive the
    semantic scorer without the real model. This drives `build_index` and
    `route` all the way through with a stub that has no model, no download and
    no provider package.
    """
    stub = StubEmbedder()
    set_embedder(stub)
    index = build_index(_Corpus(DOCS))
    assert semantic_ready() is True
    hits = route("alpha question", index)
    assert [h.name for h in hits][:1] == ["alpha"]
    # The scorer really used the stub: one call to embed the corpus, one per route.
    assert stub.calls[0] == ["alpha things about deployment", "beta things about testing"]


def test_a_not_ready_embedder_degrades_to_lexical_only_without_raising():
    set_embedder(StubEmbedder(ready=False))
    index = build_index(_Corpus(DOCS))
    assert semantic_ready() is False
    # Lexical still works, which is the whole point of degradation.
    assert [h.name for h in route("alpha things", index)][:1] == ["alpha"]


def test_the_index_reports_the_installed_embedders_model_id():
    set_embedder(StubEmbedder())
    index = build_index(_Corpus(DOCS))
    assert index.stats.model_id == "stub/deterministic-v1"


def test_set_embedder_none_restores_the_default():
    set_embedder(StubEmbedder())
    set_embedder(None)
    assert isinstance(get_embedder(), FastEmbedEmbedder)


# ── the boundary ────────────────────────────────────────────────────────────


def test_no_module_outside_the_adapter_imports_the_provider():
    """One module names a provider. A seam erodes one convenience import at a time."""
    offenders = []
    for path in SRC.rglob("*.py"):
        if path == ADAPTER_MODULE:
            continue
        if PROVIDER_IMPORT.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, (
        f"provider imported outside {ADAPTER_MODULE.name}: {offenders}. "
        "Substituting the embedding runtime must be one adapter, not a sweep."
    )


def test_the_router_names_no_provider_model_id_or_dimension():
    source = (SRC / "router.py").read_text(encoding="utf-8")
    body = source.split('"""', 2)[-1]  # skip the module docstring's history
    for token in ("fastembed", "all-MiniLM", "TextEmbedding", "384"):
        assert token not in body, f"router.py still names {token!r}"


# ── configuration error vs environment condition ────────────────────────────


def test_an_unpublished_model_id_raises_rather_than_degrading():
    """The distinction #19 exists for.

    An id the provider does not publish will never load on any machine — no
    retry, no cache warm-up, no different host fixes it. Degrading it to
    lexical-only makes it indistinguishable from a machine that merely lacks the
    model cache, which is exactly how 0.1.0-0.1.1 shipped for four days naming a
    model that does not exist.
    """
    pytest.importorskip("fastembed")
    with pytest.raises(UnknownEmbeddingModel) as excinfo:
        FastEmbedEmbedder(model_id="definitely/not-a-published-model").ready()
    assert "not a model this provider publishes" in str(excinfo.value)


def test_a_missing_provider_package_degrades_instead_of_raising(monkeypatch):
    """An absent provider is an ENVIRONMENT condition, not a configuration error.

    This is the half that must keep degrading — the layer never fails closed
    because the package is not installed on some machine.
    """
    import builtins

    real_import = builtins.__import__

    def no_fastembed(name, *args, **kwargs):
        if name == "fastembed":
            raise ImportError("no fastembed here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_fastembed)
    embedder = FastEmbedEmbedder()
    assert embedder.ready() is False  # degraded, not raised


def test_the_shipped_model_id_loads_without_raising():
    # The guard must not fire on the configuration the package actually ships.
    pytest.importorskip("fastembed")
    assert FastEmbedEmbedder().ready() is True
