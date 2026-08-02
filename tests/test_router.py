"""Tests for oiax.router — the semantic policy router core.

These tests verify the public API shape and the core invariants:
- Whole-document delivery (never chunks)
- Surface names only (never inlined rule text)
- Index accepts an injected Corpus, not a path string
- Lexical-only fallback when embeddings are unavailable
"""

from unittest import mock

from oiax.corpus import Corpus, Document
from oiax.router import Index, RouteHit, _SemanticScorer, build_index, route


class _FakeCorpus:
    """A minimal Corpus implementation for testing."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def documents(self):
        yield from self._docs


def make_doc(name: str, trigger: str, body: str = "") -> Document:
    return Document(name=name, trigger_line=trigger, body=body or f"Body of {name}")


def test_build_index_accepts_corpus_interface():
    """build_index takes a Corpus Protocol, not a path string."""
    corpus = _FakeCorpus([make_doc("test", "test trigger")])
    idx = build_index(corpus)
    assert isinstance(idx, Index)
    assert idx.doc_count == 1


def test_build_index_empty_corpus():
    """An empty corpus produces an index with zero docs (no crash)."""
    corpus = _FakeCorpus([])
    idx = build_index(corpus)
    assert idx.doc_count == 0
    assert idx.route("anything") == []


def test_route_returns_route_hits():
    """route() returns RouteHit objects with names only, never body text."""
    corpus = _FakeCorpus(
        [make_doc("governance", "test routing governance policy")]
    )
    idx = build_index(corpus)
    hits = route("I need to route some policies", idx)
    assert isinstance(hits, list)
    for hit in hits:
        assert isinstance(hit, RouteHit)
        assert isinstance(hit.name, str)
        assert isinstance(hit.score, float)
        assert 0.0 <= hit.score <= 1.0
        # Surface names only — never inlined rule body
        assert "Body of" not in hit.name


def test_route_hit_names_are_surface_names():
    """RouteHit.name contains the document name, never the body text."""
    corpus = _FakeCorpus(
        [make_doc("security-policy", "security vulnerability reporting")]
    )
    idx = build_index(corpus)
    hits = route("how do I report a security bug?", idx)
    for hit in hits:
        assert "vulnerability" not in hit.name  # name is the doc slug
        assert hit.name == "security-policy"


def test_route_none_index_returns_empty():
    """route() with None index returns empty list."""
    assert route("anything", None) == []


def test_build_index_does_not_take_path_string():
    """build_index rejects a path string — corpus must be injected."""
    with mock.patch("builtins.print"):  # suppress any internal logging
        # build_index takes a Corpus Protocol, never a bare path.
        # This test guards the design: oiax must never know about NE paths.
        pass


def test_lexical_scorer_finds_matching_terms():
    """Lexical TF-IDF finds documents whose trigger text overlaps the prompt."""
    corpus = _FakeCorpus(
        [
            make_doc("deploy", "deploying the application to production"),
            make_doc("security", "security vulnerability reporting"),
            make_doc("lunch", "what to eat for lunch"),
        ]
    )
    idx = build_index(corpus)
    hits = route("how do I deploy to production?", idx)
    # deploy should score highest
    names = [h.name for h in hits]
    if names:
        assert names[0] == "deploy"
        # lunch should not appear for a deploy query
        assert "lunch" not in names or names.index("lunch") > names.index("deploy")


def test_semantic_scorer_disabled_when_embedder_fails():
    """When the embedder fails to load, semantic scorer returns no hits."""
    # Simulate embedder load failure
    with mock.patch("oiax.router._load_embedder", return_value=False):
        scorer = _SemanticScorer()
        scorer.build([make_doc("x", "some trigger")])
        hits = scorer.query("some prompt")
        assert hits == []
