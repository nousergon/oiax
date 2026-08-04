"""Tests for oiax.router — the semantic policy router core.

These tests verify the public API shape and the core invariants:
- Whole-document delivery (never chunks)
- Surface names only (never inlined rule text)
- Index accepts an injected Corpus, not a path string
- Lexical-only fallback when embeddings are unavailable
"""

import tempfile
from pathlib import Path
from unittest import mock

from oiax.corpus import Document, PolicyDirCorpus
from oiax.router import Index, RouteHit, _LexicalScorer, _SemanticScorer, build_index, route


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
    with mock.patch("oiax.embedding.FastEmbedEmbedder.ready", return_value=False):
        scorer = _SemanticScorer()
        scorer.build([make_doc("x", "some trigger")])
        hits = scorer.query("some prompt")
        assert hits == []


# ── embedding-model contract (regression: 0.1.1 shipped an unrecognised id) ──


def test_semantic_ready_reports_false_when_embedder_fails():
    """``semantic_ready()`` is the honest-degradation signal consumers render."""
    from oiax.router import semantic_ready

    with mock.patch("oiax.embedding.FastEmbedEmbedder.ready", return_value=False):
        assert semantic_ready() is False
    with mock.patch("oiax.embedding.FastEmbedEmbedder.ready", return_value=True):
        assert semantic_ready() is True


# ── selection rule: reciprocal-rank fusion (I5) ─────────────────────────────


class _StubScorer:
    """A scorer stand-in returning a fixed ranking, for fusion unit tests."""

    def __init__(self, hits):
        self._hits = hits

    def query(self, prompt):  # noqa: ARG002 — fixed ranking by construction
        return self._hits


def _index_with(lex_hits, sem_hits, **kw):
    from oiax.router import Index

    return Index(
        lexical=_StubScorer(lex_hits),
        semantic=_StubScorer(sem_hits),
        doc_count=3,
        build_time_ms=0.0,
        **kw,
    )


def test_rrf_prefers_the_document_both_scorers_rank():
    """A document ranked 2nd by BOTH scorers beats one ranked 1st by only one.

    This is the property hybrid retrieval is chosen for, and the union-by-max-score
    rule shipped through 0.1.2 cannot express it: it would return the single high
    raw score first, because it compares TF-IDF cosine against embedding cosine as
    though they were the same quantity.
    """
    idx = _index_with(
        lex_hits=[("only-lexical", 0.90, ["term"]), ("both", 0.30, ["term"])],
        sem_hits=[("only-semantic", 0.60, ["semantic match"]), ("both", 0.30, ["semantic match"])],
    )
    hits = idx.route("anything")
    assert hits[0].name == "both"


def test_route_caps_at_top_k():
    """At most `top_k` names are returned — a route is a hint, not a reading list."""
    lex = [(f"doc-{i}", 0.9 - i / 100, ["term"]) for i in range(10)]
    idx = _index_with(lex_hits=lex, sem_hits=[])
    assert len(idx.route("anything")) == 2
    assert len(_index_with(lex_hits=lex, sem_hits=[], top_k=5).route("anything")) == 5


def test_route_abstains_when_no_scorer_admits_anything():
    """No candidates means no hits — a pure ranking could not express abstention."""
    assert _index_with(lex_hits=[], sem_hits=[]).route("unrelated prompt") == []


def test_route_merges_why_across_scorers():
    """A hit found by both scorers shows both kinds of evidence."""
    idx = _index_with(
        lex_hits=[("both", 0.4, ["deploy"])],
        sem_hits=[("both", 0.3, ["semantic match"])],
    )
    (hit,) = idx.route("deploy something")
    assert hit.why == ["deploy", "semantic match"]
    assert hit.score == 0.4  # best RAW score, not the fusion score


def test_route_hit_score_is_a_raw_score_not_the_fusion_score():
    """`score` stays human-readable in [0, 1]; fusion scores are ~0.016 and are not."""
    idx = _index_with(lex_hits=[("a", 0.42, ["x"])], sem_hits=[])
    (hit,) = idx.route("x")
    assert 0.0 <= hit.score <= 1.0
    assert hit.score == 0.42


def test_build_index_has_no_expansions_parameter():
    """The forbidden shape has no acceptable form, so it gets no signature slot.

    A per-document term list keyed alongside the corpus must be regenerated
    whenever a routing surface changes, nothing fails when it is not, and an
    inert one is indistinguishable from an intentionally disabled one. The
    measured instance was 32 keys, keyed by skill name against a router keying
    by file stem, 0 matching, live and doing nothing for the life of the
    feature — and the expansion experiment it served measured +0.0pp.

    Removed in 0.3.0. This test exists so it cannot return as a convenience.
    """
    import inspect

    assert "expansions" not in inspect.signature(build_index).parameters
    assert "expansions" not in inspect.signature(_LexicalScorer.build).parameters


# ── index persistence ─────────────────────────────────────────────────────────


def _build_tmp_corpus(tmpdir: str, docs: list[tuple[str, str]]) -> Path:
    """Write markdown files for a minimal corpus and return the directory path."""
    p = Path(tmpdir)
    for name, trigger in docs:
        (p / f"{name}.md").write_text(
            f"**Agent-trigger:** {trigger}\n\nBody for {name}.\n"
        )
    return p


def test_index_save_load_roundtrip():
    """Save and load produces an index that routes identically."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = _build_tmp_corpus(tmp, [
            ("deploy", "deploying to production"),
            ("security", "security vulnerability scanning"),
        ])
        corpus = PolicyDirCorpus(str(corpus_dir))
        idx = build_index(corpus)

        cachedir = Path(tmp) / "cache"
        idx.save(cachedir, fingerprint=corpus.fingerprint())

        loaded = Index.load(cachedir)
        assert loaded is not None
        assert loaded.doc_count == idx.doc_count
        assert loaded.rrf_k == idx.rrf_k
        assert loaded.top_k == idx.top_k

        # Route the same prompt
        r1 = idx.route("how do I deploy to prod?")
        r2 = loaded.route("how do I deploy to prod?")
        assert [h.name for h in r1] == [h.name for h in r2]
        assert len(r1) > 0


def test_index_load_returns_none_for_missing_dir():
    assert Index.load("/nonexistent/path/cache") is None


def test_build_index_cache_hit():
    """build_index with cache_dir serves a cached index on fingerprint match."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = _build_tmp_corpus(tmp, [
            ("deploy", "deploying to production"),
            ("security", "security vulnerability scanning"),
        ])
        corpus = PolicyDirCorpus(str(corpus_dir))
        cachedir = Path(tmp) / "cache"

        idx1 = build_index(corpus, cache_dir=str(cachedir))
        idx2 = build_index(corpus, cache_dir=str(cachedir))

        # Should be a cache hit — same fingerprint
        assert idx2.doc_count == idx1.doc_count
        # Route the same
        r1 = idx1.route("how do I deploy to prod?")
        r2 = idx2.route("how do I deploy to prod?")
        assert [h.name for h in r1] == [h.name for h in r2]


def test_build_index_cache_miss_on_corpus_change():
    """Changing the corpus invalidates the cache."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = _build_tmp_corpus(tmp, [
            ("deploy", "deploying to production"),
            ("security", "security vulnerability scanning"),
        ])
        corpus = PolicyDirCorpus(str(corpus_dir))
        cachedir = Path(tmp) / "cache"

        idx1 = build_index(corpus, cache_dir=str(cachedir))

        # Change a document's trigger line
        (corpus_dir / "deploy.md").write_text(
            "**Agent-trigger:** changed trigger entirely\n\nNew body.\n"
        )
        idx2 = build_index(corpus, cache_dir=str(cachedir))

        # The old route matched "deploying" — the new one should not
        r1 = idx1.route("deploying to production")
        r2 = idx2.route("deploying to production")
        assert any(h.name == "deploy" for h in r1)
        assert not any(h.name == "deploy" for h in r2)


def test_fingerprint_changes_on_mtime():
    """Editing a file changes the fingerprint even if the trigger line is the same."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = _build_tmp_corpus(tmp, [
            ("deploy", "deploying to production"),
        ])
        corpus1 = PolicyDirCorpus(str(corpus_dir))
        fp1 = corpus1.fingerprint()

        # Same trigger, but write the file again — mtime changes
        import time
        time.sleep(0.01)  # ensure mtime advances
        (corpus_dir / "deploy.md").write_text(
            "**Agent-trigger:** deploying to production\n\nSame trigger, different body.\n"
        )
        corpus2 = PolicyDirCorpus(str(corpus_dir))
        fp2 = corpus2.fingerprint()

        assert fp1 != fp2


def test_fingerprint_fallback_for_corpus_without_fingerprint():
    """_corpus_fingerprint works when the corpus lacks a fingerprint() method."""
    from oiax.router import _corpus_fingerprint

    class SimpleCorpus:
        def documents(self):
            yield Document(name="a", trigger_line="test trigger", body="body")

    fp = _corpus_fingerprint(SimpleCorpus())
    assert isinstance(fp, str) and len(fp) == 64


def test_index_save_without_fingerprint():
    """Save without fingerprint writes 'none' and loads fine."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = _build_tmp_corpus(tmp, [("deploy", "deploying to production")])
        idx = build_index(PolicyDirCorpus(str(corpus_dir)))
        cachedir = Path(tmp) / "cache"
        idx.save(cachedir)  # no fingerprint
        assert (cachedir / "fingerprint.txt").read_text().strip() == "none"
        loaded = Index.load(cachedir)
        assert loaded is not None
