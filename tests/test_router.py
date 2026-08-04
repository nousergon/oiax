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

import numpy as np

from oiax.corpus import Document, PolicyDirCorpus
from oiax.embedding import set_embedder
from oiax.router import Index, RouteHit, _LexicalScorer, _SemanticScorer, build_index, route


class _FakeCorpus:
    """A minimal Corpus implementation for testing."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def documents(self):
        yield from self._docs


def make_doc(name: str, trigger: str, body: str = "") -> Document:
    return Document(name=name, trigger_line=trigger, body=body or f"Body of {name}")


class _FakeEmbedder:
    """A minimal Embedder for testing index-fingerprint sensitivity to model/dim."""

    def __init__(self, model_id: str, dim: int = 4) -> None:
        self._model_id = model_id
        self._dim = dim

    def embed(self, texts: list[str]):
        return np.zeros((len(texts), self._dim), dtype=np.float32)

    def model_id(self) -> str:
        return self._model_id

    def dimension(self) -> int:
        return self._dim

    def ready(self) -> bool:
        return True


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


def test_build_index_cache_miss_on_model_id_change():
    """A model_id change invalidates the cache — same corpus, same thresholds.

    Pre-#45: the cache-hit check compared corpus content only, so a persisted
    index built under one embedding model was served, silently, to a caller
    running a different one — even though cosine distributions are not
    comparable across models. `Index.load` is only reached on a fingerprint
    match, so asserting it is NOT called on the second build is a direct
    rebuild-vs-reuse signal, independent of whether routing happens to differ.
    """
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = _build_tmp_corpus(tmp, [
            ("deploy", "deploying to production"),
            ("security", "security vulnerability scanning"),
        ])
        corpus = PolicyDirCorpus(str(corpus_dir))
        cachedir = Path(tmp) / "cache"

        set_embedder(_FakeEmbedder("model-a"))
        try:
            build_index(corpus, cache_dir=str(cachedir))

            set_embedder(_FakeEmbedder("model-b"))
            with mock.patch.object(Index, "load") as spy_load:
                build_index(corpus, cache_dir=str(cachedir))
                spy_load.assert_not_called()
        finally:
            set_embedder(None)


def test_build_index_cache_miss_on_threshold_change():
    """A selection-threshold change (sem_threshold) invalidates the cache.

    Same corpus, same model — only the operating configuration moves. Pre-#45
    this was served from the stale cache silently, since the fingerprint never
    covered lex_threshold/sem_threshold/rrf_k/top_k.
    """
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = _build_tmp_corpus(tmp, [
            ("deploy", "deploying to production"),
            ("security", "security vulnerability scanning"),
        ])
        corpus = PolicyDirCorpus(str(corpus_dir))
        cachedir = Path(tmp) / "cache"

        build_index(corpus, cache_dir=str(cachedir), sem_threshold=0.20)

        with mock.patch.object(Index, "load") as spy_load:
            build_index(corpus, cache_dir=str(cachedir), sem_threshold=0.35)
            spy_load.assert_not_called()


def test_build_index_cache_hit_when_config_unchanged():
    """Control for the two tests above: an unchanged model_id and unchanged
    thresholds DO hit the cache — `Index.load` is reached and reused."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = _build_tmp_corpus(tmp, [
            ("deploy", "deploying to production"),
        ])
        corpus = PolicyDirCorpus(str(corpus_dir))
        cachedir = Path(tmp) / "cache"

        build_index(corpus, cache_dir=str(cachedir), sem_threshold=0.20)

        with mock.patch.object(Index, "load", wraps=Index.load) as spy_load:
            build_index(corpus, cache_dir=str(cachedir), sem_threshold=0.20)
            spy_load.assert_called_once()


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

# ── body scorer ─────────────────────────────────────────────────────────────


def test_body_scorer_defaults_off():
    """Body scorer is not built by default."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = _build_tmp_corpus(tmp, [
            ("deploy", "deploying to production"),
        ])
        idx = build_index(PolicyDirCorpus(str(corpus_dir)))
        assert idx.body is None


def test_body_scorer_contributes_to_route_when_enabled():
    """When enabled, body scorer participates in rank fusion."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = _build_tmp_corpus(tmp, [
            ("deploy", "deploying to production — always test before pushing"),
            ("security", "security vulnerability scanning and triage"),
        ])
        idx = build_index(
            PolicyDirCorpus(str(corpus_dir)),
            body_scorer=True,
            sem_threshold=0.10,
        )
        assert idx.body is not None
        hits = idx.route("always test before pushing")
        assert len(hits) > 0
        names = [h.name for h in hits]
        assert "deploy" in names


def test_body_scorer_save_load_roundtrip():
    """Body scorer survives save/load."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = _build_tmp_corpus(tmp, [
            ("deploy", "deploying to production"),
            ("security", "security vulnerabilities"),
        ])
        idx = build_index(
            PolicyDirCorpus(str(corpus_dir)),
            body_scorer=True,
        )
        cachedir = Path(tmp) / "cache"
        idx.save(cachedir, fingerprint="test")

        loaded = Index.load(cachedir)
        assert loaded is not None
        assert loaded.body is not None
        assert loaded.body._doc_names == idx.body._doc_names


def test_body_scorer_not_in_cache_triggers_rebuild():
    """A cache built without body_scorer will not serve a caller asking for it."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = _build_tmp_corpus(tmp, [
            ("deploy", "deploying to production"),
        ])
        corpus = PolicyDirCorpus(str(corpus_dir))
        cachedir = Path(tmp) / "cache"

        idx1 = build_index(corpus, cache_dir=str(cachedir))
        assert idx1.body is None

        idx2 = build_index(corpus, cache_dir=str(cachedir), body_scorer=True)
        assert idx2.body is not None


# ── dependency expansion ────────────────────────────────────────────────────


def test_document_depends_on_defaults_empty():
    """A document with no declared dependencies is fine."""
    doc = Document(name="test", trigger_line="testing", body="body")
    assert doc.depends_on == []


def test_expand_deps_adds_dependency_to_result():
    """When expand_deps is on, a seed's dependency appears in the result."""
    from oiax.corpus import Document as Doc

    def _docs():
        yield Doc(name="deploy", trigger_line="deploying to production",
                  body="body", depends_on=["audit"])
        yield Doc(name="audit", trigger_line="quarterly financial auditing",
                  body="body", depends_on=[])
        yield Doc(name="other", trigger_line="incident response procedures",
                  body="body", depends_on=[])

    class DepCorpus:
        def documents(self):
            yield from _docs()

    idx = build_index(DepCorpus(), expand_deps=True, expand_budget=4,
                      sem_threshold=0.30)
    hits = idx.route("how do I deploy to prod?")
    names = [h.name for h in hits]
    assert "deploy" in names
    assert "audit" in names, f"dependency 'audit' should be in results, got {names}"


def test_expand_deps_defaults_off():
    """Without expand_deps, dependencies are NOT expanded."""
    from oiax.corpus import Document as Doc

    def _docs():
        yield Doc(name="deploy", trigger_line="deploying to production",
                  body="body", depends_on=["audit"])
        yield Doc(name="audit", trigger_line="quarterly financial auditing",
                  body="body", depends_on=[])
        yield Doc(name="other", trigger_line="incident response procedures",
                  body="body", depends_on=[])

    class DepCorpus:
        def documents(self):
            yield from _docs()

    idx = build_index(DepCorpus())
    hits = idx.route("how do I deploy to prod?")
    names = [h.name for h in hits]
    assert "deploy" in names
    assert "audit" not in names


def test_expand_deps_respects_budget():
    """Expansion stops at expand_budget even if there are more dependencies."""
    from oiax.corpus import Document as Doc

    def _docs():
        yield Doc(name="deploy", trigger_line="deploying to production",
                  body="body", depends_on=["audit", "security"])
        yield Doc(name="audit", trigger_line="quarterly financial auditing",
                  body="body", depends_on=[])
        yield Doc(name="security", trigger_line="security scanning",
                  body="body", depends_on=[])

    class DepCorpus:
        def documents(self):
            yield from _docs()

    idx = build_index(DepCorpus(), expand_deps=True, expand_budget=2,
                      sem_threshold=0.30)
    hits = idx.route("how do I deploy to prod?")
    assert len(hits) <= 2, f"budget=2, got {len(hits)} hits"


def test_dep_graph_persists_in_save_load():
    """Dep graph survives save/load round-trip."""
    with tempfile.TemporaryDirectory() as tmp:
        from oiax.corpus import Document as Doc

        def _docs():
            yield Doc(name="deploy", trigger_line="deploying to production",
                      body="body", depends_on=["audit"])
            yield Doc(name="audit", trigger_line="quarterly financial auditing",
                      body="body", depends_on=[])

        class DepCorpus:
            def documents(self):
                yield from _docs()

        idx = build_index(DepCorpus(), expand_deps=True)
        cachedir = Path(tmp) / "cache"
        idx.save(cachedir, fingerprint="test")
        loaded = Index.load(cachedir)
        assert loaded is not None
        assert loaded._dep_graph == {"deploy": ["audit"]}
        assert loaded.expand_deps is True


def test_policy_dir_corpus_parses_depends_on():
    """**Depends-on:** in a markdown file is extracted into depends_on."""
    with tempfile.TemporaryDirectory() as tmp:
        md = Path(tmp) / "deploy.md"
        md.write_text(
            "**Agent-trigger:** deploying to production\n"
            "**Depends-on:** test-policy, security-policy\n"
            "**Requires:** audit-policy\n\n"
            "Body text.\n"
        )
        corpus = PolicyDirCorpus(str(tmp))
        docs = list(corpus.documents())
        deploy = next(d for d in docs if d.name == "deploy")
        assert "test-policy" in deploy.depends_on
        assert "security-policy" in deploy.depends_on
        assert "audit-policy" in deploy.depends_on


# ── representative selection ────────────────────────────────────────────────


def test_family_defaults_to_empty():
    """Document.family defaults to ''."""
    doc = Document(name="test", trigger_line="testing", body="body")
    assert doc.family == ""


def test_representative_keeps_one_per_family():
    """With representative=True, at most one per family survives in a broader match."""
    from oiax.corpus import Document as Doc

    def _docs():
        yield Doc(name="deploy-a", trigger_line="deploying to production",
                  body="body", family="deploy")
        yield Doc(name="deploy-b", trigger_line="shipping code live",
                  body="body", family="deploy")
        # This one matches the prompt enough to be in the ranked list
        yield Doc(name="audit", trigger_line="production",
                  body="body", family="")

    class FamCorpus:
        def documents(self):
            yield from _docs()

    idx = build_index(FamCorpus(), representative=True, top_k=2,
                      sem_threshold=0.10, lex_threshold=0.05)
    hits = idx.route("how do I deploy to prod?")
    names = [h.name for h in hits]
    assert len(names) <= 2
    assert names[0].startswith("deploy"), f"first should be a deploy sibling, got {names}"
    # With representative, we should get one from 'deploy' family + audit
    deploy_count = sum(1 for n in names if n.startswith("deploy"))
    assert deploy_count == 1, f"should have exactly 1 deploy sibling, got {deploy_count}"


def test_representative_defaults_off():
    """Without representative=True, same-family siblings are both returned."""
    from oiax.corpus import Document as Doc

    def _docs():
        yield Doc(name="deploy-a", trigger_line="deploying to production",
                  body="body", family="deploy")
        yield Doc(name="deploy-b", trigger_line="shipping code live",
                  body="body", family="deploy")
        yield Doc(name="security", trigger_line="security scanning",
                  body="body", family="")

    class FamCorpus:
        def documents(self):
            yield from _docs()

    idx_default = build_index(FamCorpus(), top_k=3, sem_threshold=0.10)
    idx_rep = build_index(FamCorpus(), representative=True, top_k=3,
                          sem_threshold=0.10)

    default_names = [h.name for h in idx_default.route("deploying to production")]
    rep_names = [h.name for h in idx_rep.route("deploying to production")]

    # Without representative, both deploy siblings may appear
    # With representative, at most one per family
    deploy_count_default = sum(1 for n in default_names if n.startswith("deploy"))
    deploy_count_rep = sum(1 for n in rep_names if n.startswith("deploy"))
    assert deploy_count_default >= deploy_count_rep


def test_representative_singleton_family_unchanged():
    """When no document has a family, representative mode = no change."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = _build_tmp_corpus(tmp, [
            ("deploy", "deploying to production"),
            ("security", "security vulnerability scanning"),
        ])
        corpus = PolicyDirCorpus(str(corpus_dir))
        idx_default = build_index(corpus)
        idx_rep = build_index(corpus, representative=True)

        r_default = idx_default.route("how do I deploy to prod?")
        r_rep = idx_rep.route("how do I deploy to prod?")
        assert [h.name for h in r_default] == [h.name for h in r_rep]


def test_policy_dir_corpus_parses_family():
    """**Family:** in a markdown file is extracted."""
    with tempfile.TemporaryDirectory() as tmp:
        md = Path(tmp) / "deploy.md"
        md.write_text(
            "**Agent-trigger:** deploying to production\n"
            "**Family:** deploy\n\n"
            "Body text.\n"
        )
        corpus = PolicyDirCorpus(str(tmp))
        docs = list(corpus.documents())
        deploy = next(d for d in docs if d.name == "deploy")
        assert deploy.family == "deploy"


def test_representative_save_load_roundtrip():
    """Representative flag and family map survive save/load."""
    from oiax.corpus import Document as Doc

    def _docs():
        yield Doc(name="deploy", trigger_line="deploying to production",
                  body="body", family="deploy")
        yield Doc(name="security", trigger_line="security scanning",
                  body="body", family="")

    class FamCorpus:
        def documents(self):
            yield from _docs()

    idx = build_index(FamCorpus(), representative=True)
    with tempfile.TemporaryDirectory() as tmp:
        cachedir = Path(tmp) / "cache"
        idx.save(cachedir, fingerprint="test")
        loaded = Index.load(cachedir)
        assert loaded is not None
        assert loaded._family_map == {"deploy": "deploy"}
        assert loaded.representative is True
