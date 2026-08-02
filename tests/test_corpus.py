"""Tests for the corpus loader interface."""

import tempfile
from pathlib import Path

from oiax.corpus import Document, PolicyDirCorpus


def test_policy_dir_corpus_loads_trigger_line():
    """PolicyDirCorpus reads markdown files with Agent-trigger headers."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "test-policy.md").write_text(
            "# Test Policy\n\n**Agent-trigger:** governs test behaviour\n\nBody text.\n"
        )
        corpus = PolicyDirCorpus(tmp)
        docs = list(corpus.documents())
        assert len(docs) == 1
        assert docs[0].name == "test-policy"
        assert docs[0].trigger_line == "governs test behaviour"
        assert "Body text." in docs[0].body


def test_policy_dir_corpus_skips_files_without_trigger():
    """Files without an Agent-trigger line are skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "no-trigger.md").write_text("# No trigger\n\nJust docs.\n")
        corpus = PolicyDirCorpus(tmp)
        docs = list(corpus.documents())
        assert len(docs) == 0


def test_policy_dir_corpus_empty_dir():
    """An empty directory produces no documents."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus = PolicyDirCorpus(tmp)
        docs = list(corpus.documents())
        assert len(docs) == 0


def test_policy_dir_corpus_nonexistent_dir():
    """A nonexistent directory produces no documents (no crash)."""
    corpus = PolicyDirCorpus("/nonexistent/path/12345")
    docs = list(corpus.documents())
    assert len(docs) == 0


def test_document_is_frozen():
    """Document is immutable."""
    doc = Document(name="test", trigger_line="t", body="b")
    try:
        doc.name = "other"  # type: ignore[misc]
        assert False, "should have raised"
    except Exception:
        pass
