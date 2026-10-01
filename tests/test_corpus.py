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


def _policy(status: str | None) -> str:
    head = f"**Status:** {status} · **Owner:** x\n" if status is not None else ""
    return f"# P\n\n{head}**Agent-trigger:** governs test behaviour\n\n## 1. Body\n\nBody.\n"


def test_policy_dir_corpus_skips_inactive_status():
    """Retired and draft documents do not route; active and undeclared ones do."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, status in [
            ("active", "Active"),
            ("undeclared", None),
            ("retired", "Retired (2026-10-01) — superseded by active.md"),
            ("draft", "Draft (successor to active.md)"),
            ("superseded", "superseded"),
        ]:
            (Path(tmp) / f"{name}.md").write_text(_policy(status))
        names = [d.name for d in PolicyDirCorpus(tmp).documents()]
        assert names == ["active", "undeclared"]


def test_policy_dir_corpus_status_read_from_header_only():
    """A body sentence quoting the Status marker cannot deactivate a document."""
    with tempfile.TemporaryDirectory() as tmp:
        body = _policy("Active") + "\n## 2. Notes\n\n**Status:** Retired\n"
        (Path(tmp) / "p.md").write_text(body)
        assert [d.name for d in PolicyDirCorpus(tmp).documents()] == ["p"]


def test_policy_dir_corpus_inactive_statuses_configurable():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "d.md").write_text(_policy("Draft"))
        assert [d.name for d in PolicyDirCorpus(tmp, inactive_statuses=()).documents()] == ["d"]


def test_fingerprint_changes_when_status_changes():
    """Activating a draft changes the corpus even when nothing else does."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "p.md"
        p.write_text(_policy("Draft"))
        before = PolicyDirCorpus(tmp).fingerprint()
        st = p.stat()
        p.write_text(_policy("Active"))
        import os
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert PolicyDirCorpus(tmp).fingerprint() != before
