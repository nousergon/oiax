"""Corpus loader interface.

A `Corpus` provides documents to the router. The interface is a Protocol —
any callable that yields `Document` objects works. One concrete loader
(`PolicyDirCorpus`) reads markdown files with `**Agent-trigger:**` headers —
one document per file, the trigger line describing when the document applies.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Document:
    """One document in the routing corpus.

    The `trigger_line` is the "match this" line — a short statement of what
    this document governs. It is used for lexical matching (TF-IDF) and
    appears in `RouteHit.why` so a bad match is dismissible at a glance.

    `body` is the full document text. It is NEVER chunked (design decision 1
    — a rule and its carve-out are semantically distant but logically
    inseparable; returning one without the other inverts the policy).
    """

    name: str
    trigger_line: str
    body: str
    depends_on: list[str] = field(default_factory=list)


@runtime_checkable
class Corpus(Protocol):
    """A source of documents for the router.

    Any callable that yields `Document` objects satisfies this interface.
    This is the injection point: oiax knows nothing about where documents
    live — the caller provides them.
    """

    def documents(self) -> Iterator[Document]: ...

    def fingerprint(self) -> str:
        """A content hash that changes only when the corpus changes."""
        ...


class PolicyDirCorpus:
    """Reads markdown files with `**Agent-trigger:**` headers from a directory.

    One document per `.md` file. Each file carries a `**Agent-trigger:**`
    line, and the text after the colon is the
    trigger_line. The filename (without extension) is the document `name`.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def documents(self) -> Iterator[Document]:
        if not self._path.is_dir():
            return
        for md_file in sorted(self._path.glob("*.md")):
            doc = self._load_doc(md_file)
            if doc is not None:
                yield doc

    def fingerprint(self) -> str:
        """Content hash over file names, mtimes and trigger lines.

        Changes when any document is added, removed, renamed, or edited —
        the mtime catches an edit that preserves the trigger line but changes
        the body, which matters once the body scoring arm (#22) lands.
        """
        h = hashlib.sha256()
        for md_file in sorted(self._path.glob("*.md")):
            h.update(md_file.name.encode())
            h.update(str(md_file.stat().st_mtime_ns).encode())
            doc = self._load_doc(md_file)
            if doc is not None:
                h.update(doc.trigger_line.encode())
        return h.hexdigest()

    @staticmethod
    def _load_doc(path: Path) -> Document | None:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        trigger_line = ""
        depends_on: list[str] = []
        for line in text.splitlines():
            if "**Agent-trigger:**" in line:
                _, sep, after = line.partition("**Agent-trigger:**")
                if sep:
                    trigger_line = after.strip()
            # Supported dependency markers in the same header block.
            # Adding one needs no router change — the edge is just data.
            stripped = line.strip()
            for marker_text in ("**Depends-on:**", "**Requires:**", "**See-also:**"):
                if marker_text in stripped:
                    _, sep, after = line.partition(marker_text)
                    if sep:
                        names = [n.strip() for n in after.split(",") if n.strip()]
                        depends_on.extend(names)
                    break

        if not trigger_line:
            return None

        return Document(
            name=path.stem,
            trigger_line=trigger_line,
            body=text,
            depends_on=depends_on,
        )
