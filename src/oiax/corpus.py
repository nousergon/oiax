"""Corpus loader interface.

A `Corpus` provides documents to the router. The interface is a Protocol —
any callable that yields `Document` objects works. One concrete loader
(`PolicyDirCorpus`) reads markdown files with `**Agent-trigger:**` headers,
matching the nous-ergon-ops policy directory convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable


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


@runtime_checkable
class Corpus(Protocol):
    """A source of documents for the router.

    Any callable that yields `Document` objects satisfies this interface.
    This is the injection point: oiax knows nothing about where documents
    live — the caller provides them.
    """

    def documents(self) -> Iterator[Document]: ...


class PolicyDirCorpus:
    """Reads markdown files with `**Agent-trigger:**` headers from a directory.

    Matches the nous-ergon-ops policy directory convention: each `.md` file
    carries a `**Agent-trigger:**` line whose text after the colon is the
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

    @staticmethod
    def _load_doc(path: Path) -> Document | None:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        trigger_line = ""
        for line in text.splitlines():
            if "**Agent-trigger:**" in line:
                # Extract everything after the colon
                _, sep, after = line.partition("**Agent-trigger:**")
                if sep:
                    trigger_line = after.strip()
                break

        if not trigger_line:
            return None

        return Document(
            name=path.stem,
            trigger_line=trigger_line,
            body=text,
        )
