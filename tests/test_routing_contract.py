"""``docs/routing-contract.md`` names, for every rule, the code that implements
it and the tests that fail if it breaks. Those names are only worth anything
while they resolve: a renamed test would leave the contract citing an enforcer
that no longer exists, and nothing else would notice.

So every relative link must resolve to a file in this repo, every in-page
anchor must match a heading, and every backticked ``test_*`` name cited
beside a test file must be defined in that file.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs" / "routing-contract.md"

_LINK = re.compile(r"\]\(([^)\s]+)\)")
# "[`tests/x.py`](../tests/x.py) (`test_a`, `test_b`)"
_CITED_TESTS = re.compile(r"\]\((\.\./tests/[^)]+\.py)\) \(([^)]*)\)")


def _text() -> str:
    return CONTRACT.read_text(encoding="utf-8")


def _slug(heading: str) -> str:
    text = heading.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def test_every_relative_link_resolves():
    missing = []
    for link in _LINK.findall(_text()):
        if re.match(r"^[a-z]+:", link):
            continue
        target, _, anchor = link.partition("#")
        if target:
            if not (CONTRACT.parent / target).resolve().exists():
                missing.append(link)
        else:
            headings = {
                _slug(line.lstrip("#"))
                for line in _text().splitlines()
                if line.startswith("#")
            }
            if anchor not in headings:
                missing.append(link)
    assert not missing, f"routing-contract.md links that do not resolve: {missing}"


def test_every_cited_test_exists():
    cited = _CITED_TESTS.findall(_text())
    assert cited, "routing-contract.md cites no tests"
    missing = []
    for path, names in cited:
        source = (CONTRACT.parent / path).resolve().read_text(encoding="utf-8")
        for name in re.findall(r"`(test_\w+)`", names):
            if not re.search(rf"^def {name}\(", source, re.M):
                missing.append(f"{path}::{name}")
    assert not missing, f"routing-contract.md cites tests that do not exist: {missing}"
