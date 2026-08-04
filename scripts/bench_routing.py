#!/usr/bin/env python3
"""Benchmark the stages of a per-turn routing invocation.

Runs against a real corpus and prints a table like the one in the README,
so the README number is measured rather than remembered.

Usage:
    python scripts/bench_routing.py [corpus-dir]

A corpus with under ~20 documents understates the index-build stage; the
recommended corpus for reproduction is the reference-policies directory.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _measure_route_warm(index, prompt: str) -> float:
    """Time a warm route() call (index already built)."""
    # Warm
    index.route(prompt)
    t0 = time.perf_counter()
    index.route(prompt)
    return (time.perf_counter() - t0) * 1000


_DEFAULT_CORPUS = (
    Path(__file__).resolve().parent.parent
    / "src" / "oiax" / "eval" / "corpora" / "reference-policies"
)

CORPUS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_CORPUS
PROMPT = (
    "the checkout service is throwing 500s for about a third of requests "
    "and it started ten minutes ago"
)


def main() -> int:
    if not CORPUS_DIR.is_dir():
        print(f"not a directory: {CORPUS_DIR}", file=sys.stderr)
        return 1

    from oiax.corpus import PolicyDirCorpus

    corpus = PolicyDirCorpus(str(CORPUS_DIR))
    docs = list(corpus.documents())
    doc_count = len(docs)

    # ── stage 1: import cost (fresh process) ────────────────────────────
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, "-c", "import oiax"],
        capture_output=True, timeout=30,
    )
    import_ms = (time.perf_counter() - t0) * 1000

    # ── stage 2: model load (triggers on first embed call) ─────────────
    from oiax.embedding import get_embedder

    t0 = time.perf_counter()
    embedder = get_embedder()
    _ = embedder.ready()  # triggers the load
    model_load_ms = (time.perf_counter() - t0) * 1000

    # ── stage 3: index build ─────────────────────────────────────────────
    from oiax.router import build_index

    t0 = time.perf_counter()
    index = build_index(corpus)
    build_ms = (time.perf_counter() - t0) * 1000

    # ── stage 4: route (warm) ────────────────────────────────────────────
    route_ms = _measure_route_warm(index, PROMPT)

    # ── cache hit ────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        cachedir = Path(tmp) / "cache"
        index = build_index(corpus, cache_dir=str(cachedir))
        t0 = time.perf_counter()
        index2 = build_index(corpus, cache_dir=str(cachedir))
        cache_hit_ms = (time.perf_counter() - t0) * 1000
        if index2.doc_count != index.doc_count:
            print("ERROR: cache-hit index doc_count mismatch", file=sys.stderr)
            return 1

    print(f"Corpus: {CORPUS_DIR}  ({doc_count} documents)\n")
    print(f"{'stage':<30} {'cost':>7}")
    print(f"{'-'*30} {'-'*7}")
    print(f"{'import oiax':<30} {import_ms:6.0f} ms")
    print(f"{'model load from cache':<30} {model_load_ms:6.0f} ms")
    print(f"{'index build':<30} {build_ms:6.0f} ms")
    print(f"{'route() warm':<30} {route_ms:6.1f} ms")
    total_ms = import_ms + model_load_ms + build_ms + route_ms
    print(f"{'total (in-process)':<30} {total_ms:6.0f} ms")
    print(f"{'cache hit (index from disk)':<30} {cache_hit_ms:6.0f} ms")
    print()
    print("`import oiax` pulls numpy, scikit-learn, and fastembed. It dominates")
    print("the per-turn cost. route() warm is the advertised ~4 ms. The cache hit")
    print("is what a resident process (MCP server) pays after the first start.")
    print()
    print("Run this script against your own corpus to measure your machine rather")
    print("than quoting a fixed table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
