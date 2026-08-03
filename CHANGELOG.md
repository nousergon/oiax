# Changelog

All notable changes to oiax are recorded here. Versions follow
[semantic versioning](https://semver.org/); each released version has a matching
`v<version>` git tag and GitHub Release, produced by `publish.yml` on merge.

While the major version is `0`, the public API may change between minor versions.

## [0.3.1] — 2026-08-04

### Changed

- **Relicensed from AGPL-3.0 to MIT.** Brian's ruling, 2026-08-04. AGPL exists to stop
  someone running the code as a closed service; oiax is not being monetised, so that
  protection bought nothing while costing the only thing the project is for — being
  installed and tried. `LICENSE`, the `pyproject.toml` licence field and PyPI
  classifier, the README and `CONTRIBUTING.md` all move together, and the version bump
  is what carries the new metadata to PyPI: without a release, `pip show oiax` keeps
  reporting AGPL-3.0 regardless of what the repo says.

  **Versions up to and including 0.3.0 remain AGPL-3.0.** A licence change is not
  retroactive.

- `CONTRIBUTING.md` now requires a DCO sign-off (`git commit -s`). MIT is silent on
  inbound contribution terms — under AGPL the copyleft carried them implicitly, and
  dropping to a permissive licence without a sign-off would leave nothing recording
  the terms a contributor's patch arrives under.

## [0.3.0] — 2026-08-03

### Removed

- **Query expansions — `expansions=` on `build_index()`, `_LexicalScorer.build()`,
  and `--expansions` on both adapters — are gone.** A breaking change, taken rather
  than deprecated: a deprecated parameter that still works is still a recipe for the
  artifact the design forbids, and the shape has no acceptable form.

  A per-document term list keyed alongside the corpus is a **second copy of the
  document's own metadata**. It must be regenerated whenever a routing surface
  changes, nothing fails when it is not, and an inert one is indistinguishable from
  an intentionally disabled one. The measured instance: a 32-key expansion file keyed
  by skill name against a router keying by file stem, **0 of 32 keys matching**, live
  and doing nothing for the life of the feature. The experiment it served measured
  **+0.0pp** on the reference corpus, and two independent lexical experiments agreed
  string matching cannot close paraphrase-shaped misses.

  The retirement had already happened in the deployment that used it. It had **not**
  happened in the library or the README, so the first thing a new adopter read taught
  them to build exactly that artifact. `tests/test_router.py::
  test_build_index_has_no_expansions_parameter` asserts it cannot return as a
  convenience.

### Added

- **`oiax.calibration`** — the operating point is now a first-class object carrying
  its own provenance, not four anonymous constants. `SHIPPED` names the corpus, its
  size and separability, the embedding model, the date and the measured metrics;
  `router.py`'s constants read from it, so there is one definition. `build_index()`
  takes `operating_point=`, with one stated precedence: explicit kwarg > operating
  point > shipped default.
- **`route_eval calibrate`** — runs the shipped grid against *your* corpus and labels
  and writes a loadable operating point. Zero false alarms is a hard gate, then F1,
  then the quieter point; the losing rows are printed. No configuration clearing the
  gate is a finding, not a failure to calibrate.
- **`Index.divergence()`** — compares the running corpus (size, separability, model)
  against what the operating point was measured on, and the Claude Code adapter
  renders the result into the context paragraph rather than a log.
- **`oiax.telemetry`** — the router reports on itself: per-attempt outcome, failure
  **class**, degraded flag, corpus size, and delivered latency alongside the warm
  route. Off by default; `OIAX_TELEMETRY_PATH` or `set_sink()` turns it on.
  `python -m oiax.eval.telemetry_report` reads the log and names the documents that
  never route.
- **`oiax.eval.outcome_eval`** — measures whether routing changes what the agent
  *does*, across arms, and **refuses to state a verdict** without a host-harness arm.

### Fixed

- `set_sink(sink_from_env())` in the Claude Code adapter overwrote a sink an embedding
  caller had installed, and with the environment variable unset replaced it with the
  no-op — switching telemetry off for a caller who had switched it on. Now
  `install_env_sink()`: fills in a default, never overrules an explicit choice.

## [0.1.4] — 2026-08-03

### Fixed

- **The 0.1.3 calibration did not reach the Claude Code adapter.** Its argparse
  carried `--lex-threshold 0.15 --sem-threshold 0.55` as *defaults* — the
  pre-calibration values — so the one deployment that exists kept routing at the old
  operating point (recall 0.185) while the library, its tests and its eval harness all
  measured 0.648. The flags are now true overrides (`default=None`), and the library
  defaults are the single source of truth. Caught by verifying the live hook output
  against a direct `route()` call, not by any test.

## [0.2.0] — 2026-08-03

### Added

- **MCP server adapter** (`oiax[mcp]`, console script `oiax-mcp`). Exposes
  `route_policies(prompt)` and `get_policy(name)` over stdio, so any MCP-capable
  harness — Cursor, Codex, Claude Desktop, an SDK agent — can route against a
  governance corpus. Two tools, not one: routing returns surface names and matched
  evidence and never rule text, and the agent fetches a whole document only once it
  judges the route relevant.
- The index is built once at server start and held in memory. Measured on the
  15-document reference corpus: **248 ms to start, 4.0 ms per `route_policies` call**,
  against ~1.26 s per turn on the fresh-process Claude Code hook path.
- Contract tests drive the server through a real MCP client over the protocol rather
  than calling the handlers directly.

### Changed

- CI coverage floor 70% → 75% (measured 80%).

## [Unreleased]

### Changed

- CI enforces a coverage floor (`--cov-fail-under=70`, measured 75%) instead of
  measuring coverage and discarding the number.
- `publish.yml` creates the `v<version>` tag and GitHub Release alongside the PyPI
  upload, so a published version always has a source ref a consumer can diff.

### Documentation

- `corpus.py` describes the `**Agent-trigger:**` convention directly instead of
  referring to a repository outside readers cannot open.

## [0.1.3] — unreleased at time of writing

### Changed

- **Selection is now reciprocal-rank fusion** across the lexical and semantic
  rankings (`rrf_k=60`), capped at `top_k=2`, replacing a union sorted by raw
  score. TF-IDF cosine and embedding cosine are not on a common scale, so an
  absolute cutoff on either is corpus-dependent.
- `lex_threshold` / `sem_threshold` are now **admission floors** (is this document
  a candidate at all), not the selection rule. Defaults `0.10` / `0.25`, calibrated
  against the new reference corpus; previously `0.15` / `0.55`.

  Measured on that corpus: recall@2 **0.185 → 0.648**, top-1 accuracy
  **0.204 → 0.673**, F1 **0.299 → 0.625**, false alarms 0.000 → 0.000. At the old
  semantic floor no semantic hit could fire at all — correct matches score
  0.40–0.48 — so the "hybrid" router was lexical-only in practice.

### Added

- `reference-policies/` + `reference_labelled.jsonl`: 15 realistic policy documents
  and 52 labelled prompts, the corpus the shipped defaults are calibrated against.
  `eval/corpora/README.md` records the sweep and what the operating point beat.
- `route_eval sweep` command, plus `recall@k`, top-1 accuracy and false-alarm-rate
  metrics. Precision alone is misleading under a top-2 cap.
- Guards that fail against the previous configuration: corpus separability, a
  ratcheting recall/top-1 floor, and an inertness check that at least one prompt
  routes on semantic evidence.

## [0.1.2] — 2026-08-02

### Fixed

- **The embedding model never loaded.** `_MODEL_NAME` was
  `fastembed/all-MiniLM-L6-v2`, an id fastembed does not publish, so every install
  since 0.1.0 raised at load and fell back to lexical-only routing — signalled by a
  single `logger.warning` on stderr, which the reference Claude Code hook discards.
  Now `sentence-transformers/all-MiniLM-L6-v2`, pinned by a test against fastembed's
  own model registry.

### Added

- `semantic_ready()` — the honest-degradation signal. The Claude Code adapter now
  renders a notice into the context paragraph when routing is lexical-only, rather
  than logging where nobody reads.

## [0.1.1] — 2026-08-02

### Fixed

- Valid PyPI licence classifier for AGPL-3.0, unblocking publication.

## [0.1.0] — 2026-08-02

Initial extraction from an internal policy router into a standalone package:
hybrid lexical + semantic routing over a markdown corpus, `PolicyDirCorpus`
loader, Claude Code and stdout adapters, and an evaluation harness.
