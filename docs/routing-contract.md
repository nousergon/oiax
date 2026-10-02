# Routing contract

This is the normative statement of what oiax guarantees about how it builds, caches and selects from an index. The README explains the design and the numbers behind it; this file states the rules. Each section names the code that implements the rule and the tests that fail if it is broken.

A change that breaks one of these rules is a breaking change to oiax, not a tuning change.

## Index identity

Any index that is persisted, cached or shared across processes carries a fingerprint, and it is rebuilt when the fingerprint does not match. The fingerprint covers:

- the corpus content;
- the embedding model id;
- the embedding dimension;
- the selection configuration: the lexical and semantic admission floors, the fusion constant and `top_k`.

The fingerprint is computed after the operating point and any explicit arguments are resolved, so it describes the configuration that will actually be active. A cache built under one model or one set of thresholds is never served to a caller running another: cosine distributions and admission floors are not comparable across either.

- **Implementation:** [`src/oiax/router.py`](../src/oiax/router.py) (`_index_fingerprint`, and the cache-hit check in `build_index`), [`src/oiax/corpus.py`](../src/oiax/corpus.py) (`fingerprint`).
- **Tests:** [`tests/test_router.py`](../tests/test_router.py) (`test_build_index_cache_miss_on_model_id_change`, `test_build_index_cache_miss_on_threshold_change`, `test_build_index_cache_hit_when_config_unchanged`, `test_fingerprint_changes_on_mtime`), [`tests/test_corpus.py`](../tests/test_corpus.py) (`test_fingerprint_changes_when_status_changes`).

## Hybrid retrieval and rank fusion

Routing is hybrid. Two scorers are both required:

- **lexical:** TF-IDF over the routing surfaces, which recovers exact and rare tokens such as command names, identifiers, file names and product names;
- **dense:** cosine similarity over local embeddings, which recovers paraphrase.

Their results are combined by reciprocal rank, never by raw score. Each scorer contributes `1 / (k + rank)` for every document it ranks, and the sums are ordered. Two things are forbidden:

- a union of the two scorers' hits sorted by score;
- any absolute cutoff applied across both scorers.

TF-IDF cosine and embedding cosine are not on a common scale, so only rank fusion is meaningful across them. The lexical and semantic thresholds are admission floors (whether a document is a candidate at all), not the selection rule.

When the embedder cannot run (no cache and no network, a read-only filesystem, an unsupported architecture), routing degrades to lexical-only and says so visibly. That is a runtime degradation, never a silent default. A configuration error that is knowable at build time is not handled this way: see [Model id validation](#model-id-validation).

- **Implementation:** [`src/oiax/router.py`](../src/oiax/router.py) (`Index.route`).
- **Tests:** [`tests/test_router.py`](../tests/test_router.py) (`test_rrf_prefers_the_document_both_scorers_rank`, `test_route_hit_score_is_a_raw_score_not_the_fusion_score`, `test_semantic_scorer_disabled_when_embedder_fails`), [`tests/test_embedding.py`](../tests/test_embedding.py) (`test_a_not_ready_embedder_degrades_to_lexical_only_without_raising`).

## Model id validation

The declared embedding model id is validated against the provider's own published registry, by a test that fails CI. An unrecognised model id is a configuration error knowable at build time, so it is caught at build time and raised. It is never treated as a runtime degradation.

Runtime degradation is reserved for conditions that are genuinely only knowable at runtime: a missing cache, a read-only filesystem, an unsupported architecture.

- **Implementation:** [`src/oiax/embedding.py`](../src/oiax/embedding.py) (the check against `TextEmbedding.list_supported_models()`).
- **Tests:** [`tests/test_embedding.py`](../tests/test_embedding.py) (`test_model_id_is_one_the_provider_publishes`, `test_an_unpublished_id_raises_from_embed_too_not_only_from_ready`, `test_dimension_is_declared_and_matches_what_the_model_returns`).

## Defaults live in the library

Selection parameters are library defaults: the admission floors, the fusion constant and `top_k`. An adapter may override one explicitly, but it may not restate one. A literal default in an adapter's argument parser is forbidden, because a default copied into a caller stops tracking the library: a recalibration then reaches the library and its tests but not the deployment.

Precedence runs one way: an explicit argument overrides the calibrated operating point, which overrides the shipped default.

- **Implementation:** [`src/oiax/router.py`](../src/oiax/router.py) (`LEX_FLOOR`, `SEM_FLOOR`, `RRF_K` and their resolution in `build_index`).
- **Tests:** [`tests/test_adapters.py`](../tests/test_adapters.py) (`test_claude_code_adapter_uses_the_calibrated_library_defaults`).
