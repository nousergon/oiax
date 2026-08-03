# Evaluation corpora

Two corpora ship with oiax. They answer different questions and are not
interchangeable. Each labelled file is JSONL, one object per line:

```json
{"prompt": "free-text prompt", "expected": ["policy-slug-a", "policy-slug-b"]}
```

`expected` names the policy documents that SHOULD be routed. An empty list is a
**negative** — a prompt no policy governs. Negatives are the only way to measure
the false-alarm rate; a suite without them can be gamed to recall 1.0 by routing
everything.

Neither corpus contains Nous Ergon policy text — the real labelled set stays
private per `repository-tiering-policy.md`.

## `reference-policies/` + `reference_labelled.jsonl` — the calibration corpus

15 policy documents for a generic engineering organisation, each with a realistic
multi-clause `**Agent-trigger:**` line, plus **52 labelled prompts** in natural
engineer voice (49 positive, 3 negative). The shipped selection defaults are
calibrated against this corpus.

```bash
python -m oiax.eval.route_eval score reference-policies < reference_labelled.jsonl
python -m oiax.eval.route_eval sweep reference-policies < reference_labelled.jsonl
```

### Shipped configuration — measured 2026-08-03

| metric | value |
|---|---|
| recall@2 | **0.648** |
| precision | 0.603 |
| F1 | 0.625 |
| top-1 accuracy | 0.673 |
| false-alarm rate (negatives that routed anything) | **0.000** |

Selection is reciprocal-rank fusion (`rrf_k=60`) across both scorers, capped at
`top_k=2`, with admission floors `lex=0.10` / `sem=0.25`.

### What it was chosen over

The full grid is reproducible with `sweep`; the rows that decided it:

| lex | sem | recall | prec | F1 | top-1 | false alarms | |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.25 | 0.704 | 0.559 | 0.623 | 0.673 | 0.333 | best recall, but fires on a negative |
| **0.10** | **0.25** | **0.648** | **0.603** | **0.625** | **0.673** | **0.000** | **shipped** |
| 0.10 | 0.35 | 0.537 | 0.674 | 0.598 | 0.571 | 0.000 | precision bought with 11 points of recall |
| 0.15 | 0.55 | 0.185 | 0.769 | 0.299 | 0.204 | 0.000 | the 0.1.2 default |

Selection *rules* were compared before the floors were tuned. Union-of-scorers
sorted by raw score — what 0.1.2 did — tops out at F1 0.600 anywhere on the grid,
because TF-IDF cosine and embedding cosine are not on a common scale and an
absolute cutoff on either is corpus-dependent. Rank fusion is scale-free and won
at every operating point tried.

### Why the 0.1.2 default was inert

At `sem=0.55` **no** semantic hit fires on this corpus: correct matches score
0.40–0.48 embedding cosine. The hybrid was lexical-only in practice, which is what
recall 0.185 measures. That figure independently reproduces the 19% recall measured
over 120 judge-labelled real prompts on 2026-07-29 against a different, private
corpus — evidence that this corpus is representative rather than tuned to flatter.

### Reading the numbers honestly

- **Precision is capped by design.** With `top_k=2` and one expected label,
  precision cannot exceed 0.5 for that prompt. Precision below 0.5 is the signal
  to watch; 0.603 over a mostly single-label set is near its ceiling.
- **The false-alarm rate rests on 3 negative prompts.** It distinguishes "fires on
  unrelated prompts" from "does not", and nothing finer.
- **The corpus and the prompts share one author.** The prompts deliberately avoid
  trigger-line vocabulary, but cannot fully escape shared framing. Treat these as a
  regression baseline, not as an accuracy claim about your corpus — calibrate
  against your own labelled set before relying on the defaults in another domain.

## `synthetic-policies/` + `synthetic_labelled.jsonl` — the smoke corpus

5 documents with templated trigger lines. A fast structural fixture: does the
loader parse, does the index build, does routing return the right shape.

**It cannot calibrate anything and must not be used to.** All five trigger lines
are one sentence with the policy name substituted, so every document embeds to
nearly the same vector — pairwise cosine spread ~0.04, against 0.55 on the
reference corpus — and recall is flat at 0.40 for every threshold from 0.55 down
to 0.30. A corpus that cannot separate its own documents cannot calibrate a
separation threshold. `tests/test_eval.py` asserts the reference corpus keeps a
spread above 0.15 so this failure mode cannot recur unnoticed.
