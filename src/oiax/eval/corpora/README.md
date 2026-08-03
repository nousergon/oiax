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

**The operating point names the model it was measured under**, and an explicitly
supplied point whose model disagrees with the installed embedder **raises**
rather than routing quietly: cosine distributions are not comparable between
models, so carrying floors across a model change is running on a number measured
on a system that no longer exists. The shipped default against a swapped
embedder is *not* an error — that is the ordinary adopter case and it goes to the
divergence signal.

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
nearly the same neighbourhood and recall is flat at 0.40 for every threshold from
0.55 down to 0.30. A corpus that cannot separate its own documents cannot
calibrate a separation threshold.

**Figure corrected 2026-08-03.** This paragraph read "pairwise cosine spread
~0.04". Measured against the shipped corpus with the metric the test and the
divergence signal both use — max minus min off-diagonal cosine — it is **0.222**,
against **0.553** on the reference corpus. Mean pairwise similarity tells the
story the old figure was reaching for: **0.663** here versus **0.258** on the
reference corpus, i.e. these five documents sit on top of each other.

The correction matters because 0.222 clears the 0.15 floor that
`tests/test_eval.py` asserts. **The floor alone does not catch this corpus** —
what catches it is the flat recall curve, and the floor's job is only to catch
the fully degenerate case.

## `reference_tasks.jsonl` — the outcome task set

Everything above measures whether **retrieval** is accurate. This measures
whether **running the layer changes what the agent does**, which is a different
question and the only one an adopter has. `oiax.eval.outcome_eval` is the
harness; this file is its task set.

Each line is a prompt whose response has a **deterministically checkable**
property:

```json
{"prompt": "We found a bad regression in prod from this morning's release. What should I do?",
 "checks": [{"kind": "mentions_any", "terms": ["roll back", "rollback"]}],
 "governing": ["deployment-policy"],
 "note": "Rollback is the default response to a production regression..."}
```

Check kinds: `mentions_all`, `mentions_any`, `mentions_none`, `regex`. All of a
task's checks must pass. **No model grades anything** — a judge would import its
own error into the one number this harness exists to produce, and an outcome
measure is worth having precisely because it is harder to fool than a retrieval
metric.

**Six positive tasks and two negatives.** The tasks are chosen so the governing
rule contradicts the prompt's own framing: the flaky-test prompt proposes a
retry the policy forbids, the coverage prompt asks for a reduction the policy
forbids outright, the sev1 prompt states the exact silence the policy was
written against. A task an agent answers correctly without the document
measures nothing. The two negatives penalise an arm that injects on every turn.

### Running it

```python
from oiax import build_index
from oiax.corpus import PolicyDirCorpus
from oiax.eval.outcome_eval import (
    NoRoutingArm, OiaxArm, CallableArm, HARNESS_ARM,
    load_tasks, run_outcome_eval, format_report,
)

tasks = load_tasks(open("reference_tasks.jsonl"))
index = build_index(PolicyDirCorpus("reference-policies"))

report = run_outcome_eval(
    tasks,
    [NoRoutingArm(), OiaxArm(index=index), CallableArm(fn=your_harness, name=HARNESS_ARM)],
    respond=your_model_call,                      # (prompt, context) -> response
    conditions={"corpus": "reference-policies", "tasks": "reference_tasks.jsonl",
                "model": "<the model you ran>"},
)
print(format_report(report))
```

**oiax takes no model dependency**, here or anywhere: `respond` is supplied by
the caller. That is what keeps the package provider-free and what makes a third
arm driving somebody else's harness possible at all.

### Two refusals built into the report

- **No verdict without the host-harness arm.** "Better than injecting nothing"
  is a bar no adopter cares about — their harness already selects context for
  free (Claude Code loads skills on the model's judgment; Cursor model-judges
  its rules). Without an arm named `harness`, `report.verdict` says so and
  declines. A two-arm result is a measurement, not an answer.
- **Quality and cost are reported together.** `delta_vs` returns both the
  pass-rate delta and the injected-bytes-per-task delta. Equal quality at triple
  the payload is a regression, and the second number is the only thing that says
  so. Cost is in **bytes** by default — exact and provider-neutral; a token count
  is one provider's opinion about the same bytes, so `count_tokens` is optional
  and supplied by the caller.

### No result is published here yet

Running this needs a model, and a number without its corpus, task set and model
is not reproducible — `format_report` warns when conditions are absent and the
bound is printed with every result. When a run happens, its table goes here with
all three conditions named, **including if routing makes no measurable
difference**, which would be worth more than any recall figure in this repo.

## Out-of-org results — SkillRet

Everything above is measured on a corpus, a prompt set and a labelling that
share one author with the design they are cited to support. It is a **regression
surface**. This section is the first evidence that is not.

**[SkillRet](https://huggingface.co/datasets/ThakiCloud/SKILLRET)** — a
skill-retrieval benchmark over **6,006 real skills scraped from public GitHub
repositories**, with 4,392 queries and 7,187 relevance judgements, in classic IR
shape. Nothing is vendored; `oiax.eval.benchmarks.fetch()` downloads the test
split (~112 MB) to a cache directory you name.

```python
from oiax.eval.benchmarks import SkillRetCorpus, load_skillret_labelled, fetch
paths = fetch("~/.cache/oiax/skillret")
corpus = SkillRetCorpus(paths["skills"])
items = load_skillret_labelled(paths["queries"], paths["qrels"], restrict_to=corpus.ids())
```

### Corpus-size curve — measured 2026-08-03, shipped defaults

Deterministic subsets (seed 0); queries whose gold skills were truncated away are
dropped, so every row is answerable at recall 1.0. Query sample capped at 1,500.

| documents | queries | recall@2 | top-1 | precision | F1 | route ms |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 61 | **0.828** | 0.787 | 0.434 | 0.570 | 4.1 |
| 1,000 | 362 | **0.657** | 0.630 | 0.363 | 0.468 | 4.4 |
| 6,006 | 1,500 | **0.407** | 0.547 | 0.331 | 0.365 | 7.7 |

**Recall@2 halves between 200 and 6,006 documents.** The 0.648 measured on the
15-document reference corpus does not describe behaviour at scale, and this table
is why §7.2 requires a size curve rather than a single figure.

**Read top-1 as the comparable number here, not recall@2.** 51% of SkillRet
queries have more than one gold skill, and `top_k=2` cannot retrieve three. So
recall@2 is structurally capped by the cap, and top-1 accuracy (0.547 at 6,006)
degrades far more gently than recall@2 (0.407).

### Does the hybrid earn its place? — 6,006 documents, 1,500 queries

| configuration | recall@2 | top-1 | precision | F1 |
|---|---:|---:|---:|---:|
| **hybrid, shipped floors** | **0.407** | **0.547** | 0.331 | 0.365 |
| lexical only (semantic disabled) | 0.393 | 0.475 | 0.320 | 0.353 |
| semantic only (lexical disabled) | 0.360 | 0.485 | 0.293 | 0.323 |

**Yes, and the margin is in the ranking rather than in the retrieval.** Hybrid
beats lexical-only by **+7.2 pp on top-1** and by only **+1.4 pp on recall@2**:
both scorers usually get the right document into the top two, and fusing them is
what puts it first. That is the clearest independent support the core design
decision has, and it is on a corpus this project did not write.

### The floors barely matter at this scale

| lex / sem | recall@2 | top-1 | F1 |
|---|---:|---:|---:|
| 0.05 / 0.15 | 0.408 | 0.547 | 0.366 |
| 0.05 / 0.25 | 0.408 | 0.547 | 0.366 |
| 0.10 / 0.25 *(shipped)* | 0.407 | 0.547 | 0.365 |
| 0.15 / 0.25 | 0.409 | 0.540 | 0.367 |

A 0.002 spread. **The operating point calibrated so carefully on 15 documents is
close to irrelevant on 6,006** — with that many candidates something always
clears a floor, so the floors gate almost nothing and the fusion does the work.
Two consequences worth stating rather than burying:

- **Abstention was 0% in every configuration.** The property oiax prizes — a
  prompt no scorer admits routes to nothing — is effectively inoperative at this
  scale on this benchmark.
- Recalibrating the floors for this corpus would recover roughly nothing. The
  transfer question §7.2 item 8 asks has an answer here, and it is *"the floors
  do not transfer because at this size they barely operate."*

### Bounds on all of the above

- **No negatives.** Every SkillRet query has a gold skill, so the false-alarm
  rate — the bar oiax actually calibrates against — **is not measurable on this
  benchmark.** These numbers say nothing about how quiet the router is.
- **Queries are LLM-generated** (the dataset records `generator_model`). Out of
  this project's authorship, but synthetic rather than real user prompts.
- **A substitution was made and it is material.** oiax scores an authored
  *routing surface* — a statement of when a document applies. SkillRet skills
  have no such field, so their `description` stands in. A description says what a
  skill *is*; a routing surface says when it *applies*. Part of the gap between
  0.828 and 0.407 is that substitution rather than the router, and no adjustment
  has been made for it.
- **The divergence signal fired at every corpus size**, correctly: the shipped
  operating point was measured on a different corpus and says so.
- The published SkillRet baselines are not reproduced here, so these are absolute
  numbers rather than a head-to-head placement.

## `reference_siblings.jsonl` — the harmful-sibling set

Recall, precision and the false-alarm rate are all blind to one failure: routing
a document from the **right family** that is the **wrong member** of it. The
sibling is not an irrelevant distractor — it shares the domain and the
vocabulary, its matched terms look entirely convincing, and the prompt genuinely
is governed by *something*. A relevance-only scorer improves recall and this
failure at the same time.

**13 prompts across 6 families.** Each carries `sibling`: the same-family
document that is plausible and wrong *for that prompt*.

```json
{"prompt": "A researcher emailed us saying they found a stored XSS...",
 "expected": ["vulnerability-disclosure-policy"],
 "sibling": ["incident-response-policy"],
 "family": "security-report",
 "note": "Incident-shaped vocabulary with no incident..."}
```

The families, each with a **mirrored pair** so the metric cannot be satisfied by
always preferring one member:

| family | members | the confusion |
|---|---|---|
| security-report | vulnerability-disclosure · incident-response · dependency | an external report has no severity and no postmortem clock; a scanner CVE has no acknowledgement clock |
| production-duty | on-call · incident-response | rotation mechanics vs. handling one incident |
| changing-production | infrastructure-as-code · deployment | a Terraform bump is not application code behind a canary |
| merge-gate | testing · pull-request | a flaky test is not a PR-body problem |
| data-governance | access-control · data-retention | revoking a leaver's keys is not a retention window |
| model-spend | model-usage · cost-management | tier choice vs. right-sizing |

One row deliberately has **two gold documents and no sibling** — a change that
adds a required field to a public endpoint is governed by *both* deployment and
api-versioning. Two members of one family may legitimately both govern, and
without that row a representative selector that collapses every family to one
document would look like a pure improvement.

### Measured 2026-08-03 at the shipped defaults

| metric | value |
|---|---|
| **HSR@2** | **0.083** (1 of 12 sibling-labelled prompts) |
| recall@2 | 0.714 |
| top-1 | 0.769 |

`tests/test_eval.py::test_harmful_sibling_rate_does_not_regress` holds a ceiling
of 0.10 — **a ratchet, not the target.** Published work reaches HSR 0 with
within-family representative selection; oiax has none, so the ceiling holds the
line while that stays true.

### Two things this number is not

**It rests on one arguable label.** The single harmful hit is *"which model tier
for a high-volume classification job"* returning cost-management alongside
model-usage. Cost is genuinely an input to tier choice, so that label is the
weakest of the twelve — and it is the only positive. **HSR here is one prompt
wide**, and a metric whose value is decided by its most debatable label cannot
drive a design change on its own.

**0.083 is not comparable to the published 0.236.** That figure is over 661 pairs
in a 7,982-candidate pool. This is 12 prompts over 15 documents with 6 families.
A small corpus has fewer near-neighbours to confuse, so a low HSR here is partly
a property of the corpus.

### The decision this records: representative selection is NOT built yet

`#24` proposes resolve → score → select. On the evidence available it stays
gated, for a reason the benchmark run supplied rather than intuition:

1. **The denominator is too small to grade a change.** One prompt moves HSR by
   8 points. Any selection change would land inside the noise.
2. **The measurement that would justify it now exists elsewhere.** SkillRet ships
   a per-skill taxonomy — `major` / `sub` / `primary_action` / `primary_object` /
   `domain` — over 6,006 real documents, which is a family signal at a scale
   where families actually collide. Its 6,006 skills carry only 5,801 distinct
   names, and those collisions are a second, independent family signal.
3. **The floors turned out to be inert at that scale** (0.002 spread, 0%
   abstention over 6,006 documents), which makes the *selection stage* the more
   likely place for a real gain — and therefore worth measuring properly rather
   than building on a 12-prompt signal.

So the next step for `#24` is measuring HSR on SkillRet families, not writing a
resolver. This file is the regression guard that keeps the current behaviour from
drifting while that happens.

### The set also surfaced four recall misses

Unrelated to siblings, and worth recording because these are squarely-in-domain
prompts:

- *"Checkout has been degraded for forty minutes and I have not posted anything"* → **routed nothing**, missing incident-response.
- *"Bump the Terraform module version for our RDS instance"* → routed **api-versioning-policy** on the word *version*, missing infrastructure-as-code.
- *"An engineer left on Friday, what about their AWS keys"* → **routed nothing**.
- The both-govern row returned api-versioning and pull-request, missing deployment.

Four misses in thirteen prompts, on a corpus this project wrote, at the
configuration it calibrated. Recorded rather than smoothed.
