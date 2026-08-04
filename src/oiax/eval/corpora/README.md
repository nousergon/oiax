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

## Out-of-org results — SRA-Bench (second corpus, two orders of magnitude from SkillRet)

`semantic-context-routing-policy.md` §7.2 items 6-8 requires **at least two**
out-of-org corpora spanning two orders of magnitude of size, a losing score
published rather than tuned away, and the operating point's transfer between
corpora reported rather than assumed. SkillRet alone cannot supply any of the
three: one corpus cannot show a design property holds *across* scales, only
that it held on the one large corpus measured.

**[SRA-Bench](https://huggingface.co/datasets/WeihangSu/SRA-Bench)**
(`WeihangSu/SRA-Bench`, MIT, arXiv:2604.24594) — a skill-retrieval-augmentation
benchmark: one shared 26,262-skill pool plus six task-domain instance files.
This adapter scores the **medcalcbench domain slice**: 55 skills whose id
starts with `medcalcbench_` (medical risk-score and dosage calculators —
"CHA2DS2-VASc Score", "Body Mass Index (BMI)", ...) and the 1,100 instances
that name one of them as gold. Nothing is vendored; `fetch_srabench()`
downloads the shared corpus file (~232 MB, immutable, cached) plus the domain
instance file to a cache directory you name.

```python
from oiax.eval.benchmarks import SRABenchCorpus, load_srabench_labelled, fetch_srabench
paths = fetch_srabench("~/.cache/oiax/srabench")
corpus = SRABenchCorpus(paths["corpus"])                 # medcalcbench_* only
items = load_srabench_labelled(paths["instances"], restrict_to=corpus.ids())
```

**55 documents is a documented subset of a larger public release, not an
independent download.** SRA-Bench ships one corpus file shared by all six
domains — there is no per-domain corpus artifact upstream. Restricting to the
`medcalcbench_` prefix is this adapter's choice: the other 25,626 entries in
the shared file are filtered out before the index is built and never scored
against. Every kept document and every kept query is verbatim upstream data.

**log10(6,006 / 55) ≈ 2.04** — SkillRet's full corpus is just over two orders
of magnitude larger than this slice, and both are real, non-author-written,
currently-downloadable corpora with queries and relevance judgements in
classic IR shape.

### Measured 2026-08-04, shipped floors, full domain slice

| metric | value |
|---|---:|
| documents | 55 |
| queries | 1,100 |
| recall@2 | **0.569** |
| precision | 0.285 |
| F1 | 0.380 |
| top-1 accuracy | **0.406** |
| false-alarm rate | 0.000 *(vacuous — see Bounds)* |
| index build | 1.5-1.6 s (55 docs, cold) |
| route | ~4.4 ms/query, warm |

### This is the losing score — published, not tuned away

**On SkillRet the hybrid beats lexical-only. On SRA-Bench it is the reverse.**

| configuration | recall@2 | top-1 | precision | F1 |
|---|---:|---:|---:|---:|
| **hybrid, shipped floors** | 0.569 | 0.406 | 0.285 | 0.380 |
| **lexical only (semantic disabled)** | **0.613** | **0.487** | 0.318 | **0.418** |
| semantic only (lexical disabled) | 0.380 | 0.309 | 0.221 | 0.280 |

Lexical-only beats the shipped hybrid by **+4.4pp recall@2** and **+8.1pp
top-1** — the semantic scorer is net-negative on this corpus at the shipped
floors, not merely inert the way it was on SkillRet's full 6,006. A plausible
reason, stated as a hypothesis rather than a finding: MedCalc-Bench queries are
multi-paragraph clinical case notes that often name the calculator's own
clinical vocabulary directly (a note mentioning atrial fibrillation and a
stroke-risk question shares surface tokens with "CHA2DS2-VASc" more than it
shares embedding-space proximity with the right calculator among 54 similarly
narrow, similarly-worded siblings) — lexical overlap dominates on a corpus of
many narrowly-scoped documents describing adjacent clinical instruments. Not
verified against SkillRet's structure, offered as a difference worth naming
rather than explaining away.

### Does the operating point transfer? No — and recalibrating does not fully recover it either

A full grid sweep (`_LEX_GRID` × `_SEM_GRID`, the same grid `calibrate` uses)
against this corpus, shipped point marked:

| lex | sem | recall@2 | prec | F1 | top-1 | |
|---|---|---:|---:|---:|---:|---|
| 0.05 | 0.35 | 0.607 | 0.304 | 0.405 | 0.463 | |
| 0.10 | 0.25 | 0.569 | 0.285 | 0.380 | 0.406 | **shipped** |
| 0.10 | 0.35 | 0.611 | 0.312 | 0.413 | 0.464 | |
| 0.15 | 0.30 | 0.555 | 0.328 | 0.412 | 0.406 | |
| **0.15** | **0.35** | 0.544 | **0.366** | **0.437** | 0.425 | **best F1 in grid** |

Every negative in this labelled set is absent (see Bounds), so the
zero-false-alarm gate that `calibrate` enforces passes vacuously everywhere on
this grid — nothing here is excluded on that basis, unlike the reference
corpus's calibration run.

**Recalibrating helps, but does not recover what disabling the semantic scorer
gets for free.** The best-F1 point in the grid (`lex=0.15, sem=0.35`) still
trails plain lexical-only on both recall@2 (0.544 vs 0.613) and top-1 (0.425
vs 0.487) — the two metrics this project weights most. So the honest answer to
§7.2 item 8 on this corpus is: **the shipped operating point does not
transfer, and neither does "hybrid" as a design choice** — for this corpus,
`sem_threshold` set high enough to admit almost nothing would outperform every
point measured here, which is a different failure mode from SkillRet's (floors
turned inert but the hybrid's *ranking* advantage held). Two corpora, two
different transfer stories — reported as measured, not reconciled into one
number.

### Bounds on all of the above

- **No negatives, same gap as SkillRet.** Every medcalcbench instance names a
  gold skill, so the false-alarm rate — the number oiax actually calibrates
  against — is not measurable on this benchmark either, and the 0.000 above is
  vacuous rather than evidence the router stays quiet.
- **A substitution was made, same shape as SkillRet's.** oiax scores an
  authored *routing surface*; SRA-Bench skills carry a `description` field
  used in its place ("Compute BMI from weight (kg) and height (cm)."). It
  reads closer to "when this applies" than SkillRet's free-text description
  does — these are single-purpose calculators, not general tools — but it is
  still a description standing in for a trigger line, and no adjustment has
  been made for that gap.
- **Queries are real clinical case reports** (165-11,297 characters, mean
  ~3,000), drawn from published case studies rather than authored or
  LLM-generated for this benchmark — a third query register alongside
  SkillRet's short GitHub-derived queries and the reference corpus's
  engineer-voice prompts.
- **The divergence signal fired correctly**: this corpus separates at ~1.00
  against the operating point's calibration separability of 0.55 — `Index.
  divergence()` reports it, the same mechanism that caught SkillRet at every
  size.
- **One domain slice, not a cross-domain result.** champ (89 skills) and
  bigcodebench (139 skills) are the other candidate slices in the same file;
  they were not run for this result, and a claim generalizing this finding
  across all of SRA-Bench's domains would be unmeasured.

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

## Body-scorer arm (2026-08-04)

`oiax` embeds only the `**Agent-trigger:**` routing surface. The claim that
body text carries no signal beyond the routing surface was **untested** until
now. Measured against the reference corpus:

| configuration | recall@2 | precision | F1 | top-1 | false alarms |
|---|---|---|---|---|---|
| incumbent (lexical + semantic) | 0.648 | 0.603 | 0.625 | 0.673 | 0.000 |
| with body scorer (3-arm RRF) | 0.648 | **0.515** | 0.574 | 0.673 | 0.000 |
| **delta** | **0.000** | **-0.089** | **-0.051** | 0.000 | 0.000 |

Body embedding adds noise, not signal: precision drops 8.9pp with zero recall
gain. The quality leg of "body is never embedded" is now tested rather than
asserted. The stability leg (a body edit does not move the routing vector) is
the one that holds.

The body scorer ships behind a `body_scorer=False` flag as the measurement
arm, not the default. `python -m oiax.eval.route_eval score ./policies/ <
labelled.jsonl` accepts `--body-scorer` to reproduce this table.

## Dependency expansion (2026-08-04)

The router can expand routes along `Document.depends_on` edges when
`build_index(expand_deps=True)`. Measured against the reference corpus:

| configuration | recall@2 | precision | F1 | false alarms |
|---|---|---|---|---|
| no expansion | 0.648 | 0.603 | 0.625 | 0.000 |
| expand_deps (budget=4) | 0.648 | 0.603 | 0.625 | 0.000 |

Zero delta — none of the reference corpus documents declare dependencies.
The field is structural: behavior is unchanged for a corpus that supplies
no edges. A corpus with declared inter-document prerequisites will see
expanded results proportional to its dependency density.

The `Document.depends_on` field, `PolicyDirCorpus` `**Depends-on:**`/
`**Requires:**`/`**See-also:**` extraction, and `--expand-deps` flag
are the mechanism; the measurement above establishes the baseline that
expansion is additive-only (no regression when the corpus has no edges).
