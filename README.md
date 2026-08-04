# oiax

> **oiax** — semantic policy routing for agent fleets. The tiller that delivers the right governance context to the right agent at the right turn, by meaning.

[![CI](https://github.com/nousergon/oiax/actions/workflows/ci.yml/badge.svg)](https://github.com/nousergon/oiax/actions/workflows/ci.yml)

oiax routes a free-text prompt against a governance corpus and delivers the policies that bear on that turn — by meaning, before the agent decides. Warm route ~4 ms with no network call; per-turn delivered cost depends on the harness (see [Performance](#performance)).

**The retrieval design is for normative text** (policies, coding standards, ADRs, compliance rules), not general knowledge. Four decisions make it correct:

1. **Whole-document delivery, never chunks.** A rule and its carve-out are semantically distant but logically inseparable.
2. **Precision over recall, asymmetric errors.** A miss degrades to the status quo; a false positive actively degrades the layer.
3. **Surface names, never rules.** Matched terms make a bad match dismissible at a glance.
4. **Runtime-agnostic core, harness-specific adapters.** The router returns structured hits; each harness gets its own thin delivery layer.

## Installation

```bash
pip install oiax
```

Requires Python ≥ 3.11.

**Provision the embedding model before the first prompt:**

```bash
python -m oiax.provision                       # fetch, verify, record a manifest
python -m oiax.provision --check               # report the state, change nothing
```

Skip it and the model (~90 MB, ONNX) downloads **inside the first routed turn** —
on the path whose whole premise is that it makes no network call. Worse, a machine
with no egress silently becomes a lexical-only router forever, because the only
fallback is degradation and degradation is not an error.

`--check` reports **three states**, because `semantic_ready()`'s single boolean is
right for the router and useless to an operator:

| state | means | exit |
|---|---|---|
| **PRESENT** | loads with the network unavailable — the promise holds | 0 |
| **FETCHABLE** | published, not on this machine. **A first prompt will pay for it.** | 1 |
| **UNAVAILABLE** | not cached and cannot be fetched. Lexical-only until that changes. | 1 |

Non-zero on anything but PRESENT, so it works as a gate in an image build or a
bootstrap script. **PRESENT is established by actually loading with the provider's
offline switch set** — never by looking for files and hoping, because a
cache-shaped directory that does not load is exactly the case worth catching.

Provisioning writes a digest manifest beside the cache and `--check` verifies it,
so a tampered or truncated cache is a **MISMATCH** rather than a mystery. Point
both at a specific location with `--cache-dir` or `$OIAX_MODEL_CACHE`.

The first-use fetch still works — `pip install oiax` and a quick trial is the whole
onboarding path. This makes the cost **avoidable and visible**, not impossible.

## Quick start

### Route a prompt

```python
from oiax import build_index, route
from oiax.corpus import PolicyDirCorpus

# Load from a directory of markdown files, each carrying an **Agent-trigger:**
# line (see "Corpus format" below)
corpus = PolicyDirCorpus("./policies/")
index = build_index(corpus)

# Route a prompt — at most two hits, ranked by reciprocal-rank fusion
hits = route("How do I deploy to production?", index)
for hit in hits:
    print(f"{hit.name} ({hit.score:.2f}): {', '.join(hit.why)}")
```

### Use a custom corpus

```python
from oiax.corpus import Document

class MyCorpus:
    def documents(self):
        yield Document(
            name="deploy-policy",
            trigger_line="deploying to production",
            body="Always run the test suite before deploying...",
        )

hits = route("deploy to prod", build_index(MyCorpus()))
```

## MCP server — Cursor, Codex, and any other MCP-capable harness

The Claude Code hook *pushes* routes in on every prompt. Most harnesses have no
per-prompt hook — Cursor's rules are model-judged with no system-injection point, and
Codex's are static — but they do speak MCP. `oiax[mcp]` serves the router as two tools
an agent can call:

| Tool | Returns |
|---|---|
| `route_policies(prompt)` | at most two `{name, score, why}` — surface names and matched evidence, **never rule text** |
| `get_policy(name)` | the whole document, so a rule never arrives without its carve-out |

```bash
pip install "oiax[mcp]"
oiax-mcp ./policies/
```

Point a harness at that command over stdio — `mcpServers` in `~/.cursor/mcp.json`, `mcp_servers` in `~/.codex/config.toml`, or the equivalent:

```json
{
  "mcpServers": {
    "oiax": {
      "command": "oiax-mcp",
      "args": ["/absolute/path/to/policies/"]
    }
  }
}
```

```toml
[mcp_servers.oiax]
command = "oiax-mcp"
args = ["/absolute/path/to/policies/"]
```

**The index lives in the server process**, which is the point: a fresh-process hook pays
~670 ms per turn (most of it pulling in numpy/scikit-learn/fastembed) against a
15-document corpus with a warm model cache. Here that happens once at start —
measured **248 ms to start on a 15-document corpus, then 4.0 ms per `route_policies`
call**.

## Corpus format

Policy files are markdown with an `**Agent-trigger:**` header — a one-line statement of what the document governs. This is used for both lexical matching (TF-IDF) and semantic matching (embeddings).

```markdown
# My deploy policy

**Agent-trigger:** deploying the application to production, CI/CD configuration

Always run the test suite before deploying. Never deploy on Friday.
```

The `PolicyDirCorpus` loader reads all `*.md` files in a directory. The filename (without `.md`) becomes the document `name` returned in route hits.

## Claude Code integration

`oiax.adapters.claude_code` is a `UserPromptSubmit` hook adapter. Register it in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "python3 -m oiax.adapters.claude_code /path/to/policies/",
          "timeout": 8
        }]
      }
    ]
  }
}
```

On every prompt, the adapter routes the prompt text against the policy corpus and injects a context paragraph naming the policies that may apply — with the matched terms, so a bad match is dismissible at a glance. Never blocks: any error exits 0 silently.

## How selection works

Both scorers rank every document. Their rankings are combined by **reciprocal-rank
fusion** — each scorer contributes `1 / (60 + rank)` — and the top two documents are
returned. A document ranked moderately by *both* scorers therefore beats one ranked
first by only one, which is the whole reason to run a hybrid.

`lex_threshold` and `sem_threshold` are **admission floors** ("is this document a
candidate at all"), not the selection rule. They are what makes abstention possible:
a prompt neither scorer admits routes to nothing.

Absolute score cutoffs are deliberately not the selection rule. TF-IDF cosine and
embedding cosine are not on a common scale, and the right cutoff for either moves
with the corpus. Through 0.1.2 oiax selected on absolute cutoffs with a semantic
threshold of 0.55; on the reference corpus, correct semantic matches score 0.40–0.48,
so the semantic half never fired and recall sat at 0.185. Rank fusion is scale-free.

Defaults are calibrated, not chosen: `src/oiax/eval/corpora/README.md` records the
sweep, the operating point, and what it was picked over.

## Performance

The warm-route claim and the per-turn delivered cost are **not the same number**.
Measured against the 15-document reference corpus on a MacBook Pro (M4, warm
model cache). Run `python scripts/bench_routing.py` to measure your own machine.

```
stage                             cost
------------------------------ -------
import oiax                       630 ms   # numpy, scikit-learn, fastembed
model load from cache             215 ms
index build                        37 ms
route() warm                      3.5 ms
total (in-process)       885 ms

cache hit (index from disk)         3 ms   # fingerprint-matched, skips build
```

**The ~4 ms route is real** — it is also ~0.4% of what a turn pays in a
fresh-process harness (a Claude Code hook, a CI agent).

**The index cache** in `build_index(corpus, cache_dir=...)` persists the built
index to disk, keyed by a corpus fingerprint. A fingerprint match reads the
pre-built index from disk instead of importing the scientific stack and
re-building. Use it in any setup where a persistent process can hold the index.

Two delivery paths:

| Path | Start | Per turn | Use case |
|---|---|---|---|
| `build_index(corpus, cache_dir=...)` + `index.route()` | 3 ms (cache hit) | 4 ms | resident process, MCP server |
| Fresh `route()` per prompt | ~885 ms | ~885 ms | Claude Code hook (no cache) |

The MCP server pays 248 ms once at start, then 4 ms per call — `route()` warm
is 1.7% of delivered, and the server is the recommended path where a persistent
process is available.

## Evaluation harness

Measure routing quality against labelled ground truth:

```bash
python -m oiax.eval.route_eval score ./policies/ < labelled.jsonl   # shipped config
python -m oiax.eval.route_eval sweep ./policies/ < labelled.jsonl   # the full grid
```

The labelled file is JSONL — one JSON object per line:

```json
{"prompt": "How do I deploy to production?", "expected": ["deploy-policy"]}
{"prompt": "What's for lunch?", "expected": []}
```

Reported: `recall@2`, `precision`, `F1`, `top-1 accuracy`, and the false-alarm rate over
negative prompts (`"expected": []`). Read precision against the two-hit cap — with one
expected label it cannot exceed 0.5 for that prompt.

Two corpora ship at `oiax/eval/corpora/`: a 15-document **reference** corpus with 52
labelled prompts (the calibration set — recall@2 0.648, top-1 0.673, zero false alarms),
and a 5-document synthetic smoke corpus that is structurally useful and **cannot**
calibrate anything. Judge labels are evidence, not proof — hand-check a slice before
treating any rate as authoritative.

### Scores on a corpus this project did not write

The numbers above come from a corpus, a prompt set and a labelling with **one
author — the author of the design they support.** That is a regression surface, not
evidence. `oiax.eval.benchmarks` maps public benchmarks onto the same interfaces so a
score lands somewhere independent. Nothing is vendored; the data is fetched to a cache
directory you name.

Against **[SkillRet](https://huggingface.co/datasets/ThakiCloud/SKILLRET)** — 6,006
real skills scraped from public GitHub repos, 4,392 queries, measured 2026-08-03:

| documents | recall@2 | top-1 |
|---:|---:|---:|
| 200 | 0.828 | 0.787 |
| 1,000 | 0.657 | 0.630 |
| 6,006 | **0.407** | **0.547** |

**Recall@2 halves between 200 and 6,006 documents.** The reference-corpus figure does
not describe behaviour at scale. And the hybrid earns its place where it matters:
against lexical-only on the full corpus it is **+7.2 pp on top-1** and only +1.4 pp on
recall@2 — both scorers usually get the right document into the top two, and fusing
them is what puts it first.

Full tables, the ablations, and the bounds that limit all of it —
no negatives in the benchmark, LLM-generated queries, and a material substitution of
`description` for an authored routing surface — are in
[`src/oiax/eval/corpora/README.md`](src/oiax/eval/corpora/README.md).

## Calibration — the shipped floors are one corpus's answer

`oiax`'s selection floors were measured against its reference corpus: **15
documents, 52 labelled prompts, one author.** They are not universal properties
of the algorithm — they are the point where *that* corpus's score distribution
separated signal from noise.

So the package tells you where they came from, lets you compute your own, and
says when it is running far from either.

### Where the defaults came from

```python
>>> from oiax.calibration import SHIPPED
>>> print(SHIPPED.describe())
lex=0.1 sem=0.25 rrf_k=60 top_k=2 — measured 2026-08-03 on oiax reference-policies
(15 documents) under sentence-transformers/all-MiniLM-L6-v2
```

### Calibrate against your own corpus

```bash
python -m oiax.eval.route_eval calibrate ./policies/ \
    --out ./oiax-operating-point.json --corpus-id "acme policies" \
    < ./labelled.jsonl
```

It prints the whole grid — **including the rows that lost**, because a table
showing only the winner hides what it was chosen over — and writes the winner as
a loadable operating point. The selection rule is stated rather than implied:

1. **Zero false alarms is a hard gate**, not a tiebreak. A configuration that
   routes a prompt no document governs is excluded whatever else it scores.
2. Among survivors, highest F1.
3. Ties go to the **quieter** point.

If nothing clears the gate, that is a finding and the command says so: fix the
corpus, not the floors.

```bash
python3 -m oiax.adapters.claude_code ./policies/ --operating-point ./oiax-operating-point.json
```

A bad path is an **error**, not a silent fallback to the shipped numbers — you
passed it because you meant to use it.

### No labels? Still supported

Run with the shipped defaults. They are the honest starting point for a corpus
nobody has calibrated, and the divergence signal below will tell you when they
have stopped applying.

### The divergence signal

At index build, `oiax` compares your corpus against the one the operating point
was measured on — document count, separability, and the embedding model — and
renders any mismatch **into the context paragraph the agent reads**, not into a
log:

```
⚠ oiax is running far from its calibration — the selection floors in force were
measured on a different corpus, so recall and precision here are unmeasured.
Reasons:
  - This corpus separates at 0.22; the operating point was calibrated at 0.55.
    The score distribution here is materially different.
```

Same reasoning as the lexical-only notice: the layer is running, and its numbers
do not mean what its documentation says. The thresholds are deliberately crude
and wide — a divergence detector that fires constantly gets ignored exactly like
a false-positive route does.

## Telemetry — is it working?

`oiax` can report on itself. Off by default; one environment variable turns it on:

```bash
export OIAX_TELEMETRY_PATH=~/.oiax/events.jsonl
```

Every route attempt appends one JSON object — **outcome, failure class, degraded
flag, corpus size, delivered latency, warm route latency, and the document names
returned.** Then:

```bash
python -m oiax.eval.telemetry_report ~/.oiax/events.jsonl --corpus-dir ./policies/
```

```
oiax telemetry — 4 route attempt(s)

  routed          1  (25.0%)
  abstained       2  (50.0%)
  failed          1  (25.0%)
      input          1
  degraded        0  (0.0%)  lexical-only

  delivered  p50 234 ms   p99 236 ms
  warm route p50 4 ms
             warm route is 1.7% of delivered

  documents routed at least once: 2/15
  NEVER ROUTED — these routing surfaces are not discriminating:
      access-control-policy
      ...
```

**Why this exists.** Before it, a turn that produced no routes because the router
crashed and a turn that produced no routes because nothing applied were *the same
observation*. A routing layer that had stopped working looked exactly like one
being appropriately quiet — which is how a version that named an embedding model
the provider does not publish shipped for four days, silently lexical-only,
warning to a stderr the reference deployment discards.

Four properties:

- **Failure is distinguishable from abstention.** A failed event names its class
  (`input`, `corpus_load`, `index_build`, `route`, `render`). "No routes" is not
  one state.
- **The delivered path is what gets timed**, with the warm route printed beside it
  and as a percentage of it. The two are routinely quoted apart, and the ratio is
  the point.
- **Documents that never route are named.** That list is the actionable half of
  routing quality — a document that never routes has a routing surface that is not
  discriminating.
- **Nothing sensitive is recorded, ever.** No prompt text, no corpus path, no file
  name. Prompts routinely carry credentials; a telemetry file accumulating them
  would be a disclosure hole opened by the observability layer itself.

Telemetry **never costs a turn**: the sink is a no-op unless you install one, a
broken sink is swallowed, and a routing failure still exits clean. To collect
events in-process instead of to a file, `oiax.telemetry.set_sink()` takes any
object with a `write(event)` method — an explicit sink is never overruled by the
environment variable.

## API

### `oiax.router`

| Callable | Signature | Returns |
|---|---|---|
| `route` | `route(prompt: str, index: Index | None = None) -> list[RouteHit]` | Scored hits |
| `build_index` | `build_index(corpus, *, operating_point, lex_threshold, sem_threshold, rrf_k, top_k) -> Index` | Built index |
| `semantic_ready` | `semantic_ready() -> bool` | `False` when the embedding model failed to load and routing is lexical-only — surface it, do not swallow it |
| `set_embedder` | `set_embedder(embedder: Embedder \| None) -> None` | Install a different embedding provider. `None` restores the default |

### Swapping the embedding provider

One module names a provider. `oiax.embedding` holds the `Embedder` protocol and
the shipped local-ONNX adapter; the router, the corpus loader and every harness
adapter address the protocol and nothing else — asserted by
`tests/test_embedding.py::test_no_module_outside_the_adapter_imports_the_provider`.

```python
from oiax.embedding import set_embedder

class MyEmbedder:
    def embed(self, texts: list[str]): ...   # float32[n, d], L2-normalised
    def model_id(self) -> str: ...           # enters the calibration provenance
    def dimension(self) -> int: ...
    def ready(self) -> bool: ...             # honest, and triggers the load

set_embedder(MyEmbedder())
```

**L2 normalisation is part of the contract**, not an implementation detail: a
consumer that assumes cosine is a dot product and an adapter that does not
honour it fail silently, and only on some corpora.

**A provider swap needs a recalibration.** Cosine distributions are not
comparable between models, so the shipped floors were measured on a system that
is not yours — run `route_eval calibrate` and the divergence signal will tell you
when they have stopped applying.

### `RouteHit`

```python
@dataclass(frozen=True)
class RouteHit:
    name: str       # document name (surface name only, never body text)
    score: float    # best RAW scorer score, [0, 1] — hits are ORDERED by fusion, not by this
    why: list[str]  # matched terms, and/or "semantic match"
```

### `oiax.corpus`

| Class | Purpose |
|---|---|
| `Document(name, trigger_line, body)` | One document in the routing corpus |
| `Corpus` (Protocol) | Any object with `.documents() -> Iterator[Document]` |
| `PolicyDirCorpus(path)` | Reads `*.md` files with `**Agent-trigger:**` headers |

### `oiax.adapters`

| Module | Purpose |
|---|---|
| `claude_code.py` | UserPromptSubmit hook adapter |
| `mcp.py` | MCP server (`oiax-mcp`) — `route_policies` + `get_policy` over stdio |
| `stdout.py` | Debug adapter — prints hits as text |

## When you need oiax

You need oiax when your rule corpus is too large to inject into every context (context-window pressure, attention dilution) and too important to leave to the agent's judgment (silent policy violations).

You do **not** need oiax when your corpus fits in a single `CLAUDE.md` — static injection is free and optimal for that case.

## Development

```bash
git clone https://github.com/nousergon/oiax.git
cd oiax
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                          # test suite
ruff check src/ tests/          # lint
mypy src/oiax                   # type check
```

All three run in CI on Python 3.11, 3.12 and 3.13 and are required to merge. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
