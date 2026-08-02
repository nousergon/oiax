# oiax

> **oiax** — semantic policy routing for agent fleets. The tiller that delivers the right governance context to the right agent at the right turn, by meaning.

[![CI](https://github.com/nousergon/oiax/actions/workflows/ci.yml/badge.svg)](https://github.com/nousergon/oiax/actions/workflows/ci.yml)

oiax routes a free-text prompt against a governance corpus and delivers the policies that bear on that turn — by meaning, before the agent decides, at ~6ms with no network call.

**The retrieval design is for normative text** (policies, coding standards, ADRs, compliance rules), not general knowledge. Four decisions make it correct:

1. **Whole-document delivery, never chunks.** A rule and its carve-out are semantically distant but logically inseparable.
2. **Precision over recall, asymmetric errors.** A miss degrades to the status quo; a false positive actively degrades the layer.
3. **Surface names, never rules.** Matched terms make a bad match dismissible at a glance.
4. **Runtime-agnostic core, harness-specific adapters.** The router returns structured hits; each harness gets its own thin delivery layer.

## Installation

```bash
pip install oiax
```

Requires Python ≥ 3.11. On first use, a ~90MB ONNX embedding model downloads and caches locally. Subsequent routes are ~6ms.

## Quick start

### Route a prompt

```python
from oiax import build_index, route
from oiax.corpus import PolicyDirCorpus

# Load from a directory of markdown files with **Agent-trigger:** headers
# (the nous-ergon-ops policy directory convention)
corpus = PolicyDirCorpus("./policies/")
index = build_index(corpus)

# Route a prompt — returns hits sorted by score descending
hits = route("How do I deploy to production?", index)
for hit in hits:
    print(f"{hit.name} ({hit.score:.2f}): {', '.join(hit.why)}")
```

### Route with query expansions

```python
import json

expansions = json.load(open("./routing-expansions.json"))
index = build_index(corpus, expansions=expansions)
hits = route("help me merge my PR", index)
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
          "command": "python3 -m oiax.adapters.claude_code /path/to/policies/ --expansions /path/to/expansions.json",
          "timeout": 8
        }]
      }
    ]
  }
}
```

On every prompt, the adapter routes the prompt text against the policy corpus and injects a context paragraph naming the policies that may apply — with the matched terms, so a bad match is dismissible at a glance. Never blocks: any error exits 0 silently.

## Evaluation harness

Measure recall and precision against labelled ground truth:

```bash
python -m oiax.eval.route_eval score ./policies/ < labelled.jsonl
```

The labelled file is JSONL — one JSON object per line:

```json
{"prompt": "How do I deploy to production?", "expected": ["deploy-policy"]}
{"prompt": "What's for lunch?", "expected": []}
```

A synthetic labelled corpus ships at `oiax/eval/corpora/`. Judge labels are evidence, not proof — hand-check a slice before treating the rate as authoritative.

## API

### `oiax.router`

| Callable | Signature | Returns |
|---|---|---|
| `route` | `route(prompt: str, index: Index | None = None) -> list[RouteHit]` | Scored hits |
| `build_index` | `build_index(corpus: Corpus, *, expansions, lex_threshold, sem_threshold) -> Index` | Built index |

### `RouteHit`

```python
@dataclass(frozen=True)
class RouteHit:
    name: str       # document name (surface name only, never body text)
    score: float    # [0, 1]
    why: list[str]  # matched terms/segments
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
| `stdout.py` | Debug adapter — prints hits as text |

## When you need oiax

You need oiax when your rule corpus is too large to inject into every context (context-window pressure, attention dilution) and too important to leave to the agent's judgment (silent policy violations).

You do **not** need oiax when your corpus fits in a single `CLAUDE.md` — static injection is free and optimal for that case.

For the full positioning, design rationale, and competitive landscape, see the [positioning doc](https://github.com/nousergon/nous-ergon-ops/blob/main/business/product-positioning/oiax.md).

## License

AGPL-3.0 — see [LICENSE](LICENSE).
