# oiax

> **oiax** — semantic policy routing for agent fleets. The tiller that delivers the right governance context to the right agent at the right turn, by meaning.

[![CI](https://github.com/nousergon/oiax/actions/workflows/ci.yml/badge.svg)](https://github.com/nousergon/oiax/actions/workflows/ci.yml)

## What it does

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

## Quick start

```python
from oiax import build_index, route
from oiax.corpus import PolicyDirCorpus

# Load from a directory of markdown files with **Agent-trigger:** headers
corpus = PolicyDirCorpus("./my-policies/")
index = build_index(corpus)

# Route a prompt
hits = route("How do I deploy to production?", index)
for hit in hits:
    print(f"{hit.name} ({hit.score:.2f}): {', '.join(hit.why)}")
```

## When you need oiax

You need oiax when your rule corpus is too large to inject into every context (context-window pressure, attention dilution degrading output quality) and too important to leave to the agent's judgment (silent policy violations).

You do **not** need oiax when your corpus fits in a single `CLAUDE.md` — static injection is free and optimal for that case.

For the full positioning and design rationale, see the [positioning doc](https://github.com/nousergon/nous-ergon-ops/blob/main/business/product-positioning/oiax.md).

## License

AGPL-3.0 — see [LICENSE](LICENSE).
