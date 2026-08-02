# Labelled evaluation corpora

Synthetic labelled examples for evaluating the router's recall and
precision. Each line is a JSON object:

```json
{"prompt": "free-text prompt", "expected": ["policy-slug-a", "policy-slug-b"]}
```

- `prompt` — the user's prompt text
- `expected` — which policy document names (slugs) SHOULD be routed.
  Empty list means the prompt should route to nothing.

**These are synthetic — they demonstrate the methodology without leaking
NE policy text.** The real NE labelled set remains private per
`repository-tiering-policy.md`.

**The acceptance bar:** 0 false positives on the negative set (prompts
with empty `expected`). A false positive degrades the reminder layer —
a policy surfaced when it shouldn't be gets tuned out.
