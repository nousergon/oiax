# Model Usage Policy

**Agent-trigger:** Using large language models inside the product and internal tooling: choosing a model tier for a task, provider portability, prompt caching, and what may be sent to a third-party endpoint. Load when adding an LLM call site, choosing a model, changing a prompt's structure, or sending data outside the organisation's boundary.

Model choice is stated explicitly at every call site and matched to the task: a small model for
mechanical extraction, a mid model for scoped generation with clear acceptance criteria, a frontier model
for genuine ambiguity. Inheriting whatever model the caller happened to use is not a choice.

Call sites address a capability class through a router, never a hardcoded model identifier or vendor SDK
constructed in place. A provider-exclusive capability may be adopted for measuring better, never for
being available only there.

Prompts put the stable content first and the variable content last, so the cached prefix survives. No
customer data leaves the boundary without a recorded basis for sending it.
