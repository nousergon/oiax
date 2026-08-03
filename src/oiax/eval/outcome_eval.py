"""Outcome evaluation — does routing change what the agent does?

``route_eval`` measures whether the right document was *selected*. This
measures whether *delivering* it changes an outcome, which is a different
question and the only one an adopter actually has.

**Why this exists.** A retrieval layer can have excellent recall and change
nothing: the delivered artifact may not be acted on, the agent may already
have reached the same conclusion without it, or the retrieved document may
not have been the binding constraint on the turn. Recall, precision and
harmful-sibling rate are evidence that *retrieval* works. Citing them as
evidence that the *layer* works is the gap this module closes.

**The comparison that matters is not against nothing.** Every adopter's host
harness already selects context by some means — Claude Code loads skills on
the model's own judgment, Cursor model-judges its rules. "Better than off" is
a bar nobody cares about. "Better than what I already have" is the question,
so :class:`OutcomeReport` **refuses to state a verdict** unless a
host-harness arm was run. A two-arm result is a measurement, not an answer,
and this module will not let it read as one.

**No model dependency.** oiax takes no LLM client, here or anywhere: the
caller supplies a ``respond(prompt, context) -> str`` callable. That keeps
the package provider-free and makes the harness usable against any model,
which is also the only way a third arm driving someone else's harness can
exist at all.

**Batch only.** Nothing here runs on the prompt path.

Judge labels are evidence, not proof — and every check here is deterministic
precisely so that bound does not compound. See ``corpora/README.md`` for the
task set and the conditions any published result must carry.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from oiax.router import Index, RouteHit, route

# A responder takes the user prompt and the arm's injected context (None when
# the arm injects nothing) and returns the agent's response text.
Responder = Callable[[str, "str | None"], str]

# The arm name that makes a verdict possible. Named rather than positional so
# a caller cannot satisfy it by ordering.
HARNESS_ARM = "harness"


# ── checks ──────────────────────────────────────────────────────────────────
# Deterministic by construction. A model-graded check would import the judge's
# error into the one number this module exists to produce, and the whole point
# of an outcome measure is that it is harder to fool than a retrieval metric.


class CheckError(ValueError):
    """A task's check spec is malformed. Never swallowed — a task that cannot
    be graded must not silently count as a failure, which would understate
    every arm equally and look like a real result."""


def _check_mentions_all(response: str, spec: dict[str, Any]) -> bool:
    terms = spec.get("terms")
    if not isinstance(terms, list) or not terms:
        raise CheckError("`mentions_all` needs a non-empty `terms` list")
    low = response.lower()
    return all(str(t).lower() in low for t in terms)


def _check_mentions_any(response: str, spec: dict[str, Any]) -> bool:
    terms = spec.get("terms")
    if not isinstance(terms, list) or not terms:
        raise CheckError("`mentions_any` needs a non-empty `terms` list")
    low = response.lower()
    return any(str(t).lower() in low for t in terms)


def _check_mentions_none(response: str, spec: dict[str, Any]) -> bool:
    terms = spec.get("terms")
    if not isinstance(terms, list) or not terms:
        raise CheckError("`mentions_none` needs a non-empty `terms` list")
    low = response.lower()
    return not any(str(t).lower() in low for t in terms)


def _check_regex(response: str, spec: dict[str, Any]) -> bool:
    pattern = spec.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise CheckError("`regex` needs a `pattern` string")
    return re.search(pattern, response, re.IGNORECASE | re.MULTILINE) is not None


_CHECKS: dict[str, Callable[[str, dict[str, Any]], bool]] = {
    "mentions_all": _check_mentions_all,
    "mentions_any": _check_mentions_any,
    "mentions_none": _check_mentions_none,
    "regex": _check_regex,
}


def run_check(response: str, spec: dict[str, Any]) -> bool:
    """Apply one check spec to a response. Raises on a malformed spec."""
    if not isinstance(spec, dict):
        raise CheckError(f"check must be an object, got {type(spec).__name__}")
    kind = spec.get("kind")
    fn = _CHECKS.get(str(kind))
    if fn is None:
        raise CheckError(f"unknown check kind {kind!r}; known: {sorted(_CHECKS)}")
    return fn(response, spec)


# ── task set ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskCase:
    """One prompt whose response has a checkable property.

    ``checks`` are ALL required to pass. Splitting a task into several checks
    is preferred over one loose check: a task that passes for the wrong reason
    inflates every arm, and inflating every arm hides the delta this module
    reports.
    """

    prompt: str
    checks: tuple[dict[str, Any], ...]
    governing: tuple[str, ...] = ()  # document names expected to bear on it
    note: str = ""

    def grade(self, response: str) -> bool:
        return all(run_check(response, spec) for spec in self.checks)


def load_tasks(stream: Iterable[str] | None = None) -> list[TaskCase]:
    """Parse a task set from JSONL. One object per line::

        {"prompt": "...", "checks": [{"kind": "mentions_all", "terms": [...]}],
         "governing": ["deployment-policy"], "note": "..."}

    A malformed line raises rather than being skipped. ``route_eval`` skips bad
    lines because a dropped label costs a little recall precision; here a
    dropped task silently shrinks the denominator of the only number that says
    whether the layer is worth running.
    """
    src = stream if stream is not None else sys.stdin
    tasks: list[TaskCase] = []
    for lineno, line in enumerate(src, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CheckError(f"task set line {lineno}: not valid JSON — {exc}") from exc
        prompt = obj.get("prompt")
        checks = obj.get("checks")
        if not isinstance(prompt, str) or not prompt.strip():
            raise CheckError(f"task set line {lineno}: `prompt` must be a non-empty string")
        if not isinstance(checks, list) or not checks:
            raise CheckError(f"task set line {lineno}: `checks` must be a non-empty list")
        for spec in checks:
            run_check("", spec)  # validate the spec shape now, not mid-run
        tasks.append(
            TaskCase(
                prompt=prompt,
                checks=tuple(checks),
                governing=tuple(obj.get("governing") or ()),
                note=str(obj.get("note") or ""),
            )
        )
    if not tasks:
        raise CheckError("task set is empty — nothing to measure")
    return tasks


# ── arms ────────────────────────────────────────────────────────────────────


class Arm(Protocol):
    """One way of putting context in front of the agent.

    Three are meaningful and all three are first-class here, because a harness
    that made the third one special-cased or optional would let a two-arm
    result be presented as an answer.
    """

    name: str

    def context_for(self, prompt: str) -> str | None: ...


@dataclass
class NoRoutingArm:
    """The floor. Injects nothing."""

    name: str = "off"

    def context_for(self, prompt: str) -> str | None:
        return None


@dataclass
class OiaxArm:
    """Routing on: the names oiax selects, rendered as the adapters render them.

    Deliberately renders SURFACE NAMES and matched terms only, matching what
    the shipped adapters deliver. An arm that injected full document bodies
    would be measuring a layer nobody runs.
    """

    index: Index
    name: str = "oiax"

    def context_for(self, prompt: str) -> str | None:
        hits: list[RouteHit] = route(prompt, self.index)
        if not hits:
            return None
        lines = ["Possibly relevant standing documents:"]
        for hit in hits:
            terms = ", ".join(hit.why) if hit.why else "semantic match"
            lines.append(f"- {hit.name} (matched: {terms})")
        return "\n".join(lines)


@dataclass
class CallableArm:
    """An arm whose context comes from a caller-supplied function.

    This is how the host-harness arm is supplied: driving Claude Code, Cursor
    or Copilot is the caller's job, not this package's. Name it
    ``HARNESS_ARM`` for the report to treat it as the comparison baseline.
    """

    fn: Callable[[str], str | None]
    name: str = HARNESS_ARM

    def context_for(self, prompt: str) -> str | None:
        return self.fn(prompt)


# ── results ─────────────────────────────────────────────────────────────────


@dataclass
class ArmResult:
    """One arm's performance over the task set."""

    name: str
    n: int
    passed: int
    context_bytes: int
    context_tokens: int | None = None
    per_task: list[bool] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n if self.n else 0.0

    @property
    def bytes_per_task(self) -> float:
        return self.context_bytes / self.n if self.n else 0.0


@dataclass
class OutcomeReport:
    """Every arm's result, plus whether a verdict is available at all."""

    arms: list[ArmResult]
    conditions: dict[str, Any] = field(default_factory=dict)

    def by_name(self, name: str) -> ArmResult | None:
        return next((a for a in self.arms if a.name == name), None)

    @property
    def has_harness_arm(self) -> bool:
        return self.by_name(HARNESS_ARM) is not None

    def delta_vs(self, baseline: str, arm: str) -> tuple[float, float] | None:
        """(pass-rate delta, context-bytes-per-task delta) of ``arm`` over
        ``baseline``. Both halves, always — a quality gain paid for with a
        large per-turn payload is a trade, and reporting one number without
        the other is how this layer would over-claim."""
        b, a = self.by_name(baseline), self.by_name(arm)
        if b is None or a is None:
            return None
        return (a.pass_rate - b.pass_rate, a.bytes_per_task - b.bytes_per_task)

    @property
    def verdict(self) -> str:
        """A sentence, or an explicit refusal to give one.

        **The refusal is the feature.** Without a host-harness arm the result
        says only that routing beats injecting nothing, which is not the
        comparison an adopter is making — their harness already selects
        context for free.
        """
        if not self.has_harness_arm:
            return (
                "NO VERDICT — the host-harness arm was not run. These numbers "
                "compare routing against injecting nothing, which is not the "
                "comparison an adopter faces: their harness already selects "
                f"context. Supply a `{HARNESS_ARM}` arm for a verdict."
            )
        d = self.delta_vs(HARNESS_ARM, "oiax")
        if d is None:
            return f"NO VERDICT — no `oiax` arm to compare against `{HARNESS_ARM}`."
        rate, cost = d
        direction = "better than" if rate > 0 else ("worse than" if rate < 0 else "level with")
        return (
            f"oiax is {rate:+.1%} {direction} the host harness on outcome, "
            f"at {cost:+.0f} bytes of injected context per task."
        )


# ── runner ──────────────────────────────────────────────────────────────────


def run_outcome_eval(
    tasks: Sequence[TaskCase],
    arms: Sequence[Arm],
    respond: Responder,
    *,
    count_tokens: Callable[[str], int] | None = None,
    conditions: dict[str, Any] | None = None,
) -> OutcomeReport:
    """Run every arm over every task and report per-arm outcomes.

    Args:
        tasks: the task set. Small and genuinely checkable beats large and
            model-graded — see this module's docstring.
        arms: the arms to compare. Include one named ``harness`` or the report
            will decline to state a verdict.
        respond: ``(prompt, context) -> response``. Supplied by the caller;
            oiax takes no model dependency.
        count_tokens: optional. Absent, cost is reported in BYTES, which is
            exact and provider-neutral; token counts are a provider's opinion
            about the same bytes.
        conditions: corpus, task-set and model identity. Recorded on the
            report because all three move the result, and a number published
            without them is not reproducible.

    Returns:
        An :class:`OutcomeReport`. Its ``verdict`` refuses to answer unless a
        host-harness arm was run.
    """
    results: list[ArmResult] = []
    for arm in arms:
        passed = 0
        ctx_bytes = 0
        ctx_tokens = 0 if count_tokens else None
        per_task: list[bool] = []
        for task in tasks:
            context = arm.context_for(task.prompt)
            if context:
                ctx_bytes += len(context.encode("utf-8"))
                if count_tokens is not None and ctx_tokens is not None:
                    ctx_tokens += count_tokens(context)
            response = respond(task.prompt, context)
            ok = task.grade(response)
            per_task.append(ok)
            passed += int(ok)
        results.append(
            ArmResult(
                name=arm.name,
                n=len(tasks),
                passed=passed,
                context_bytes=ctx_bytes,
                context_tokens=ctx_tokens,
                per_task=per_task,
            )
        )
    return OutcomeReport(arms=results, conditions=dict(conditions or {}))


def format_report(report: OutcomeReport) -> str:
    """Human-readable summary. Prints the conditions, because a result without
    its corpus, task set and model is not reproducible and must not be quoted."""
    lines = ["Outcome evaluation — does routing change what the agent does?", ""]
    lines.append(f"{'arm':<12} {'n':>4} {'pass':>6} {'rate':>8} {'ctx B/task':>12}")
    for a in report.arms:
        lines.append(
            f"{a.name:<12} {a.n:>4} {a.passed:>6} {a.pass_rate:>7.1%} {a.bytes_per_task:>12.0f}"
        )
    lines.append("")
    if report.conditions:
        lines.append("Conditions (all three move the result):")
        for k, v in sorted(report.conditions.items()):
            lines.append(f"  {k}: {v}")
        lines.append("")
    else:
        lines.append(
            "WARNING: no conditions recorded. A result without its corpus, task "
            "set and model is not reproducible and must not be published."
        )
        lines.append("")
    lines.append(report.verdict)
    lines.append("")
    lines.append(
        "Bound: this measures a POPULATION difference. It does not attribute "
        "any individual response to any delivered document, and no number here "
        "shows that a rule was followed."
    )
    return "\n".join(lines)
