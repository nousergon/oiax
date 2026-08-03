"""Tests for the outcome harness.

The load-bearing assertions here are not the arithmetic — they are the
REFUSALS. A harness that reports a confident two-arm verdict is the specific
way this measurement would over-claim, so the tests that matter are the ones
proving it declines.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from oiax import build_index
from oiax.corpus import PolicyDirCorpus
from oiax.eval.outcome_eval import (
    HARNESS_ARM,
    CallableArm,
    CheckError,
    NoRoutingArm,
    OiaxArm,
    TaskCase,
    format_report,
    load_tasks,
    run_check,
    run_outcome_eval,
)

CORPORA = Path(__file__).resolve().parent.parent / "src" / "oiax" / "eval" / "corpora"
REFERENCE_TASKS = CORPORA / "reference_tasks.jsonl"


# ── checks ──────────────────────────────────────────────────────────────────


def test_mentions_all_requires_every_term():
    spec = {"kind": "mentions_all", "terms": ["roll back", "canary"]}
    assert run_check("Roll back, then re-run the canary.", spec)
    assert not run_check("Roll back immediately.", spec)


def test_mentions_none_is_the_negative_form():
    spec = {"kind": "mentions_none", "terms": ["policy"]}
    assert run_check("Use a generator expression.", spec)
    assert not run_check("The policy says otherwise.", spec)


def test_checks_are_case_insensitive():
    assert run_check("ROLLBACK NOW", {"kind": "mentions_any", "terms": ["rollback"]})


def test_regex_check():
    assert run_check("within 5 working days", {"kind": "regex", "pattern": r"\d+ working days"})


def test_unknown_check_kind_raises_rather_than_failing_the_task():
    # A task that cannot be graded must not silently count as a failure: that
    # would understate every arm equally and still look like a real result.
    with pytest.raises(CheckError):
        run_check("anything", {"kind": "vibes"})


def test_malformed_check_spec_raises():
    with pytest.raises(CheckError):
        run_check("anything", {"kind": "mentions_all", "terms": []})


# ── task loading ────────────────────────────────────────────────────────────


def test_load_tasks_parses_jsonl():
    line = json.dumps(
        {"prompt": "p", "checks": [{"kind": "mentions_any", "terms": ["x"]}], "governing": ["d"]}
    )
    tasks = load_tasks(io.StringIO(line))
    assert len(tasks) == 1
    assert tasks[0].governing == ("d",)


def test_load_tasks_raises_on_a_bad_line_instead_of_skipping():
    # route_eval skips bad lines; here a dropped task silently shrinks the
    # denominator of the only number that says whether the layer is worth
    # running, so it must be loud.
    with pytest.raises(CheckError):
        load_tasks(io.StringIO('{"prompt": "p"}\n'))


def test_load_tasks_validates_check_specs_up_front():
    bad = json.dumps({"prompt": "p", "checks": [{"kind": "nope"}]})
    with pytest.raises(CheckError):
        load_tasks(io.StringIO(bad))


def test_load_tasks_rejects_an_empty_set():
    with pytest.raises(CheckError):
        load_tasks(io.StringIO("\n# comment only\n"))


def test_shipped_task_set_loads_and_carries_negatives():
    tasks = load_tasks(REFERENCE_TASKS.read_text(encoding="utf-8").splitlines())
    assert len(tasks) >= 6
    negatives = [t for t in tasks if not t.governing]
    assert negatives, (
        "the task set needs prompts no document governs — without them an arm "
        "that injects on every turn cannot be penalised"
    )


# ── arms ────────────────────────────────────────────────────────────────────


def test_no_routing_arm_injects_nothing():
    assert NoRoutingArm().context_for("anything") is None


def test_oiax_arm_renders_names_not_bodies():
    index = build_index(PolicyDirCorpus(CORPORA / "reference-policies"))
    ctx = OiaxArm(index=index).context_for("we need to roll back a bad deploy")
    assert ctx is not None
    # Surface names and matched terms only — an arm injecting document bodies
    # would be measuring a layer nobody runs.
    assert "deployment-policy" in ctx
    assert "canary that serves 5% of traffic" not in ctx


def test_oiax_arm_injects_nothing_when_nothing_routes():
    index = build_index(PolicyDirCorpus(CORPORA / "reference-policies"))
    assert OiaxArm(index=index).context_for("zzzz") is None


# ── the refusals ────────────────────────────────────────────────────────────


def _tasks():
    return [
        TaskCase(prompt="p1", checks=({"kind": "mentions_any", "terms": ["yes"]},)),
        TaskCase(prompt="p2", checks=({"kind": "mentions_any", "terms": ["yes"]},)),
    ]


def test_report_refuses_a_verdict_without_a_harness_arm():
    report = run_outcome_eval(
        _tasks(),
        [NoRoutingArm(), CallableArm(fn=lambda p: "ctx", name="oiax")],
        respond=lambda prompt, ctx: "yes" if ctx else "no",
    )
    assert not report.has_harness_arm
    assert report.verdict.startswith("NO VERDICT")
    assert HARNESS_ARM in report.verdict


def test_report_states_a_verdict_once_the_harness_arm_is_present():
    report = run_outcome_eval(
        _tasks(),
        [
            NoRoutingArm(),
            CallableArm(fn=lambda p: "harness ctx", name=HARNESS_ARM),
            CallableArm(fn=lambda p: "oiax ctx", name="oiax"),
        ],
        respond=lambda prompt, ctx: "yes" if ctx and "oiax" in ctx else "no",
    )
    assert report.has_harness_arm
    assert not report.verdict.startswith("NO VERDICT")
    assert "better than" in report.verdict


def test_delta_reports_quality_and_cost_together():
    report = run_outcome_eval(
        _tasks(),
        [
            CallableArm(fn=lambda p: "x" * 100, name=HARNESS_ARM),
            CallableArm(fn=lambda p: "x" * 300, name="oiax"),
        ],
        respond=lambda prompt, ctx: "yes",
    )
    delta = report.delta_vs(HARNESS_ARM, "oiax")
    assert delta is not None
    rate, cost = delta
    assert rate == 0.0
    # Equal quality at triple the payload is a REGRESSION, and the second half
    # of the tuple is the only thing that says so.
    assert cost == pytest.approx(200.0)


def test_report_warns_when_conditions_are_absent():
    report = run_outcome_eval(
        _tasks(), [NoRoutingArm()], respond=lambda prompt, ctx: "no"
    )
    assert "no conditions recorded" in format_report(report)


def test_report_prints_conditions_when_supplied():
    report = run_outcome_eval(
        _tasks(),
        [NoRoutingArm()],
        respond=lambda prompt, ctx: "no",
        conditions={"corpus": "reference-policies", "model": "some-model"},
    )
    out = format_report(report)
    assert "reference-policies" in out and "some-model" in out


def test_report_always_states_the_population_bound():
    report = run_outcome_eval(
        _tasks(),
        [CallableArm(fn=lambda p: "c", name=HARNESS_ARM), NoRoutingArm()],
        respond=lambda prompt, ctx: "yes",
    )
    out = format_report(report)
    assert "POPULATION" in out
    assert "rule was followed" in out


# ── counting ────────────────────────────────────────────────────────────────


def test_token_counting_is_optional_and_bytes_are_always_reported():
    report = run_outcome_eval(
        _tasks(),
        [CallableArm(fn=lambda p: "abcd", name="oiax")],
        respond=lambda prompt, ctx: "yes",
    )
    arm = report.by_name("oiax")
    assert arm is not None
    assert arm.context_bytes == 8  # 4 bytes x 2 tasks
    assert arm.context_tokens is None

    counted = run_outcome_eval(
        _tasks(),
        [CallableArm(fn=lambda p: "abcd", name="oiax")],
        respond=lambda prompt, ctx: "yes",
        count_tokens=lambda s: len(s) // 2,
    )
    assert counted.by_name("oiax").context_tokens == 4
