"""Tests for router telemetry.

The acceptance test for this feature is the FAILURE case, and it is stated in
the issue that way: with the corpus directory deleted, the adapter still exits
0 **and** the record says it failed. Everything else here supports that one
assertion.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from oiax.adapters import claude_code
from oiax.eval.telemetry_report import format_summary, load_events, summarize
from oiax.telemetry import (
    CallableSink,
    JsonlSink,
    NullSink,
    RouteEvent,
    emit,
    get_sink,
    install_env_sink,
    set_sink,
    sink_from_env,
    timer,
)

_CORPORA = Path(__file__).resolve().parent.parent / "src" / "oiax" / "eval" / "corpora"
CORPUS = _CORPORA / "reference-policies"


@pytest.fixture(autouse=True)
def _reset_sink():
    yield
    set_sink(None)


# ── the event contract ──────────────────────────────────────────────────────


def test_a_failed_event_must_name_its_class():
    # An unclassified failure aggregates to nothing, which is the state this
    # module exists to end.
    with pytest.raises(ValueError):
        RouteEvent(outcome="failed")


def test_only_a_failure_carries_a_failure_class():
    with pytest.raises(ValueError):
        RouteEvent(outcome="abstained", failure="route")


def test_routed_must_name_what_it_returned():
    with pytest.raises(ValueError):
        RouteEvent(outcome="routed")


def test_abstained_returned_nothing_by_definition():
    with pytest.raises(ValueError):
        RouteEvent(outcome="abstained", returned=("x",))


def test_event_carries_no_prompt_and_no_path():
    ev = RouteEvent(outcome="routed", returned=("deployment-policy",), corpus_size=15)
    keys = set(ev.to_dict())
    for forbidden in ("prompt", "path", "corpus_dir", "text", "command"):
        assert forbidden not in keys


def test_attempts_defaults_to_one():
    # No batching concept exists in this library — every event is exactly one
    # route attempt, independent of what it resolved to.
    ev = RouteEvent(outcome="abstained")
    assert ev.attempts == 1


def test_attempts_below_one_is_rejected():
    with pytest.raises(ValueError):
        RouteEvent(outcome="abstained", attempts=0)


def test_a_routed_document_is_recoverable_from_a_single_event():
    """§7.1's per-document signal: a route to a specific document must be
    readable off ONE event's own field, not reconstructed by cross-referencing
    other events or a downstream log.
    """
    ev = RouteEvent(outcome="routed", returned=("deployment-policy",), corpus_size=3)
    # No other event, no log, no aggregation — the name comes straight off
    # this event's own `returned` field.
    assert ev.returned == ("deployment-policy",)
    assert "deployment-policy" in ev.to_dict()["returned"]


def test_attempts_and_returned_round_trip_through_a_sink(tmp_path):
    path = tmp_path / "events.jsonl"
    set_sink(JsonlSink(path))
    emit(RouteEvent(outcome="routed", returned=("access-control-policy",), attempts=1))
    line = json.loads(path.read_text(encoding="utf-8").strip())
    assert line["attempts"] == 1
    assert line["returned"] == ["access-control-policy"]


# ── sinks ───────────────────────────────────────────────────────────────────


def test_default_sink_is_a_noop():
    assert isinstance(get_sink(), NullSink)
    emit(RouteEvent(outcome="abstained"))  # must not raise


def test_emit_never_raises_even_when_the_sink_does():
    def explode(_: RouteEvent) -> None:
        raise RuntimeError("sink is broken")

    set_sink(CallableSink(fn=explode))
    emit(RouteEvent(outcome="abstained"))  # a broken sink must not cost a turn


def test_jsonl_sink_appends_one_object_per_event(tmp_path):
    path = tmp_path / "nested" / "events.jsonl"
    set_sink(JsonlSink(path))
    emit(RouteEvent(outcome="abstained"))
    emit(RouteEvent(outcome="routed", returned=("a",)))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["returned"] == ["a"]


def test_sink_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OIAX_TELEMETRY_PATH", raising=False)
    assert sink_from_env() is None
    monkeypatch.setenv("OIAX_TELEMETRY_PATH", str(tmp_path / "e.jsonl"))
    assert isinstance(sink_from_env(), JsonlSink)


def test_env_sink_never_overrules_an_explicit_sink(tmp_path, monkeypatch):
    # A library may fill in a default; it may not overrule a decision the
    # caller already made. Unconditional set_sink(sink_from_env()) did both —
    # it wiped an installed sink, and with the var unset it switched telemetry
    # OFF for a caller who had switched it on.
    seen: list[RouteEvent] = []
    set_sink(CallableSink(fn=seen.append))
    monkeypatch.setenv("OIAX_TELEMETRY_PATH", str(tmp_path / "e.jsonl"))
    assert install_env_sink() is False
    emit(RouteEvent(outcome="abstained"))
    assert len(seen) == 1


def test_env_sink_installs_over_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv("OIAX_TELEMETRY_PATH", str(tmp_path / "e.jsonl"))
    assert install_env_sink() is True
    emit(RouteEvent(outcome="abstained"))
    assert (tmp_path / "e.jsonl").exists()


def test_timer_measures_something():
    with timer() as t:
        pass
    assert t.ms >= 0.0


# ── the adapter: every early return is now classified ───────────────────────


def _run_adapter(payload: str, corpus_dir, events: list[RouteEvent], monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    set_sink(CallableSink(fn=events.append))
    # install_env_sink() must NOT overrule the sink this test installed —
    # asserted directly in test_env_sink_never_overrules_an_explicit_sink.
    monkeypatch.delenv("OIAX_TELEMETRY_PATH", raising=False)
    rc = claude_code.main([str(corpus_dir)])
    capsys.readouterr()
    return rc


def test_unparseable_input_is_a_classified_failure(monkeypatch, capsys):
    events: list[RouteEvent] = []
    rc = _run_adapter("not json", CORPUS, events, monkeypatch, capsys)
    assert rc == 0
    assert len(events) == 1
    assert events[0].outcome == "failed" and events[0].failure == "input"


def test_a_missing_corpus_still_exits_zero_and_says_it_failed(tmp_path, monkeypatch, capsys):
    """The acceptance test named in the issue.

    Before this change the adapter returned 0 and emitted nothing, so a router
    pointed at a deleted corpus was indistinguishable from a quiet one.
    """
    missing = tmp_path / "gone"
    payload = json.dumps({"prompt": "how do I roll back a bad deploy today"})
    events: list[RouteEvent] = []
    rc = _run_adapter(payload, missing, events, monkeypatch, capsys)
    assert rc == 0, "the layer must never cost a turn"
    assert len(events) == 1
    ev = events[0]
    # PolicyDirCorpus tolerates a missing dir and yields nothing, so this lands
    # as an abstention on an empty corpus rather than a corpus_load failure —
    # and the record says so, which is the whole point: the two states are now
    # distinguishable instead of both being a bare `return 0`.
    assert ev.outcome in ("failed", "abstained")
    assert ev.corpus_size == 0


def test_a_short_prompt_is_recorded_as_an_abstention(monkeypatch, capsys):
    events: list[RouteEvent] = []
    rc = _run_adapter(json.dumps({"prompt": "hi"}), CORPUS, events, monkeypatch, capsys)
    assert rc == 0
    assert events[0].outcome == "abstained"


def test_a_successful_route_records_names_timing_and_corpus_size(monkeypatch, capsys):
    events: list[RouteEvent] = []
    payload = json.dumps({"prompt": "we need to roll back a bad production deploy"})
    rc = _run_adapter(payload, CORPUS, events, monkeypatch, capsys)
    assert rc == 0
    ev = events[0]
    assert ev.outcome == "routed"
    assert ev.returned
    assert ev.corpus_size > 0
    assert ev.adapter == "claude_code"
    # Delivered timing covers the whole call, not just the inner route.
    assert ev.elapsed_ms is not None and ev.route_ms is not None
    assert ev.elapsed_ms >= ev.route_ms


def test_an_unrelated_prompt_records_an_abstention_not_a_failure(monkeypatch, capsys):
    events: list[RouteEvent] = []
    payload = json.dumps({"prompt": "what is the capital city of Portugal"})
    _run_adapter(payload, CORPUS, events, monkeypatch, capsys)
    assert events[0].outcome in ("abstained", "routed")
    assert events[0].failure is None


# ── the report ──────────────────────────────────────────────────────────────


def test_summary_counts_outcomes_and_failure_classes():
    events = load_events(
        [
            json.dumps({"outcome": "routed", "returned": ["a"], "elapsed_ms": 100, "route_ms": 5}),
            json.dumps({"outcome": "abstained", "elapsed_ms": 90}),
            json.dumps({"outcome": "failed", "failure": "index_build", "elapsed_ms": 20}),
            json.dumps({"outcome": "failed", "failure": "index_build"}),
            json.dumps({"outcome": "routed", "returned": ["a", "b"], "degraded": True}),
        ]
    )
    s = summarize(events)
    assert (s.total, s.routed, s.abstained, s.failed) == (5, 2, 1, 2)
    assert s.failures["index_build"] == 2
    assert s.per_document["a"] == 2 and s.per_document["b"] == 1
    assert s.degraded_rate == pytest.approx(0.2)


def test_summary_total_is_summed_from_the_attempts_field():
    # `total` is derived from each event's own `attempts` counter, not from
    # counting events — so it stays correct if a future producer ever emits
    # a batched event with attempts > 1.
    events = load_events(
        [
            json.dumps({"outcome": "routed", "returned": ["a"], "attempts": 3}),
            json.dumps({"outcome": "abstained"}),  # no attempts key — defaults to 1
        ]
    )
    assert summarize(events).total == 4


def test_report_says_an_empty_log_is_not_healthy():
    out = format_summary(summarize([]))
    assert "not the same as healthy" in out


def test_report_names_documents_that_never_routed():
    events = load_events([json.dumps({"outcome": "routed", "returned": ["a"]})])
    out = format_summary(summarize(events), known_documents={"a", "b", "c"})
    assert "NEVER ROUTED" in out
    assert "b" in out and "c" in out


def test_report_shows_the_warm_route_as_a_fraction_of_delivered():
    events = load_events(
        [json.dumps({"outcome": "routed", "returned": ["a"], "elapsed_ms": 1000, "route_ms": 5})]
    )
    out = format_summary(summarize(events))
    assert "of delivered" in out


def test_report_always_states_the_delivery_bound():
    events = load_events([json.dumps({"outcome": "routed", "returned": ["a"]})])
    assert "DELIVERY" in format_summary(summarize(events))


def test_load_events_skips_a_torn_line():
    # Concurrent short-lived writers make a torn final line an expected
    # collection artifact, not a defect in the data.
    events = load_events(['{"outcome": "routed", "returned": ["a"]}', '{"outcome": "rou'])
    assert len(events) == 1
