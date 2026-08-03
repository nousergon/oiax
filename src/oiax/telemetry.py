"""What the router says about itself.

Before this module, `oiax` emitted nothing. The Claude Code adapter returned
``0`` on every error path — unparseable stdin, corpus load failure, index
build failure, route failure — and wrote nothing; the router logged two
warnings to a logger nobody configures. So **a turn that produced no routes
because the router crashed and a turn that produced no routes because nothing
applied were the same observation at the read side.** A routing layer that has
stopped working looked exactly like one being appropriately quiet.

That is not hypothetical for this package. 0.1.0-0.1.1 named an embedding
model the provider does not publish; every install raised at load, fell back
to lexical-only, and shipped that way for four days. The only surface that
knew was a ``logging.warning`` on a path the reference deployment invokes with
``2>/dev/null``.

**Four properties, each load-bearing:**

- **Failure is distinguishable from abstention.** :class:`RouteEvent` carries
  an ``outcome`` and, when it failed, a ``failure`` CLASS. "No routes" is not
  one state.
- **The sink is the caller's choice.** Default is a no-op. A library that
  decides where telemetry goes has taken an environment assumption into a core
  that is otherwise path-free — the same boundary ``PolicyDirCorpus`` exists to
  keep.
- **Emission never costs the caller anything.** :func:`emit` swallows every
  exception from the sink, by design and against this project's usual
  fail-loud default. Rationale, per the deviation rule: the failure mode
  swallowed is a broken sink; the primary deliverable (the route) survives
  because emission happens after it; and the recording surface is the sink
  itself, which is the thing that just failed. A telemetry layer that can
  break a turn is worse than no telemetry layer.
- **Nothing sensitive is recorded, ever.** No prompt text, no corpus path, no
  file name. Prompts routinely carry credentials; a telemetry file
  accumulating them would be a disclosure hole opened by the observability
  layer itself. Document NAMES and counts only — those are already returned to
  the caller, so they leak nothing the route did not.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

__all__ = [
    "RouteEvent",
    "Sink",
    "NullSink",
    "JsonlSink",
    "CallableSink",
    "emit",
    "get_sink",
    "install_env_sink",
    "set_sink",
    "timer",
]

Outcome = Literal["routed", "abstained", "failed"]

# Failure classes. A closed vocabulary rather than free text: an aggregate over
# free-text reasons cannot be counted, and counting is the point.
Failure = Literal[
    "input",        # the caller's input was unusable (unparseable, too short)
    "corpus_load",  # the corpus could not be read
    "index_build",  # the index could not be built
    "route",        # scoring or fusion raised
    "render",       # the hits could not be rendered for delivery
]


@dataclass(frozen=True)
class RouteEvent:
    """One route attempt, from the delivery surface's point of view.

    ``elapsed_ms`` is the WHOLE DELIVERED PATH — process start to emitted
    output where the adapter is a subprocess — not the warm inner call. The
    distinction is the difference between an advertised number and a true one:
    the warm route is single-digit milliseconds while a per-turn subprocess
    pays imports, model load and index build every turn, and quoting the former
    describes a fraction of a percent of what a turn actually costs.
    """

    outcome: Outcome
    returned: tuple[str, ...] = ()      # document names — never bodies, never prompts
    corpus_size: int = 0
    degraded: bool = False              # lexical-only: the embedder did not load
    failure: Failure | None = None
    elapsed_ms: float | None = None     # delivered path
    route_ms: float | None = None       # warm inner call, when measured separately
    adapter: str = ""                   # which delivery surface emitted this
    operating_point: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome == "failed" and self.failure is None:
            raise ValueError(
                "a failed RouteEvent must name its failure class — an unclassified "
                "failure aggregates to nothing, which is the state this module exists "
                "to end"
            )
        if self.outcome != "failed" and self.failure is not None:
            raise ValueError("only a failed RouteEvent carries a failure class")
        if self.outcome == "routed" and not self.returned:
            raise ValueError("a routed RouteEvent names the documents it returned")
        if self.outcome == "abstained" and self.returned:
            raise ValueError("an abstained RouteEvent returned nothing by definition")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["returned"] = list(self.returned)
        return d


class Sink(Protocol):
    """Somewhere an event goes. Implement one line."""

    def write(self, event: RouteEvent) -> None: ...


class NullSink:
    """The default. Emits nothing, costs nothing, breaks nothing."""

    def write(self, event: RouteEvent) -> None:
        return None


@dataclass
class CallableSink:
    """Adapts any ``(RouteEvent) -> None`` callable into a sink."""

    fn: Callable[[RouteEvent], None]

    def write(self, event: RouteEvent) -> None:
        self.fn(event)


@dataclass
class JsonlSink:
    """Append one JSON object per event to a file.

    Append-only and opened per write: the delivery surface is frequently a
    short-lived subprocess, so a held handle would be a handle held across a
    process that is about to exit. Concurrent appends of a single short line
    are atomic enough for this purpose on every platform the package targets;
    a sink needing stronger guarantees is the caller's to supply.
    """

    path: str | Path

    def write(self, event: RouteEvent) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict(), separators=(",", ":")) + "\n")


_SINK: Sink = NullSink()


def set_sink(sink: Sink | None) -> None:
    """Install the process-wide sink. ``None`` restores the no-op default."""
    global _SINK
    _SINK = sink if sink is not None else NullSink()


def get_sink() -> Sink:
    return _SINK


def emit(event: RouteEvent) -> None:
    """Send an event to the installed sink. **Never raises.**

    A deliberate deviation from the fail-loud default, with the three things
    that deviation requires named in this module's docstring. The short form:
    a telemetry layer that can break a turn is worse than no telemetry layer,
    and the only failure being swallowed here is the sink's own.
    """
    try:
        _SINK.write(event)
    except Exception:  # noqa: BLE001 — see the rationale above
        pass


def sink_from_env(var: str = "OIAX_TELEMETRY_PATH") -> Sink | None:
    """A :class:`JsonlSink` at ``$OIAX_TELEMETRY_PATH``, or None when unset.

    Environment-driven rather than a flag so a delivery surface the caller does
    not invoke directly — a hook registered in someone else's settings file —
    can still be instrumented without editing that file.
    """
    path = os.environ.get(var, "").strip()
    return JsonlSink(path) if path else None


def install_env_sink(var: str = "OIAX_TELEMETRY_PATH") -> bool:
    """Install the env-configured sink, but **never over an explicit one.**

    Returns True when it installed something.

    The guard is the point. An adapter that called ``set_sink(sink_from_env())``
    unconditionally would wipe a sink the embedding caller had already
    installed — and when the env var is unset it would replace that sink with
    the no-op default, silently switching telemetry off for a caller who had
    switched it on. A library may fill in a default; it may not overrule a
    decision the caller already made.
    """
    if not isinstance(_SINK, NullSink):
        return False
    sink = sink_from_env(var)
    if sink is None:
        return False
    set_sink(sink)
    return True


class timer:
    """Wall-clock context manager. ``timer().ms`` after exit.

    Deliberately trivial: the honesty in this module is about WHAT is timed
    (the delivered path) rather than about how precisely it is measured.
    """

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.ms: float = 0.0

    def __enter__(self) -> timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.ms = (time.perf_counter() - self._t0) * 1000
        return False
