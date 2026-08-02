"""Harness-specific delivery adapters.

The router core (`oiax.router`) is runtime-agnostic — it returns `RouteHit`
objects. Adapters render those hits into whatever format a specific agent
harness consumes.

Each adapter lives in its own module (e.g. `claude_code.py`). The
`DeliveryAdapter` Protocol defines the interface they all satisfy.
"""

from __future__ import annotations

from typing import Protocol

from oiax.router import RouteHit


class DeliveryAdapter(Protocol):
    """Renders route hits for a specific harness.

    Takes the hits the router produced and the original prompt, returns
    the output the harness expects (e.g. a JSON hook response).
    """

    def deliver(self, hits: list[RouteHit], prompt: str) -> str: ...
