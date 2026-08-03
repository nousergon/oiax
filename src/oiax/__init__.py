"""oiax — semantic policy routing for agent fleets.

Delivers the right governance context to the right agent at the right turn,
by meaning. Runtime-agnostic core; harness-specific adapters deliver into
each agent runtime.
"""

from oiax.router import RouteHit, build_index, route, semantic_ready  # noqa: F401

__version__ = "0.1.4"
__all__ = ["RouteHit", "build_index", "route", "semantic_ready"]
