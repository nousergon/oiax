"""oiax — semantic policy routing for agent fleets.

Delivers the right governance context to the right agent at the right turn,
by meaning. Runtime-agnostic core; harness-specific adapters deliver into
each agent runtime.
"""

from importlib.metadata import PackageNotFoundError, version

from oiax.router import RouteHit, build_index, route, semantic_ready  # noqa: F401

# Single-sourced from pyproject.toml via the installed metadata: a hardcoded
# string sat at "0.2.0" through the 0.3.x releases.
try:
    __version__ = version("oiax")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"
__all__ = ["RouteHit", "build_index", "route", "semantic_ready"]
