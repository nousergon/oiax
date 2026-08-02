"""stdout adapter — prints route hits for interactive debugging.

Proves the adapter interface is harness-neutral: the router returns
``RouteHit`` objects; this adapter renders them to plain text.
"""

from oiax.router import RouteHit


def deliver(hits: list[RouteHit], prompt: str) -> str:
    """Render hits as human-readable text."""
    if not hits:
        return ""
    lines = [f"Routing results for: {prompt[:100]}...\n"]
    for hit in hits:
        lines.append(f"  {hit.name} ({hit.score:.2f})")
        if hit.why:
            lines.append(f"    matched: {', '.join(hit.why)}")
    return "\n".join(lines)
