# Api Versioning Policy

**Agent-trigger:** Changing a public API without breaking the people who depend on it: what counts as a breaking change, version negotiation, deprecation windows, and sunset notices. Load before changing any public endpoint, response shape, or default, and when planning to remove something clients still call.

Removing a field, narrowing an accepted input, changing a default, or changing an error code is a
breaking change. Adding an optional field is not.

Breaking changes ship behind a new major version. The previous major stays available for twelve months
from the day its successor becomes generally available.

Deprecation is announced in the changelog, in the response headers of the deprecated route, and directly
to identifiable callers. A sunset that a caller first learns about from a failed request is a defect in
this process.
