# Dependency Policy

**Agent-trigger:** Adding, upgrading and pinning third-party dependencies: version pinning, upgrade cadence, licence compatibility, and responding to a vulnerable transitive package. Load before adding a dependency, when an automated upgrade lands, when a licence is unclear, and when a scanner flags a package.

Every dependency is pinned to an exact version in a committed lockfile. A range in a lockfile is not a
pin.

Automated upgrade proposals run weekly and merge on green for patch and minor versions. A major version
is reviewed by a person.

A dependency whose licence is incompatible with the project's own licence is not added regardless of how
convenient it is. A vulnerable transitive package is upgraded, replaced, or explicitly accepted with a
written expiry.
