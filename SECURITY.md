# Security Policy

## Reporting a vulnerability

If you find a security vulnerability in oiax, please report it privately:

- **Preferred:** open a [GitHub Security Advisory](https://github.com/nousergon/oiax/security/advisories/new). This keeps the discussion private until a fix ships.
- **Alternative:** email `security@nousergon.ai` with a description and reproduction steps.

Please **do not** open a public issue for security reports. I aim to acknowledge within 72 hours and ship a fix or mitigation within 14 days for high-severity issues.

## Scope

oiax is a semantic policy routing library installed via `pip`. The following are in scope:

- **Policy content leakage**: any path that exposes a routed policy body in a way that bypasses the "surface names, never rules" invariant.
- **Injection**: prompt injection that causes oiax to route to a malicious policy or expose a policy body the prompt shouldn't see.
- **Dependency supply chain**: vulnerabilities in pinned or declared dependencies that oiax exposes through its public API.

The following are **out of scope**:

- Vulnerabilities in upstream dependencies that have not been disclosed publicly — please report those to the upstream project first.
- Issues that require local filesystem access (oiax is a library; if your machine is compromised, oiax's threat model has already failed).
- DoS via unbounded memory consumption from an oversized corpus — oiax provides configuration knobs; set them to match your environment.

## Supported versions

Only the latest published version receives security patches. Pin to a specific version and monitor releases.
