# Observability Policy

**Agent-trigger:** Making a running system legible: what every service must emit, how alerts are defined, service level objectives, and the rule that an absence of data is never healthy. Load when adding a service, defining an alert or dashboard, setting an SLO, and when a component reports nothing at all.

Every service emits request rate, error rate, latency distribution and saturation, with a dashboard that
a person who did not build it can read.

Alerts fire on symptoms a user would notice, not on causes. An alert nobody acts on is deleted rather
than muted.

No data is never rendered as healthy. A component that has stopped emitting is treated as failing until
proven otherwise, because a silent component and a healthy one look identical on every dashboard that
gets this wrong.
