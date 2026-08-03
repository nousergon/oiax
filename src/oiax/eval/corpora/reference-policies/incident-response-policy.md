# Incident Response Policy

**Agent-trigger:** Handling production outages and degradations: severity levels, who gets paged, the communication cadence during an incident, and the postmortem that must follow. Load when declaring an incident, when deciding its severity, and when writing up what happened afterwards.

Severity one is a total loss of a customer-facing capability. Severity two is a degradation with a
workaround. Severity three is contained and not customer-visible. Anyone may declare an incident; only
the incident commander may downgrade one.

During a severity one, the commander posts a status update every thirty minutes even when the update is
"no change". Silence reads as an abandoned incident.

Every incident above severity three gets a written postmortem within five working days, naming the
contributing causes and the specific change that prevents recurrence. Postmortems are blameless and
are not optional.
