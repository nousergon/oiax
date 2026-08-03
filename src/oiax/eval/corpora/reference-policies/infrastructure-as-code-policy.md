# Infrastructure As Code Policy

**Agent-trigger:** Provisioning infrastructure through committed code rather than console clicks: which repository owns which surface, drift detection, and the prohibition on manual changes. Load before creating or modifying infrastructure, when deciding where provisioning code lives, and when console drift is discovered.

Every provisioned resource is described in code in exactly one repository. Two repositories describing
the same resource is a conflict waiting for the worst possible moment to surface.

Manual console changes are permitted only during an active incident and are reconciled back into code
within one working day. Anything else is drift.

Drift is detected on a schedule, not by accident. A drift report nobody reads is not detection.
