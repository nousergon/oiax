# Data Retention Policy

**Agent-trigger:** How long each class of data is kept and when it must be deleted: retention windows, customer deletion requests, backups, and what may be retained after an account closes. Load before storing a new class of data, before writing a deletion routine, and when a customer asks for their data to be removed.

Operational logs are retained ninety days. Application telemetry without personal data is retained two
years. Anything containing personal data carries the shortest window that still serves its stated purpose,
and the purpose is written down before the data is collected.

A customer deletion request is honoured within thirty days across primary storage, replicas, and backups.
A backup that cannot be selectively purged must expire on a schedule short enough to satisfy the same
window.

Deleting data you were required to keep is as serious as keeping data you were required to delete.
