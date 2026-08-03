# Deployment Policy

**Agent-trigger:** Shipping code to production: release windows, canary rollouts, rollback criteria, and who may push a deploy. Load before deploying, before changing a deploy pipeline, when a release needs to go out off-schedule, and when deciding whether to roll back or roll forward.

Deploys go out behind a canary that serves 5% of traffic for at least fifteen minutes. A canary that
raises the error rate above its baseline is rolled back automatically; no human confirmation is required
to roll back, and one is always required to roll forward past a failed canary.

Release windows are Monday through Thursday, 09:00-16:00 local. Deploying outside the window needs a
named approver and a written reason recorded on the change.

Rollback is the default response to a production regression. Fixing forward is permitted only when the
rollback itself would lose data.
