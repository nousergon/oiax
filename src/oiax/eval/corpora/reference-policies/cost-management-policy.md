# Cost Management Policy

**Agent-trigger:** Controlling cloud spend: right-sizing, interruptible compute, committed-use discounts, always-on services, and responding to a spend anomaly. Load before provisioning compute, choosing spot versus on-demand, adding a scheduled or always-on job, buying a committed term, and when a bill comes in unexpectedly high.

Batch and retryable work runs on interruptible compute by default. Choosing on-demand for such work
requires a written reason.

Instances are sized from measured utilisation, not from the shape of the largest expected load. A
resource idle more than eighty percent of the week is a candidate for removal, not for a smaller size.

Committed-use discounts are bought against a demonstrated twelve-month floor of usage, never against a
forecast. A spend anomaly is investigated within one business day; unexplained growth is treated as a
defect.
