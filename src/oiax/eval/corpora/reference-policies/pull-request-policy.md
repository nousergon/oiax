# Pull Request Policy

**Agent-trigger:** What a pull request must contain before it can be reviewed or merged: description, linked issue, test evidence, and the review approvals required. Load before opening a pull request, before merging one, and when deciding how many reviewers a change needs.

Every pull request states what changed and why, links the issue it closes, and shows evidence the test
suite passed locally before it was opened. A description that only restates the diff is not a description.

One approval merges a change confined to a single module. Two approvals, one from a code owner, are
required for a change touching authentication, billing, or a public interface.

A pull request left open more than ten days is closed. Reopen it when the work resumes; a stale branch
is a liability, not a bookmark.
