# Testing Policy

**Agent-trigger:** What must be tested and to what standard before a change can merge: the test tiers, coverage expectations, and how flaky tests are handled. Load when adding tests, when a test fails intermittently, when changing coverage thresholds, and when deciding whether a change needs an integration test.

Unit tests cover behaviour, not implementation detail. An integration test is required whenever a change
crosses a process or network boundary. An end-to-end test is required for a user-visible flow that money
depends on.

Coverage is a floor that ratchets: it may rise and may not fall. A pull request that lowers coverage
states why in its description.

A flaky test is quarantined the day it is identified and either fixed or deleted within two weeks. A
permanently retried test asserts nothing and trains everyone to ignore red.
