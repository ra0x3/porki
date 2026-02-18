# Logging Contract

This project treats logs as an operator-facing API.

## Record Schema

All log lines emitted through the CLI runtime include a structured suffix:

- `evt`: stable event code (defaults to `GEN` when unspecified)
- `goal`: goal identifier or `-`
- `role`: agent role or `-`
- `task`: task identifier or `-`
- `state`: lifecycle state (`active`, `running`, `blocked`, `backoff`, `complete`, `error`, `idle`, `stopped`) or `-`
- `next_retry`: ISO timestamp for retries/backoff or `-`

## Event Stability

- Event codes are designed for machine parsing and dashboards.
- Message text may evolve; event codes should remain stable.

## Deduplication

- Consecutive identical low-severity messages are compacted (`+N ...`).
- Structured fields are preserved on compacted summaries.

## Critical Runtime Events

- `GOAL_COMPLETE`
- `GOAL_BLOCKED_INACTIVE_ROLE`
- `BACKOFF_ACTIVE`
- `BACKOFF_CLEARED`
- `TASK_SELECTED`
- `TASK_EXECUTE_START`
- `TASK_EXECUTE_RESULT`
- `TASK_FAILED`
