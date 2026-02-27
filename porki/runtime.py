"""Deterministic runtime skeleton for plan execution."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .intent import CanonicalPolicy
from .ir import CompiledPlan


class RunState(StrEnum):
    """Run lifecycle states."""

    CAPTURING = "capturing"
    CLARIFYING = "clarifying"
    COMPILING = "compiling"
    LEGALIZING = "legalizing"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    REPLANNING = "replanning"
    SUCCEEDED = "succeeded"
    FAILED_POLICY = "failed_policy"
    FAILED_VERIFICATION = "failed_verification"
    FAILED_COMPILATION = "failed_compilation"
    FAILED_INVARIANTS = "failed_invariants"
    ESCALATED_HUMAN = "escalated_human"
    ABORTED_USER = "aborted_user"


class TaskState(StrEnum):
    """Task lifecycle states for deterministic scheduler."""

    PENDING = "pending"
    ADMISSIBLE = "admissible"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELED = "canceled"


class RuntimeEvent(BaseModel):
    """Structured runtime trace event."""

    step: int
    event_type: str
    run_state: RunState
    task_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class DeterministicRuntimeSnapshot(BaseModel):
    """Serializable runtime checkpoint snapshot."""

    run_id: str
    plan_id: str
    run_state: RunState
    task_states: dict[str, TaskState]
    trace: list[RuntimeEvent]


class DeterministicRuntime:
    """Single-writer deterministic runtime skeleton."""

    def __init__(self, run_id: str, plan: CompiledPlan, policy: CanonicalPolicy):
        self.run_id = run_id
        self.plan = plan
        self.policy = policy
        self.run_state = RunState.EXECUTING
        self.task_states: dict[str, TaskState] = {
            task.id: TaskState.PENDING for task in sorted(plan.tasks, key=lambda item: item.id)
        }
        self.trace: list[RuntimeEvent] = []
        self._step = 0

    def _emit(
        self, event_type: str, *, task_id: str | None = None, payload: dict | None = None
    ) -> None:
        """Append deterministic runtime event."""
        self._step += 1
        self.trace.append(
            RuntimeEvent(
                step=self._step,
                event_type=event_type,
                run_state=self.run_state,
                task_id=task_id,
                payload=payload or {},
            )
        )

    def admissible_frontier(self) -> list[str]:
        """Return policy/invariant admissible task frontier."""
        completed = {
            task_id for task_id, state in self.task_states.items() if state == TaskState.COMPLETED
        }
        frontier: list[str] = []
        tasks_by_id = {task.id: task for task in self.plan.tasks}
        for task_id in sorted(tasks_by_id):
            state = self.task_states[task_id]
            if state != TaskState.PENDING:
                continue
            deps = tasks_by_id[task_id].deps
            if all(dep in completed for dep in deps):
                frontier.append(task_id)
        for task_id in frontier:
            self.task_states[task_id] = TaskState.ADMISSIBLE
        return frontier

    def execute_next(self, evidence: dict[str, Any] | None = None) -> str | None:
        """Execute one admissible task deterministically and commit state."""
        frontier = self.admissible_frontier()
        if not frontier:
            if all(state == TaskState.COMPLETED for state in self.task_states.values()):
                self.run_state = RunState.SUCCEEDED
                self._emit("run_succeeded")
            return None

        task_id = frontier[0]
        self.task_states[task_id] = TaskState.RUNNING
        self._emit("task_started", task_id=task_id)

        self.task_states[task_id] = TaskState.VERIFYING
        self.run_state = RunState.VERIFYING
        self._emit("task_verifying", task_id=task_id, payload={"evidence": evidence or {}})

        self.task_states[task_id] = TaskState.COMPLETED
        self.run_state = RunState.EXECUTING
        self._emit("task_completed", task_id=task_id)
        return task_id

    def checkpoint(self, path: Path) -> Path:
        """Write deterministic checkpoint snapshot."""
        snapshot = DeterministicRuntimeSnapshot(
            run_id=self.run_id,
            plan_id=self.plan.plan_id,
            run_state=self.run_state,
            task_states=self.task_states,
            trace=self.trace,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True), "utf-8"
        )
        return path


__all__ = [
    "RunState",
    "TaskState",
    "RuntimeEvent",
    "DeterministicRuntimeSnapshot",
    "DeterministicRuntime",
]
