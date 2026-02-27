"""Tests for deterministic runtime skeleton."""

from __future__ import annotations

import json

from porki.intent import PolicySource, normalize_policy
from porki.ir import (
    CompiledPlan,
    FailureClass,
    RetryClass,
    RoleAssignmentMode,
    TaskIRNode,
)
from porki.runtime import DeterministicRuntime, RunState, TaskState


def _task(task_id: str, deps: list[str] | None = None) -> TaskIRNode:
    return TaskIRNode(
        id=task_id,
        op="op",
        deps=deps or [],
        role_assignment=RoleAssignmentMode.HARD,
        required_capabilities=[],
        effects=[],
        preconditions=[],
        postconditions=[],
        evidence_contract={},
        verifier_policy={},
        idempotency={},
        retry_class=RetryClass.NONE,
        failure_class=FailureClass.NON_RETRYABLE,
        cost_model={},
    )


def test_runtime_executes_tasks_in_deterministic_order(tmp_path):
    plan = CompiledPlan(plan_id="p", tasks=[_task("b", ["a"]), _task("a")])
    policy, _ = normalize_policy(PolicySource(profile="fast"))
    assert policy is not None

    runtime = DeterministicRuntime("run-1", plan, policy)
    assert runtime.execute_next() == "a"
    assert runtime.execute_next() == "b"
    assert runtime.execute_next() is None
    assert runtime.run_state == RunState.SUCCEEDED

    checkpoint_path = runtime.checkpoint(tmp_path / "checkpoints" / "run-1.json")
    payload = json.loads(checkpoint_path.read_text("utf-8"))

    assert payload["run_state"] == "succeeded"
    assert payload["task_states"]["a"] == TaskState.COMPLETED
    assert payload["task_states"]["b"] == TaskState.COMPLETED


def test_runtime_frontier_marks_admissible_tasks():
    plan = CompiledPlan(plan_id="p", tasks=[_task("a"), _task("b", ["a"])])
    policy, _ = normalize_policy(PolicySource(profile="fast"))
    assert policy is not None

    runtime = DeterministicRuntime("run-2", plan, policy)
    frontier = runtime.admissible_frontier()

    assert frontier == ["a"]
    assert runtime.task_states["a"] == TaskState.ADMISSIBLE
