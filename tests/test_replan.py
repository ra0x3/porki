"""Tests for bounded replanning and invariant exceptions."""

from __future__ import annotations

import pytest

from porki.ir import (
    CompiledPlan,
    FailureClass,
    InvariantKind,
    InvariantSpec,
    RetryClass,
    RoleAssignmentMode,
    TaskIRNode,
)
from porki.replan import ExceptionRequestNode, ReplanController


def _task(task_id: str, op: str = "op") -> TaskIRNode:
    return TaskIRNode(
        id=task_id,
        op=op,
        deps=[],
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


def _plan() -> CompiledPlan:
    return CompiledPlan(
        plan_id="plan-1",
        tasks=[_task("t1"), _task("t2")],
        invariants=[
            InvariantSpec(
                id="inv-hard",
                kind=InvariantKind.HARD,
                expression="must_not_relax",
                metadata={},
            ),
            InvariantSpec(
                id="inv-soft",
                kind=InvariantKind.SOFT,
                expression="nice_to_have",
                metadata={},
            ),
        ],
    )


def test_replan_updates_task_and_emits_delta():
    controller = ReplanController()

    result = controller.replan_with_updates(
        _plan(),
        updated_tasks=[_task("t2", op="op.updated")],
        justification="recover from verifier disagreement",
    )

    assert result.new_plan.plan_id == "plan-1.r1"
    assert len(result.delta.operations) == 1
    assert result.delta.operations[0].task_id == "t2"
    assert result.delta.operations[0].op == "update"


def test_replan_requires_approved_exception_for_hard_invariant_relaxation():
    controller = ReplanController()

    with pytest.raises(ValueError, match="requires approved exception"):
        controller.replan_with_updates(
            _plan(),
            updated_tasks=[_task("t1", op="op.updated")],
            justification="relax hard invariant",
            exceptions=[
                ExceptionRequestNode(
                    invariant_id="inv-hard",
                    justification="urgent",
                    impact_analysis="bounded",
                    approved=False,
                )
            ],
        )


def test_replan_accepts_approved_exception_reference():
    controller = ReplanController()
    result = controller.replan_with_updates(
        _plan(),
        updated_tasks=[_task("t1", op="op.updated")],
        justification="approved relaxation",
        exceptions=[
            ExceptionRequestNode(
                invariant_id="inv-hard",
                justification="urgent",
                impact_analysis="bounded",
                approved=True,
                approval_reference="apr-001",
            )
        ],
    )

    assert result.delta.approval_reference == "apr-001"
