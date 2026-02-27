"""Tests for typed IR v4 models and diff semantics."""

from __future__ import annotations

from porki.intent import EffectTuple
from porki.ir import (
    CompiledPlan,
    FailureClass,
    InvariantKind,
    InvariantSpec,
    RetryClass,
    RoleAssignmentMode,
    TaskIRNode,
    VerifierPlan,
    diff_compiled_plans,
)


def _task(task_id: str, deps: list[str] | None = None) -> TaskIRNode:
    return TaskIRNode(
        id=task_id,
        op="research.query",
        deps=deps or [],
        role_assignment=RoleAssignmentMode.HARD,
        required_capabilities=["research"],
        effects=[
            EffectTuple(
                primitive="tool.read",
                data_class="public",
                recipient_class="internal",
                consent_class="na",
                legal_basis_class="na",
                reversibility_class="reversible",
            )
        ],
        preconditions=["input_available"],
        postconditions=["result_stored"],
        evidence_contract={"requires": ["source_url"]},
        verifier_policy={"min_tier": 1},
        idempotency={"strategy": "payload_hash"},
        retry_class=RetryClass.TRANSIENT,
        failure_class=FailureClass.RETRYABLE,
        cost_model={"tokens": 100},
    )


def _plan(plan_id: str) -> CompiledPlan:
    return CompiledPlan(
        plan_id=plan_id,
        tasks=[_task("t1"), _task("t2", ["t1"])],
        invariants=[
            InvariantSpec(
                id="inv-hard-1",
                kind=InvariantKind.HARD,
                expression="no_net_send_without_approval",
                metadata={"requires_task_ids": ["t1"]},
            )
        ],
        verifier_plan=[VerifierPlan(task_id="t1", allowed_tiers=[1, 2], escalation_rule="on_fail")],
        failure_criteria=["verification_failed"],
    )


def test_compiled_plan_validates_dependency_references():
    plan = _plan("p1")

    assert plan.plan_id == "p1"
    assert len(plan.tasks) == 2


def test_diff_compiled_plans_emits_deterministic_update_operation():
    before = _plan("p1")
    updated = _task("t2", ["t1"])
    updated.postconditions = ["result_stored", "review_complete"]
    after = CompiledPlan(
        plan_id="p2",
        tasks=[_task("t1"), updated],
        invariants=before.invariants,
        verifier_plan=before.verifier_plan,
        failure_criteria=before.failure_criteria,
    )

    delta = diff_compiled_plans(
        before,
        after,
        justification="new review obligation",
        hard_invariant_preserved=True,
    )

    assert delta.from_plan_id == "p1"
    assert delta.to_plan_id == "p2"
    assert len(delta.operations) == 1
    assert delta.operations[0].op == "update"
    assert delta.operations[0].task_id == "t2"
