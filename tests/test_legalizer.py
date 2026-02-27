"""Tests for IR legalizer policy and invariant checks."""

from __future__ import annotations

from porki.intent import EffectTuple, PolicySource, normalize_policy
from porki.ir import (
    CompiledPlan,
    FailureClass,
    InvariantKind,
    InvariantSpec,
    RetryClass,
    RoleAssignmentMode,
    TaskIRNode,
)
from porki.legalizer import legalize_compiled_plan


def _task(task_id: str, deps: list[str] | None = None, primitive: str = "tool.read") -> TaskIRNode:
    return TaskIRNode(
        id=task_id,
        op="op",
        deps=deps or [],
        role_assignment=RoleAssignmentMode.HARD,
        required_capabilities=[],
        effects=[
            EffectTuple(
                primitive=primitive,
                data_class="public",
                recipient_class="internal",
                consent_class="na",
                legal_basis_class="na",
                reversibility_class="reversible",
            )
        ],
        preconditions=[],
        postconditions=[],
        evidence_contract={},
        verifier_policy={},
        idempotency={},
        retry_class=RetryClass.NONE,
        failure_class=FailureClass.NON_RETRYABLE,
        cost_model={},
    )


def test_legalizer_rejects_cycle_and_denied_effect():
    policy, diagnostics = normalize_policy(PolicySource(profile="guarded"))
    assert policy is not None
    assert not diagnostics

    plan = CompiledPlan(
        plan_id="p1",
        tasks=[
            _task("a", ["b"], primitive="net.send"),
            _task("b", ["a"]),
        ],
        invariants=[
            InvariantSpec(
                id="i1",
                kind=InvariantKind.HARD,
                expression="must_include_a",
                metadata={"requires_task_ids": ["a"]},
            )
        ],
    )

    report = legalize_compiled_plan(plan, policy)

    assert report.legal is False
    codes = {item.code for item in report.diagnostics}
    assert "graph.cycle.detected" in codes
    assert "policy.effect.denied" in codes


def test_legalizer_accepts_read_only_plan():
    policy, diagnostics = normalize_policy(PolicySource(profile="fast"))
    assert policy is not None
    assert not diagnostics

    plan = CompiledPlan(
        plan_id="p2",
        tasks=[_task("a")],
        invariants=[
            InvariantSpec(
                id="i2",
                kind=InvariantKind.HARD,
                expression="must_include_a",
                metadata={"requires_task_ids": ["a"]},
            )
        ],
    )

    report = legalize_compiled_plan(plan, policy)

    assert report.legal is True
