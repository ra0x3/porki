"""Domain operator pack interfaces and reference implementations."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from .intent import EffectTuple, GoalClass
from .ir import FailureClass, RetryClass, RoleAssignmentMode, TaskIRNode


class PackContext(BaseModel):
    """Context provided to domain packs for task synthesis."""

    goal_statement: str
    lane: str
    max_tasks: int = Field(default=3, ge=1)


class DomainPack(Protocol):
    """Operator pack contract."""

    name: str

    def supports(self, goal_class: GoalClass) -> bool:
        """Return whether pack supports the requested goal class."""

    def build_tasks(self, context: PackContext) -> list[TaskIRNode]:
        """Build initial task nodes for compile/runtime pipeline."""


class CodingPack:
    """Reference coding domain pack."""

    name = "coding"

    def supports(self, goal_class: GoalClass) -> bool:
        return goal_class in {GoalClass.TRANSFORM, GoalClass.OPERATE}

    def build_tasks(self, context: PackContext) -> list[TaskIRNode]:
        tasks = [
            TaskIRNode(
                id="coding.plan",
                op="coding.plan",
                deps=[],
                role_assignment=RoleAssignmentMode.SOFT,
                required_capabilities=["coding"],
                effects=[],
                preconditions=["goal_parsed"],
                postconditions=["implementation_plan_ready"],
                evidence_contract={"requires": ["plan_doc"]},
                verifier_policy={"min_tier": 1},
                idempotency={"strategy": "task_id"},
                retry_class=RetryClass.NONE,
                failure_class=FailureClass.NON_RETRYABLE,
                cost_model={"tokens": 400},
            ),
            TaskIRNode(
                id="coding.implement",
                op="coding.implement",
                deps=["coding.plan"],
                role_assignment=RoleAssignmentMode.HARD,
                required_capabilities=["coding", "testing"],
                effects=[
                    EffectTuple(
                        primitive="fs.write",
                        data_class="internal",
                        recipient_class="internal",
                        consent_class="na",
                        legal_basis_class="na",
                        reversibility_class="reversible",
                    )
                ],
                preconditions=["implementation_plan_ready"],
                postconditions=["tests_pass"],
                evidence_contract={"requires": ["diff", "test_output"]},
                verifier_policy={"min_tier": 1},
                idempotency={"strategy": "payload_hash"},
                retry_class=RetryClass.TRANSIENT,
                failure_class=FailureClass.RETRYABLE,
                cost_model={"tokens": 1200},
            ),
        ]
        return tasks[: context.max_tasks]


class OutreachPack:
    """Reference outreach domain pack with guarded defaults."""

    name = "outreach"

    def supports(self, goal_class: GoalClass) -> bool:
        return goal_class == GoalClass.OUTREACH

    def build_tasks(self, context: PackContext) -> list[TaskIRNode]:
        tasks = [
            TaskIRNode(
                id="outreach.draft",
                op="outreach.draft",
                deps=[],
                role_assignment=RoleAssignmentMode.SOFT,
                required_capabilities=["copywriting"],
                effects=[],
                preconditions=["recipient_scope_defined"],
                postconditions=["draft_ready"],
                evidence_contract={"requires": ["draft_payload"]},
                verifier_policy={"min_tier": 2},
                idempotency={"strategy": "task_id"},
                retry_class=RetryClass.NONE,
                failure_class=FailureClass.NON_RETRYABLE,
                cost_model={"tokens": 300},
            ),
            TaskIRNode(
                id="outreach.send",
                op="outreach.send",
                deps=["outreach.draft"],
                role_assignment=RoleAssignmentMode.HARD,
                required_capabilities=["communications"],
                effects=[
                    EffectTuple(
                        primitive="net.send",
                        data_class="business",
                        recipient_class="trusted_partner",
                        consent_class="explicit",
                        legal_basis_class="contract",
                        reversibility_class="irreversible",
                    )
                ],
                preconditions=["approval_granted"],
                postconditions=["receipt_recorded"],
                evidence_contract={"requires": ["delivery_receipt"]},
                verifier_policy={"min_tier": 2, "require_human": True},
                idempotency={"strategy": "effect_tuple_hash"},
                retry_class=RetryClass.TRANSIENT,
                failure_class=FailureClass.POLICY_BLOCKED,
                cost_model={"tokens": 250},
            ),
        ]
        if context.lane == "fast":
            return tasks[:1]
        return tasks[: context.max_tasks]


class ResearchPack:
    """Reference research pack with anti-proxy verification requirements."""

    name = "research"

    def supports(self, goal_class: GoalClass) -> bool:
        return goal_class in {GoalClass.RETRIEVE_RESEARCH, GoalClass.DECIDE}

    def build_tasks(self, context: PackContext) -> list[TaskIRNode]:
        tasks = [
            TaskIRNode(
                id="research.retrieve",
                op="research.retrieve",
                deps=[],
                role_assignment=RoleAssignmentMode.SOFT,
                required_capabilities=["research"],
                effects=[],
                preconditions=["query_defined"],
                postconditions=["sources_collected"],
                evidence_contract={"requires": ["source_list", "provenance"]},
                verifier_policy={"min_tier": 1, "anti_proxy": True},
                idempotency={"strategy": "query_hash"},
                retry_class=RetryClass.TRANSIENT,
                failure_class=FailureClass.RETRYABLE,
                cost_model={"tokens": 500},
            ),
            TaskIRNode(
                id="research.summarize",
                op="research.summarize",
                deps=["research.retrieve"],
                role_assignment=RoleAssignmentMode.SOFT,
                required_capabilities=["analysis"],
                effects=[],
                preconditions=["sources_collected"],
                postconditions=["summary_ready"],
                evidence_contract={"requires": ["summary", "citations"]},
                verifier_policy={"min_tier": 2, "anti_proxy": True},
                idempotency={"strategy": "payload_hash"},
                retry_class=RetryClass.TRANSIENT,
                failure_class=FailureClass.RETRYABLE,
                cost_model={"tokens": 450},
            ),
        ]
        return tasks[: context.max_tasks]


def default_domain_packs() -> list[DomainPack]:
    """Return default pack set for runtime."""
    return [CodingPack(), OutreachPack(), ResearchPack()]


__all__ = [
    "PackContext",
    "DomainPack",
    "CodingPack",
    "OutreachPack",
    "ResearchPack",
    "default_domain_packs",
]
