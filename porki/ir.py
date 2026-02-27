"""Typed IR models and deterministic diff utilities."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .intent import EffectTuple


class RoleAssignmentMode(StrEnum):
    """Task role assignment policy."""

    HARD = "hard"
    SOFT = "soft"


class RetryClass(StrEnum):
    """Task retry policy classes."""

    NONE = "none"
    TRANSIENT = "transient"
    EXPONENTIAL = "exponential"


class FailureClass(StrEnum):
    """Task failure classification."""

    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    POLICY_BLOCKED = "policy_blocked"


class InvariantKind(StrEnum):
    """Invariant hardness type."""

    HARD = "hard"
    SOFT = "soft"


class InvariantSpec(BaseModel):
    """Invariant record for execution and replanning controls."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: InvariantKind
    expression: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskIRNode(BaseModel):
    """Typed task node contract for IR v0."""

    model_config = ConfigDict(extra="forbid")

    id: str
    op: str
    deps: list[str] = Field(default_factory=list)
    role_assignment: RoleAssignmentMode
    required_capabilities: list[str] = Field(default_factory=list)
    effects: list[EffectTuple] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    evidence_contract: dict[str, Any] = Field(default_factory=dict)
    verifier_policy: dict[str, Any] = Field(default_factory=dict)
    idempotency: dict[str, Any] = Field(default_factory=dict)
    retry_class: RetryClass
    failure_class: FailureClass
    cost_model: dict[str, Any] = Field(default_factory=dict)


class VerifierPlan(BaseModel):
    """Verifier plan contract keyed by task ID."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    allowed_tiers: list[int] = Field(default_factory=list)
    escalation_rule: str | None = None


class CompiledPlan(BaseModel):
    """Compiled IR plan bundle."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    source_schema_version: str = "4"
    tasks: list[TaskIRNode]
    invariants: list[InvariantSpec] = Field(default_factory=list)
    verifier_plan: list[VerifierPlan] = Field(default_factory=list)
    failure_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> CompiledPlan:
        """Ensure IDs are unique and dependencies refer to existing tasks."""
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Task IDs must be unique")
        id_set = set(ids)
        for task in self.tasks:
            for dep in task.deps:
                if dep not in id_set:
                    raise ValueError(f"Unknown dependency {dep} for task {task.id}")
        return self


class IRDeltaOperation(BaseModel):
    """One structured IR diff operation."""

    model_config = ConfigDict(extra="forbid")

    op: str
    task_id: str
    justification: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


class IRDelta(BaseModel):
    """Structured IR delta with proof linkage."""

    model_config = ConfigDict(extra="forbid")

    from_plan_id: str
    to_plan_id: str
    preserved_id_map: dict[str, str]
    operations: list[IRDeltaOperation]
    hard_invariant_preserved: bool
    approval_reference: str | None = None


def diff_compiled_plans(
    before: CompiledPlan,
    after: CompiledPlan,
    *,
    justification: str,
    hard_invariant_preserved: bool,
    approval_reference: str | None = None,
) -> IRDelta:
    """Compute deterministic structured diff between two plans."""
    before_map = {task.id: task for task in before.tasks}
    after_map = {task.id: task for task in after.tasks}

    preserved_id_map = {
        task_id: task_id
        for task_id in sorted(set(before_map).intersection(after_map))
        if before_map[task_id].model_dump(mode="json") == after_map[task_id].model_dump(mode="json")
    }

    operations: list[IRDeltaOperation] = []

    for task_id in sorted(set(before_map) - set(after_map)):
        operations.append(
            IRDeltaOperation(
                op="remove",
                task_id=task_id,
                justification=justification,
                before=before_map[task_id].model_dump(mode="json"),
                after=None,
            )
        )

    for task_id in sorted(set(after_map) - set(before_map)):
        operations.append(
            IRDeltaOperation(
                op="add",
                task_id=task_id,
                justification=justification,
                before=None,
                after=after_map[task_id].model_dump(mode="json"),
            )
        )

    for task_id in sorted(set(before_map).intersection(after_map)):
        old_node = before_map[task_id].model_dump(mode="json")
        new_node = after_map[task_id].model_dump(mode="json")
        if old_node != new_node:
            operations.append(
                IRDeltaOperation(
                    op="update",
                    task_id=task_id,
                    justification=justification,
                    before=old_node,
                    after=new_node,
                )
            )

    return IRDelta(
        from_plan_id=before.plan_id,
        to_plan_id=after.plan_id,
        preserved_id_map=preserved_id_map,
        operations=operations,
        hard_invariant_preserved=hard_invariant_preserved,
        approval_reference=approval_reference,
    )


__all__ = [
    "RoleAssignmentMode",
    "RetryClass",
    "FailureClass",
    "InvariantKind",
    "InvariantSpec",
    "TaskIRNode",
    "VerifierPlan",
    "CompiledPlan",
    "IRDeltaOperation",
    "IRDelta",
    "diff_compiled_plans",
]
