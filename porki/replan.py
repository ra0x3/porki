"""Bounded replanning with invariant exception workflow."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .ir import CompiledPlan, InvariantKind, IRDelta, TaskIRNode, diff_compiled_plans


class ExceptionRequestNode(BaseModel):
    """Explicit invariant exception request."""

    model_config = ConfigDict(extra="forbid")

    invariant_id: str
    justification: str
    impact_analysis: str
    approved: bool = False
    approval_reference: str | None = None


class ReplanResult(BaseModel):
    """Replanning result including new plan and diff proof."""

    new_plan: CompiledPlan
    delta: IRDelta


class ReplanController:
    """Replanning controller bounded by hard/soft invariants."""

    def replan_with_updates(
        self,
        current: CompiledPlan,
        *,
        updated_tasks: list[TaskIRNode],
        justification: str,
        exceptions: list[ExceptionRequestNode] | None = None,
    ) -> ReplanResult:
        """Apply task updates and enforce hard invariant relaxation rules."""
        exception_map = {item.invariant_id: item for item in (exceptions or [])}
        hard_invariants = [item for item in current.invariants if item.kind == InvariantKind.HARD]

        for invariant in hard_invariants:
            requested = exception_map.get(invariant.id)
            if requested is not None and not requested.approved:
                raise ValueError(
                    f"Hard invariant {invariant.id} relaxation requires approved exception"
                )

        updated_by_id = {task.id: task for task in updated_tasks}
        merged_tasks: list[TaskIRNode] = []
        for task in current.tasks:
            merged_tasks.append(updated_by_id.get(task.id, task))

        new_plan = CompiledPlan(
            plan_id=f"{current.plan_id}.r1",
            source_schema_version=current.source_schema_version,
            tasks=merged_tasks,
            invariants=current.invariants,
            verifier_plan=current.verifier_plan,
            failure_criteria=current.failure_criteria,
        )

        approval_ref = None
        if exceptions:
            refs = [item.approval_reference for item in exceptions if item.approval_reference]
            approval_ref = refs[0] if refs else None

        delta = diff_compiled_plans(
            current,
            new_plan,
            justification=justification,
            hard_invariant_preserved=True,
            approval_reference=approval_ref,
        )
        return ReplanResult(new_plan=new_plan, delta=delta)


__all__ = [
    "ExceptionRequestNode",
    "ReplanResult",
    "ReplanController",
]
