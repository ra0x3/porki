"""IR legalizer checks for policy and invariant admissibility."""

from __future__ import annotations

from collections import defaultdict, deque

from pydantic import BaseModel, Field

from .intent import CanonicalPolicy, EffectTuple, Severity, ValidationDiagnostic
from .ir import CompiledPlan, InvariantKind


class LegalizationReport(BaseModel):
    """Legalization result for compiled plans."""

    legal: bool
    diagnostics: list[ValidationDiagnostic] = Field(default_factory=list)


def _diag(
    code: str, path: str, message: str, severity: Severity = Severity.ERROR
) -> ValidationDiagnostic:
    """Build a typed legalizer diagnostic."""
    return ValidationDiagnostic(code=code, severity=severity, path=path, message=message)


def _is_acyclic(plan: CompiledPlan) -> bool:
    """Return whether plan graph is acyclic via Kahn's algorithm."""
    incoming: dict[str, int] = {task.id: 0 for task in plan.tasks}
    outgoing: dict[str, list[str]] = defaultdict(list)

    for task in plan.tasks:
        for dep in task.deps:
            incoming[task.id] += 1
            outgoing[dep].append(task.id)

    queue = deque(sorted([task_id for task_id, degree in incoming.items() if degree == 0]))
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for nxt in sorted(outgoing.get(node, [])):
            incoming[nxt] -= 1
            if incoming[nxt] == 0:
                queue.append(nxt)

    return visited == len(plan.tasks)


def _effect_matches(candidate: EffectTuple, rule: EffectTuple) -> bool:
    """Return whether a candidate effect matches a wildcard rule."""
    return all(
        candidate_value == rule_value or rule_value == "*"
        for candidate_value, rule_value in (
            (candidate.primitive, rule.primitive),
            (candidate.data_class, rule.data_class),
            (candidate.recipient_class, rule.recipient_class),
            (candidate.consent_class, rule.consent_class),
            (candidate.legal_basis_class, rule.legal_basis_class),
            (candidate.reversibility_class, rule.reversibility_class),
        )
    )


def _is_effect_denied(effect: EffectTuple, policy: CanonicalPolicy) -> bool:
    """Return whether effect is denied and not explicitly allowed."""
    is_allowed = any(_effect_matches(effect, rule) for rule in policy.allow_effects)
    is_denied = any(_effect_matches(effect, rule) for rule in policy.deny_effects)
    return is_denied and not is_allowed


def _validate_invariant_satisfiability(plan: CompiledPlan) -> list[ValidationDiagnostic]:
    """Validate invariant references against plan task IDs."""
    diagnostics: list[ValidationDiagnostic] = []
    task_ids = {task.id for task in plan.tasks}
    for invariant in plan.invariants:
        refs = invariant.metadata.get("requires_task_ids", [])
        if not isinstance(refs, list):
            diagnostics.append(
                _diag(
                    "invariant.metadata.invalid",
                    f"invariants.{invariant.id}",
                    "Invariant metadata requires_task_ids must be a list",
                )
            )
            continue
        for task_id in refs:
            if task_id not in task_ids:
                diagnostics.append(
                    _diag(
                        "invariant.unsatisfiable",
                        f"invariants.{invariant.id}",
                        f"Invariant references unknown task {task_id}",
                    )
                )

    hard_invariants = [item for item in plan.invariants if item.kind == InvariantKind.HARD]
    if not hard_invariants:
        diagnostics.append(
            _diag(
                "invariant.hard.missing",
                "invariants",
                "Plan should include at least one hard invariant",
                severity=Severity.WARNING,
            )
        )
    return diagnostics


def legalize_compiled_plan(plan: CompiledPlan, policy: CanonicalPolicy) -> LegalizationReport:
    """Run hard legalization checks over IR and policy."""
    diagnostics: list[ValidationDiagnostic] = []

    if not _is_acyclic(plan):
        diagnostics.append(
            _diag("graph.cycle.detected", "tasks", "Compiled plan graph must be acyclic")
        )

    for task in sorted(plan.tasks, key=lambda item: item.id):
        for index, effect in enumerate(task.effects):
            if _is_effect_denied(effect, policy):
                diagnostics.append(
                    _diag(
                        "policy.effect.denied",
                        f"tasks.{task.id}.effects[{index}]",
                        "Task effect is denied by canonical policy",
                    )
                )

    diagnostics.extend(_validate_invariant_satisfiability(plan))
    diagnostics = sorted(
        diagnostics,
        key=lambda item: (item.severity, item.code, item.path, item.message),
    )
    return LegalizationReport(
        legal=not any(item.severity == Severity.ERROR for item in diagnostics),
        diagnostics=diagnostics,
    )


__all__ = [
    "LegalizationReport",
    "legalize_compiled_plan",
]
