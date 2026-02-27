"""Source-to-IR compiler and execution pipeline for the refactored runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from .domain_packs import PackContext, default_domain_packs
from .intent import (
    CanonicalPolicy,
    GoalClass,
    SourceV4,
    normalize_policy,
    validate_file,
)
from .ir import CompiledPlan, InvariantKind, InvariantSpec, VerifierPlan
from .legalizer import LegalizationReport, legalize_compiled_plan
from .runtime import DeterministicRuntime


class CompileResult(BaseModel):
    """Compile + legalize bundle."""

    valid: bool
    plan: CompiledPlan | None = None
    policy: CanonicalPolicy | None = None
    legalization: LegalizationReport | None = None
    diagnostics: list[dict[str, Any]]


class ExecuteResult(BaseModel):
    """Execution bundle with emitted checkpoint."""

    run_id: str
    plan_id: str
    executed_tasks: list[str]
    checkpoint_path: str


def load_source(path: Path) -> SourceV4:
    """Load and validate typed source document from YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Source must be a YAML mapping")
    return SourceV4.model_validate(payload)


def _resolve_pack(goal_class: GoalClass):
    packs = default_domain_packs()
    for pack in packs:
        if pack.supports(goal_class):
            return pack
    return None


def compile_source(path: Path) -> CompileResult:
    """Compile source file into typed plan and run legalizer checks."""
    report = validate_file(path)
    if not report.valid or report.canonical_policy is None:
        return CompileResult(
            valid=False,
            diagnostics=[item.model_dump(mode="json") for item in report.diagnostics],
        )

    source = load_source(path)
    policy, policy_diagnostics = normalize_policy(source.policy)
    if policy is None:
        return CompileResult(
            valid=False,
            diagnostics=[item.model_dump(mode="json") for item in policy_diagnostics],
        )

    pack = _resolve_pack(source.goal_typing.primary_goal_class)
    if pack is None:
        return CompileResult(
            valid=False,
            diagnostics=[
                {
                    "code": "compile.pack.unsupported",
                    "path": "goal_typing.primary_goal_class",
                    "severity": "error",
                    "message": (
                        f"No domain pack supports goal class {source.goal_typing.primary_goal_class}"
                    ),
                }
            ],
        )

    tasks = pack.build_tasks(
        context=PackContext(
            goal_statement=source.goal.statement,
            lane=policy.lane,
            max_tasks=4,
        )
    )

    verifier_plan = [
        VerifierPlan(
            task_id=task.id,
            allowed_tiers=[1, 2] if task.verifier_policy.get("min_tier", 1) <= 2 else [3],
            escalation_rule="on_fail",
        )
        for task in tasks
    ]

    invariants = [
        InvariantSpec(
            id="inv-hard-policy",
            kind=InvariantKind.HARD,
            expression="no_denied_effects",
            metadata={"requires_task_ids": [task.id for task in tasks]},
        ),
        InvariantSpec(
            id="inv-soft-latency",
            kind=InvariantKind.SOFT,
            expression="prefer_lane_budget",
            metadata={"lane": policy.lane},
        ),
    ]

    plan = CompiledPlan(
        plan_id=f"plan.{source.goal_typing.primary_goal_class.value.lower().replace('/', '-')}",
        tasks=tasks,
        invariants=invariants,
        verifier_plan=verifier_plan,
        failure_criteria=["failed_policy", "failed_verification", "failed_invariants"],
    )

    legalization = legalize_compiled_plan(plan, policy)
    diagnostics = [item.model_dump(mode="json") for item in legalization.diagnostics]
    return CompileResult(
        valid=legalization.legal,
        plan=plan,
        policy=policy,
        legalization=legalization,
        diagnostics=diagnostics,
    )


def execute_source(path: Path, *, run_id: str, checkpoint_path: Path) -> ExecuteResult:
    """Compile, legalize, execute deterministic runtime, and checkpoint."""
    compiled = compile_source(path)
    if not compiled.valid or compiled.plan is None or compiled.policy is None:
        raise ValueError("Source did not compile/legalize successfully")
    source = load_source(path)

    runtime = DeterministicRuntime(run_id=run_id, plan=compiled.plan, policy=compiled.policy)
    executed: list[str] = []
    while True:
        task_id = runtime.execute_next(evidence={"source": "deterministic"})
        if task_id is None:
            break
        executed.append(task_id)
        _materialize_artifacts(source, task_id, checkpoint_path.parent)

    written = runtime.checkpoint(checkpoint_path)
    return ExecuteResult(
        run_id=run_id,
        plan_id=compiled.plan.plan_id,
        executed_tasks=executed,
        checkpoint_path=str(written),
    )


def emit_compile_result(result: CompileResult, *, path: Path) -> Path:
    """Persist compile result as deterministic JSON."""
    payload: dict[str, Any] = result.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _materialize_artifacts(source: SourceV4, task_id: str, base_dir: Path) -> None:
    """Materialize deterministic demo artifacts for transformed coding goals."""
    if task_id != "coding.implement":
        return
    target = str(source.inputs.get("artifact_path", "")).strip()
    if not target:
        return
    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = (base_dir / target_path).resolve()
    statement = source.goal.statement.lower()
    if "calculator" in statement:
        content = (
            "def add(a: float, b: float) -> float:\n"
            "    return a + b\n\n"
            "def sub(a: float, b: float) -> float:\n"
            "    return a - b\n\n"
            "def mul(a: float, b: float) -> float:\n"
            "    return a * b\n\n"
            "def div(a: float, b: float) -> float:\n"
            "    if b == 0:\n"
            '        raise ZeroDivisionError("division by zero")\n'
            "    return a / b\n"
        )
    else:
        content = 'def run() -> str:\n    return "artifact generated"\n'
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")


__all__ = [
    "CompileResult",
    "ExecuteResult",
    "load_source",
    "compile_source",
    "execute_source",
    "emit_compile_result",
]
