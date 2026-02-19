"""Typed schema references for generated instruction templates."""

from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, Field

INSTRUCTION_SCHEMA_VERSION = "orchestrator.v1"
INSTRUCTION_TEMPLATE_VERSION = "instructions.v1"


class GoalStatus(str, Enum):
    """Enumerates goal lifecycle states."""

    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"


class GoalSchema(BaseModel):
    """Canonical goal tracker schema."""

    goal_id: str
    title: str
    status: GoalStatus
    completion_timestamp: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class DagTaskSchema(BaseModel):
    """Canonical DAG node schema."""

    id: str
    title: str
    priority: int = Field(ge=0)
    expected_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class TaskStateSchema(BaseModel):
    """Canonical mutable task-state schema."""

    status: str
    owner: str | None = None
    lease_expires: str | None = None
    progress: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    last_error: str | None = None


class FinishedTaskSchema(BaseModel):
    """Canonical finished-task summary schema."""

    task_id: str
    status: str = "done"
    outputs: list[str] = Field(default_factory=list)
    notes: str
    follow_ups: list[str] = Field(default_factory=list)
    finished_at: str


class TaskSelectionSchema(BaseModel):
    """Canonical LLM task-selection response schema."""

    selected_task_id: str | None
    justification: str
    confidence: float = Field(ge=0.0, le=1.0)


class TaskSummarySchema(BaseModel):
    """Canonical LLM task-summary response schema."""

    status: str
    outputs: list[str] = Field(default_factory=list)
    notes: str
    follow_ups: list[str] = Field(default_factory=list)


def render_instruction_schema_reference() -> str:
    """Render markdown schema section with validated example payloads."""
    goal_example = GoalSchema(
        goal_id="goal-orchestrator-ui",
        title="Build systemg orchestration dashboard",
        status=GoalStatus.ACTIVE,
        completion_timestamp=None,
        metadata={"owner": "team-lead", "priority": "high"},
    )
    dag_task_example = DagTaskSchema(
        id="task-001",
        title="Initialize project skeleton",
        priority=10,
        expected_artifacts=["orchestrator-ui/package.json", "orchestrator-ui/src/main.tsx"],
        metadata={"owner_hint": "team-lead"},
    )
    task_state_example = TaskStateSchema(
        status="running",
        owner="agent-team-lead",
        lease_expires="2026-02-19T18:12:00Z",
        progress="Scaffolded Vite + React + TypeScript project",
        artifacts=["orchestrator-ui/package.json"],
        last_error=None,
    )
    finished_task_example = FinishedTaskSchema(
        task_id="task-001",
        status="done",
        outputs=["orchestrator-ui/package.json", "orchestrator-ui/tsconfig.json"],
        notes="Project skeleton is ready for feature and infra workstreams.",
        follow_ups=["task-002", "task-003"],
        finished_at="2026-02-19T18:30:00Z",
    )
    selection_example = TaskSelectionSchema(
        selected_task_id="task-002",
        justification="task-001 is complete and task-002 is highest ready priority.",
        confidence=0.86,
    )
    summary_example = TaskSummarySchema(
        status="done",
        outputs=["reports/integration-checklist.md"],
        notes="Validated acceptance criteria and documented outcomes.",
        follow_ups=["task-010__qa"],
    )

    return (
        "## Schema Reference\n"
        f"Schema version: `{INSTRUCTION_SCHEMA_VERSION}`.\n"
        "Use these canonical JSON schemas in prompts and outputs. Keep field names stable.\n\n"
        "### Goal Schema (`goal:<goal_id>`)\n"
        "```json\n"
        f"{json.dumps(goal_example.model_dump(mode='json'), indent=2)}\n"
        "```\n\n"
        "### DAG Task Schema (`dag:<goal_id>:nodes`)\n"
        "```json\n"
        f"{json.dumps(dag_task_example.model_dump(mode='json'), indent=2)}\n"
        "```\n\n"
        "### Task State Schema (`task:<task_id>`)\n"
        "```json\n"
        f"{json.dumps(task_state_example.model_dump(mode='json'), indent=2)}\n"
        "```\n\n"
        "### Finished Task Schema\n"
        "```json\n"
        f"{json.dumps(finished_task_example.model_dump(mode='json'), indent=2)}\n"
        "```\n\n"
        "### LLM Task Selection Response\n"
        "```json\n"
        f"{json.dumps(selection_example.model_dump(mode='json'), indent=2)}\n"
        "```\n\n"
        "### LLM Task Summary Response\n"
        "```json\n"
        f"{json.dumps(summary_example.model_dump(mode='json'), indent=2)}\n"
        "```\n"
    )
