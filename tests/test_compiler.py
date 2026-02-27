"""Tests for source compiler and deterministic execute pipeline."""

from __future__ import annotations

import json

from porki.compiler import compile_source, execute_source


def _source_yaml() -> str:
    return """instruction_schema_version: "4"
goal_typing:
  primary_goal_class: Transform
  secondary_goal_classes: [Decide]
goal:
  statement: "Refactor scheduler policy"
  requested_effects:
    - primitive: fs.write
      data_class: internal
      recipient_class: internal
      consent_class: na
      legal_basis_class: na
      reversibility_class: reversible
inputs:
  repo: local
policy:
  profile: fast
success:
  rubric: "tests green"
assumptions:
  - id: a1
    statement: "workspace is writable"
    status: confirmed
confidence:
  target_statement: "Plan can satisfy rubric"
  calibration_source: none
  interval:
    low: 0.5
    high: 0.8
  assumption_sensitivity:
    - assumption_id: a1
      rank: 1
      breaks_if_false: true
  evidence_gap: []
  last_updated_at_stage: capture
"""


def test_compile_source_returns_plan_and_policy(tmp_path):
    path = tmp_path / "INSTRUCTIONS.yaml"
    path.write_text(_source_yaml(), encoding="utf-8")

    result = compile_source(path)

    assert result.valid is True
    assert result.plan is not None
    assert result.policy is not None
    assert result.legalization is not None
    assert result.plan.tasks


def test_execute_source_writes_checkpoint(tmp_path):
    source_path = tmp_path / "INSTRUCTIONS.yaml"
    source_path.write_text(_source_yaml(), encoding="utf-8")
    checkpoint = tmp_path / "checkpoints" / "run-1.json"

    result = execute_source(source_path, run_id="run-1", checkpoint_path=checkpoint)

    assert result.run_id == "run-1"
    assert result.executed_tasks
    payload = json.loads(checkpoint.read_text("utf-8"))
    assert payload["run_id"] == "run-1"
