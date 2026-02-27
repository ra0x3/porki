"""Local-only integration tests for end-to-end schema execution."""

from __future__ import annotations

import json

import pytest

import main as orchestrator_main


@pytest.mark.local_integration
def test_local_integration_execute_creates_calculator_artifact(tmp_path):
    source = tmp_path / "INSTRUCTIONS.yaml"
    checkpoint = tmp_path / "artifacts" / "run-local.json"
    calculator = tmp_path / "artifacts" / "calculator.py"

    source.write_text(
        f"""instruction_schema_version: "4"
goal_typing:
  primary_goal_class: Transform
  secondary_goal_classes: [Decide]
goal:
  statement: "Create a python calculator program"
  requested_effects:
    - primitive: fs.write
      data_class: internal
      recipient_class: internal
      consent_class: na
      legal_basis_class: na
      reversibility_class: reversible
inputs:
  artifact_path: "{calculator}"
policy:
  profile: fast
success:
  rubric: "calculator artifact generated"
assumptions:
  - id: a1
    statement: "filesystem is writable"
    status: confirmed
confidence:
  target_statement: "Plan can produce calculator artifact"
  calibration_source: none
  interval:
    low: 0.4
    high: 0.9
  assumption_sensitivity:
    - assumption_id: a1
      rank: 1
      breaks_if_false: true
  evidence_gap: []
  last_updated_at_stage: capture
""",
        encoding="utf-8",
    )

    exit_code = orchestrator_main.run_cli(
        [
            "instructions",
            "execute",
            "--path",
            str(source),
            "--run-id",
            "run-local",
            "--checkpoint",
            str(checkpoint),
        ]
    )

    assert exit_code == 0
    assert checkpoint.exists()
    assert calculator.exists()

    checkpoint_payload = json.loads(checkpoint.read_text("utf-8"))
    assert checkpoint_payload["run_id"] == "run-local"
    assert checkpoint_payload["run_state"] == "succeeded"

    content = calculator.read_text("utf-8")
    assert "def add" in content
    assert "def div" in content
    assert "ZeroDivisionError" in content


@pytest.mark.local_integration
def test_local_integration_compile_emits_artifact_json(tmp_path):
    source = tmp_path / "INSTRUCTIONS.yaml"
    compile_out = tmp_path / "artifacts" / "compile.json"

    source.write_text(
        """instruction_schema_version: "4"
goal_typing:
  primary_goal_class: Transform
  secondary_goal_classes: [Decide]
goal:
  statement: "Create a python calculator program"
  requested_effects:
    - primitive: fs.write
      data_class: internal
      recipient_class: internal
      consent_class: na
      legal_basis_class: na
      reversibility_class: reversible
inputs: {}
policy:
  profile: fast
success:
  rubric: "artifact generated"
assumptions:
  - id: a1
    statement: "filesystem is writable"
    status: confirmed
confidence:
  target_statement: "Plan can produce artifact"
  calibration_source: none
  interval:
    low: 0.4
    high: 0.9
  assumption_sensitivity:
    - assumption_id: a1
      rank: 1
      breaks_if_false: true
  evidence_gap: []
  last_updated_at_stage: capture
""",
        encoding="utf-8",
    )

    exit_code = orchestrator_main.run_cli(
        [
            "instructions",
            "compile",
            "--path",
            str(source),
            "--out",
            str(compile_out),
        ]
    )

    assert exit_code == 0
    assert compile_out.exists()
    payload = json.loads(compile_out.read_text("utf-8"))
    assert payload["valid"] is True
    assert payload["plan"]["tasks"]
