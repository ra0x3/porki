import json

import pytest

import main as orchestrator_main


def test_run_parser_accepts_direct_prompt():
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(["run", "--prompt", "summarize this repository"])

    assert args.prompt == "summarize this repository"
    assert args.instructions is None


def test_run_parser_accepts_run_id_and_checkpoint():
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(
        [
            "run",
            "--instructions",
            "INSTRUCTIONS.yaml",
            "--run-id",
            "run-9000",
            "--checkpoint",
            "./artifacts/run-9000.json",
        ]
    )

    assert args.run_id == "run-9000"
    assert str(args.checkpoint).endswith("run-9000.json")


def test_instructions_parser_subcommands_args():
    parser = orchestrator_main._build_parser()

    validate_args = parser.parse_args(["instructions", "validate", "--path", "./I.yaml"])
    compile_args = parser.parse_args(["instructions", "compile", "--path", "./I.yaml"])
    execute_args = parser.parse_args(
        [
            "instructions",
            "execute",
            "--path",
            "./I.yaml",
            "--run-id",
            "run-1",
            "--checkpoint",
            "./ckpt.json",
        ]
    )

    assert validate_args.instructions_command == "validate"
    assert compile_args.instructions_command == "compile"
    assert execute_args.instructions_command == "execute"


def test_instructions_create_writes_template(tmp_path):
    target_dir = tmp_path / "instructions"
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(
        [
            "instructions",
            "create",
            "--name",
            "QA automation",
            "--path",
            str(target_dir),
            "--log-level",
            "ERROR",
        ]
    )

    exit_code = orchestrator_main._handle_instructions_command(args, parser)

    assert exit_code == 0
    target = target_dir / "QA_AUTOMATION.yaml"
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert 'instruction_schema_version: "4"' in content
    assert "goal_typing:" in content
    assert "confidence:" in content


def test_instructions_validate_returns_structured_diagnostics(tmp_path, capsys):
    path = tmp_path / "INSTRUCTIONS.yaml"
    path.write_text(
        """instruction_schema_version: "4"
goal_typing:
  primary_goal_class: Transform
  secondary_goal_classes: [Retrieve/Research]
goal:
  statement: "Send outreach update"
  requested_effects:
    - primitive: net.send
      data_class: business
      recipient_class: trusted_partner
      consent_class: explicit
      legal_basis_class: unknown
      reversibility_class: irreversible
inputs: {}
policy:
  profile: guarded
success:
  rubric: "ack delivered"
assumptions:
  - id: a1
    statement: "consent still valid"
    status: confirmed
confidence:
  target_statement: "Plan success under assumptions"
  calibration_source: none
  interval:
    low: 0.3
    high: 0.6
  assumption_sensitivity:
    - assumption_id: a1
      rank: 1
      breaks_if_false: true
  evidence_gap: []
  last_updated_at_stage: capture
""",
        encoding="utf-8",
    )
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(
        ["instructions", "validate", "--path", str(path), "--log-level", "ERROR"]
    )
    exit_code = orchestrator_main._handle_instructions_command(args, parser)

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["valid"] is False
    assert payload["schema_version"] == "4"


def test_instructions_compile_emits_structured_result(monkeypatch, tmp_path, capsys):
    parser = orchestrator_main._build_parser()
    source = tmp_path / "INSTRUCTIONS.yaml"
    source.write_text("instruction_schema_version: '4'\n", encoding="utf-8")

    class FakeResult:
        valid = True

        @staticmethod
        def model_dump(mode: str = "json"):
            return {"valid": True, "diagnostics": [], "plan": {"plan_id": "p1"}}

    monkeypatch.setattr(orchestrator_main, "compile_source", lambda _: FakeResult())
    monkeypatch.setattr(orchestrator_main, "emit_compile_result", lambda result, path: path)

    args = parser.parse_args(
        ["instructions", "compile", "--path", str(source), "--out", str(tmp_path / "out.json")]
    )
    exit_code = orchestrator_main._handle_instructions_command(args, parser)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["valid"] is True


def test_instructions_execute_emits_runtime_result(monkeypatch, tmp_path, capsys):
    parser = orchestrator_main._build_parser()
    source = tmp_path / "INSTRUCTIONS.yaml"
    source.write_text("instruction_schema_version: '4'\n", encoding="utf-8")

    class FakeExecuteResult:
        @staticmethod
        def model_dump(mode: str = "json"):
            return {
                "run_id": "run-42",
                "plan_id": "plan.transform",
                "executed_tasks": ["a", "b"],
                "checkpoint_path": str(tmp_path / "run-42.json"),
            }

    monkeypatch.setattr(
        orchestrator_main,
        "execute_source",
        lambda path, run_id, checkpoint_path: FakeExecuteResult(),
    )

    args = parser.parse_args(
        [
            "instructions",
            "execute",
            "--path",
            str(source),
            "--run-id",
            "run-42",
            "--checkpoint",
            str(tmp_path / "run-42.json"),
        ]
    )
    exit_code = orchestrator_main._handle_instructions_command(args, parser)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["run_id"] == "run-42"


def test_run_executes_without_role(monkeypatch, tmp_path, capsys):
    parser = orchestrator_main._build_parser()
    source = tmp_path / "INSTRUCTIONS.yaml"
    source.write_text('instruction_schema_version: "4"\n', encoding="utf-8")

    class FakeExecuteResult:
        @staticmethod
        def model_dump(mode: str = "json"):
            return {
                "run_id": "run-test",
                "plan_id": "plan.transform",
                "executed_tasks": ["coding.plan", "coding.implement"],
                "checkpoint_path": str(tmp_path / "checkpoints" / "run-test.json"),
            }

    monkeypatch.setattr(
        orchestrator_main,
        "execute_source",
        lambda path, run_id, checkpoint_path: FakeExecuteResult(),
    )

    args = parser.parse_args(
        [
            "run",
            "--instructions",
            str(source),
            "--run-id",
            "run-test",
            "--checkpoint",
            str(tmp_path / "checkpoints" / "run-test.json"),
        ]
    )

    exit_code = orchestrator_main._handle_run_command(args, parser)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["run_id"] == "run-test"


def test_run_non_schema_instructions_error(tmp_path):
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(["run", "--instructions", str(tmp_path / "INSTRUCTIONS.md")])
    (tmp_path / "INSTRUCTIONS.md").write_text("agents: []\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        orchestrator_main._handle_run_command(args, parser)
