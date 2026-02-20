import pytest

import main as orchestrator_main
from porki.instruction_schemas import INSTRUCTION_SCHEMA_VERSION, INSTRUCTION_TEMPLATE_VERSION


def test_run_parser_defaults_for_refresh_intervals():
    """CLI should default both refresh intervals to five minutes."""
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(
        [
            "run",
            "--role",
            "orchestrator",
            "--instructions",
            "INSTRUCTIONS.md",
        ]
    )
    assert args.heartbeat_interval == 300.0
    assert args.instruction_interval == 300.0
    assert args.idle_log_interval == 60.0


def test_run_parser_accepts_refresh_interval_overrides():
    """CLI should parse user-provided refresh intervals."""
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(
        [
            "run",
            "--role",
            "orchestrator",
            "--instructions",
            "INSTRUCTIONS.md",
            "--heartbeat-interval",
            "45",
            "--instruction-interval",
            "90",
            "--idle-log-interval",
            "15",
        ]
    )
    assert args.heartbeat_interval == 45.0
    assert args.instruction_interval == 90.0
    assert args.idle_log_interval == 15.0


def test_run_parser_accepts_direct_prompt():
    """CLI should accept direct one-shot prompt mode."""
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(["run", "--prompt", "summarize this repository"])

    assert args.prompt == "summarize this repository"
    assert args.role is None
    assert args.instructions is None


def test_instructions_parser_create_template_args():
    """Instructions subcommand should parse create template arguments."""
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(
        [
            "instructions",
            "create",
            "--name",
            "create backend role instructions",
            "--path",
            "./instructions",
        ]
    )

    assert args.instructions_command == "create"
    assert str(args.path).endswith("instructions")
    assert args.name == "create backend role instructions"


def test_instructions_create_writes_default_template(tmp_path):
    """`porki instructions create` should create starter markdown."""
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
    target = target_dir / "QA_AUTOMATION.md"
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "porki_instruction_template_version" in content
    assert INSTRUCTION_TEMPLATE_VERSION in content
    assert "porki_schema_version" in content
    assert INSTRUCTION_SCHEMA_VERSION in content
    assert "generated_at_utc" in content
    assert "## Role" in content
    assert "## Responsibilities" in content
    assert "## Schema Reference" in content
    assert '"goal_id"' in content
    assert '"task_id"' in content


def test_run_requires_role_without_prompt():
    """`porki run` should reject runtime mode without role."""
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(["run", "--instructions", "INSTRUCTIONS.md"])

    with pytest.raises(SystemExit):
        orchestrator_main._handle_run_command(args, parser)


def test_instructions_create_name_to_upper_snake_filename(tmp_path):
    """Instruction filename should be uppercase snake case from --name."""
    target_dir = tmp_path / "instructions"
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(
        [
            "instructions",
            "create",
            "--name",
            "Accomplish task",
            "--path",
            str(target_dir),
            "--log-level",
            "ERROR",
        ]
    )

    exit_code = orchestrator_main._handle_instructions_command(args, parser)
    assert exit_code == 0

    target = target_dir / "ACCOMPLISH_TASK.md"
    assert target.exists()
