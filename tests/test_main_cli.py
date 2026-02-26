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
    assert args.log_style == "concise"


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
            "--log-style",
            "event",
        ]
    )
    assert args.heartbeat_interval == 45.0
    assert args.instruction_interval == 90.0
    assert args.idle_log_interval == 15.0
    assert args.log_style == "event"


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


def test_instructions_parser_validate_args():
    """Instructions validate subcommand should parse required path."""
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(
        [
            "instructions",
            "validate",
            "--path",
            "./INSTRUCTIONS.yaml",
        ]
    )
    assert args.instructions_command == "validate"
    assert str(args.path).endswith("INSTRUCTIONS.yaml")


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


def test_instructions_validate_accepts_strict_yaml(tmp_path, capsys):
    """`porki instructions validate` should accept strict schema-v2 YAML."""
    path = tmp_path / "INSTRUCTIONS.yaml"
    (tmp_path / "instructions" / "heartbeat").mkdir(parents=True)
    (tmp_path / "instructions").mkdir(exist_ok=True)
    (tmp_path / "instructions" / "heartbeat" / "QA.md").write_text("RESUME\n", encoding="utf-8")
    (tmp_path / "instructions" / "QA.md").write_text("qa", encoding="utf-8")
    path.write_text(
        """instruction_schema_version: "2"
agents:
  - name: qa-dev
    goal: goal-demo
    heartbeat: instructions/heartbeat/QA.md
    instructions: instructions/QA.md
""",
        encoding="utf-8",
    )
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(
        ["instructions", "validate", "--path", str(path), "--log-level", "ERROR"]
    )
    exit_code = orchestrator_main._handle_instructions_command(args, parser)
    assert exit_code == 0
    assert "valid:" in capsys.readouterr().out


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


def test_handle_run_command_gracefully_stops_on_ctrl_c_agent(monkeypatch, tmp_path):
    """Agent mode should stop cleanly and return success on Ctrl+C."""
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(
        [
            "run",
            "--role",
            "agent",
            "--agent-name",
            "qa-dev",
            "--goal-id",
            "goal-demo",
            "--heartbeat",
            str(tmp_path / "heartbeat.md"),
            "--instructions",
            str(tmp_path / "instructions.md"),
            "--redis-url",
            "fakeredis://",
        ]
    )
    (tmp_path / "heartbeat.md").write_text("RESUME\n", encoding="utf-8")
    (tmp_path / "instructions.md").write_text("test", encoding="utf-8")

    monkeypatch.setattr(orchestrator_main, "_redis_client_from_url", lambda _: object())
    monkeypatch.setattr(orchestrator_main, "RedisStore", lambda client: object())
    monkeypatch.setattr(orchestrator_main, "create_llm_client", lambda *a, **k: object())

    class FakeRuntime:
        last = None

        def __init__(self, *args, **kwargs):
            self.stopped = False
            FakeRuntime.last = self

        def run(self):
            raise KeyboardInterrupt

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(orchestrator_main, "AgentRuntime", FakeRuntime)

    exit_code = orchestrator_main._handle_run_command(args, parser)
    assert exit_code == 0
    assert FakeRuntime.last is not None
    assert FakeRuntime.last.stopped is True


def test_handle_run_command_gracefully_stops_on_ctrl_c_orchestrator(monkeypatch, tmp_path):
    """Orchestrator mode should stop cleanly and return success on Ctrl+C."""
    parser = orchestrator_main._build_parser()
    args = parser.parse_args(
        [
            "run",
            "--role",
            "orchestrator",
            "--instructions",
            str(tmp_path / "INSTRUCTIONS.md"),
            "--redis-url",
            "fakeredis://",
        ]
    )
    (tmp_path / "INSTRUCTIONS.md").write_text("agents: []\n", encoding="utf-8")

    monkeypatch.setattr(orchestrator_main, "_redis_client_from_url", lambda _: object())
    monkeypatch.setattr(orchestrator_main, "RedisStore", lambda client: object())
    monkeypatch.setattr(orchestrator_main, "create_llm_client", lambda *a, **k: object())
    monkeypatch.setattr(orchestrator_main, "RealSpawnAdapter", lambda: object())

    class FakeOrchestrator:
        last = None

        def __init__(self, *args, **kwargs):
            self.stopped = False
            FakeOrchestrator.last = self

        def run(self):
            raise KeyboardInterrupt

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(orchestrator_main, "Orchestrator", FakeOrchestrator)

    exit_code = orchestrator_main._handle_run_command(args, parser)
    assert exit_code == 0
    assert FakeOrchestrator.last is not None
    assert FakeOrchestrator.last.stopped is True
