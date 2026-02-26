"""Tests for strict YAML instruction parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from porki.instructions import InstructionParser


@pytest.fixture
def strict_yaml_instructions(tmp_path: Path) -> Path:
    """Create a strict YAML instruction file."""
    instructions_path = tmp_path / "INSTRUCTIONS.yaml"
    instructions_path.write_text(
        """instruction_schema_version: "2"
agents:
  - name: test-agent-1
    role: ui-dev
    goal: test-goal
    heartbeat: instructions/heartbeat/TEST1.md
    instructions: instructions/TEST1.md
    log-level: DEBUG
    cadence: 10s

  - name: test-agent-2
    goal: test-goal-2
    heartbeat: instructions/heartbeat/TEST2.md
    instructions: instructions/TEST2.md
    log-level: INFO
    cadence: 5s
""",
        encoding="utf-8",
    )
    (tmp_path / "instructions" / "heartbeat").mkdir(parents=True)
    (tmp_path / "instructions").mkdir(exist_ok=True)
    (tmp_path / "instructions" / "heartbeat" / "TEST1.md").write_text("heartbeat 1")
    (tmp_path / "instructions" / "heartbeat" / "TEST2.md").write_text("heartbeat 2")
    (tmp_path / "instructions" / "TEST1.md").write_text("instructions 1")
    (tmp_path / "instructions" / "TEST2.md").write_text("instructions 2")
    return instructions_path


class TestInstructionParser:
    """Test the strict YAML-only InstructionParser."""

    def test_get_instructions_returns_full_yaml(self, strict_yaml_instructions: Path):
        parser = InstructionParser(strict_yaml_instructions)
        content = parser.get_instructions()

        assert "agents:" in content
        assert "test-agent-1" in content
        assert "heartbeat:" in content

    def test_parse_agents_from_strict_yaml(self, strict_yaml_instructions: Path):
        parser = InstructionParser(strict_yaml_instructions)
        agents = parser.parse_agents()

        assert len(agents) == 2
        agent1 = agents[0]
        assert agent1.name == "test-agent-1"
        assert agent1.effective_role == "ui-dev"
        assert agent1.goal_id == "test-goal"
        assert agent1.log_level == "DEBUG"
        assert agent1.cadence_seconds == 10
        assert agent1.heartbeat_path.name == "TEST1.md"
        assert agent1.instructions_path.name == "TEST1.md"

        agent2 = agents[1]
        assert agent2.name == "test-agent-2"
        assert agent2.effective_role == "test-agent-2"
        assert agent2.goal_id == "test-goal-2"
        assert agent2.cadence_seconds == 5

    def test_parse_agents_with_missing_file(self, tmp_path: Path):
        parser = InstructionParser(tmp_path / "nonexistent.yaml")
        assert parser.get_instructions() == ""
        assert parser.parse_agents() == []

    def test_rejects_markdown_codeblock_format(self, tmp_path: Path):
        instructions_path = tmp_path / "INSTRUCTIONS.md"
        instructions_path.write_text(
            """# Old format

```yaml
agents:
  - name: old-agent
    heartbeat: h.md
    instructions: i.md
```
""",
            encoding="utf-8",
        )
        parser = InstructionParser(instructions_path)
        with pytest.raises(ValueError, match="strict YAML documents"):
            parser.parse_agents()

    def test_parse_agents_with_invalid_yaml(self, tmp_path: Path):
        instructions_path = tmp_path / "INVALID.yaml"
        instructions_path.write_text(
            'instruction_schema_version: "2"\nagents:\n  - name: bad\n    : nope\n',
            encoding="utf-8",
        )
        parser = InstructionParser(instructions_path)
        with pytest.raises(ValueError, match="Invalid YAML instructions document"):
            parser.parse_agents()

    def test_requires_schema_version(self, tmp_path: Path):
        instructions_path = tmp_path / "NO_SCHEMA.yaml"
        instructions_path.write_text("agents: []\n", encoding="utf-8")
        parser = InstructionParser(instructions_path)
        with pytest.raises(ValueError, match="instruction_schema_version"):
            parser.parse_agents()

    def test_parse_agents_with_missing_required_fields(self, tmp_path: Path):
        instructions_path = tmp_path / "INCOMPLETE.yaml"
        instructions_path.write_text(
            """instruction_schema_version: "2"
agents:
  - name: incomplete-agent
""",
            encoding="utf-8",
        )
        parser = InstructionParser(instructions_path)
        with pytest.raises(ValueError, match="Missing required agent attribute"):
            parser.parse_agents()

    def test_rejects_non_list_agents(self, tmp_path: Path):
        instructions_path = tmp_path / "BAD_AGENTS.yaml"
        instructions_path.write_text(
            'instruction_schema_version: "2"\nagents: {}\n', encoding="utf-8"
        )
        parser = InstructionParser(instructions_path)
        with pytest.raises(ValueError, match="'agents' list"):
            parser.parse_agents()

    def test_cadence_parsing_variations(self, tmp_path: Path):
        instructions_path = tmp_path / "CADENCE.yaml"
        instructions_path.write_text(
            """instruction_schema_version: "2"
agents:
  - name: agent-1
    heartbeat: h1.md
    instructions: i1.md
    cadence: 30s
  - name: agent-2
    heartbeat: h2.md
    instructions: i2.md
    cadence: 15
  - name: agent-3
    heartbeat: h3.md
    instructions: i3.md
""",
            encoding="utf-8",
        )
        for f in ["h1.md", "h2.md", "h3.md", "i1.md", "i2.md", "i3.md"]:
            (tmp_path / f).write_text("content", encoding="utf-8")

        parser = InstructionParser(instructions_path)
        agents = parser.parse_agents()

        assert agents[0].cadence_seconds == 30
        assert agents[1].cadence_seconds == 15
        assert agents[2].cadence_seconds == 5
