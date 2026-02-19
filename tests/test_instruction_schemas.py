"""Tests for instruction schema references used in template generation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from porki.instruction_schemas import (
    INSTRUCTION_SCHEMA_VERSION,
    TaskSelectionSchema,
    render_instruction_schema_reference,
)


def test_render_instruction_schema_reference_contains_core_sections():
    """Rendered schema reference should include goal/task/finished-task blocks."""
    rendered = render_instruction_schema_reference()

    assert "## Schema Reference" in rendered
    assert "### Goal Schema" in rendered
    assert "### DAG Task Schema" in rendered
    assert "### Task State Schema" in rendered
    assert "### Finished Task Schema" in rendered
    assert "### LLM Task Selection Response" in rendered
    assert "### LLM Task Summary Response" in rendered
    assert INSTRUCTION_SCHEMA_VERSION in rendered


def test_task_selection_schema_confidence_bounds():
    """Task selection confidence should be constrained to [0, 1]."""
    with pytest.raises(ValidationError):
        TaskSelectionSchema(
            selected_task_id="task-001",
            justification="invalid confidence",
            confidence=1.2,
        )
