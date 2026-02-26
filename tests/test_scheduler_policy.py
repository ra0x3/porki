from porki.models import TaskNode
from porki.scheduler_policy import candidate_fallback_roles, parse_bool_metadata, role_assignment_is_hard


def test_parse_bool_metadata_accepts_common_values():
    assert parse_bool_metadata(True) is True
    assert parse_bool_metadata("yes") is True
    assert parse_bool_metadata("0") is False
    assert parse_bool_metadata("off") is False
    assert parse_bool_metadata("maybe") is None


def test_role_assignment_is_hard_defaults_by_phase():
    dev = TaskNode(id="t-dev", title="Dev", priority=1, metadata={"role_assignment": "soft"})
    qa = TaskNode(id="t-qa", title="QA", priority=1, metadata={"role_assignment": "hard"})

    assert role_assignment_is_hard(dev) is False
    assert role_assignment_is_hard(qa) is True


def test_role_assignment_is_hard_obeys_explicit_metadata():
    soft = TaskNode(
        id="t-soft",
        title="Soft",
        priority=1,
        metadata={"role_assignment": "soft"},
    )
    hard = TaskNode(
        id="t-hard",
        title="Hard",
        priority=1,
        metadata={"role_assignment": "hard"},
    )

    assert role_assignment_is_hard(soft) is False
    assert role_assignment_is_hard(hard) is True


def test_role_assignment_is_hard_rejects_missing_or_invalid_mode():
    missing = TaskNode(id="t-miss", title="Missing", priority=1)
    invalid = TaskNode(id="t-bad", title="Invalid", priority=1, metadata={"role_assignment": "preferred"})

    import pytest

    with pytest.raises(ValueError, match="role_assignment"):
        role_assignment_is_hard(missing)
    with pytest.raises(ValueError, match="role_assignment"):
        role_assignment_is_hard(invalid)


def test_candidate_fallback_roles_prefers_current_runtime_when_compatible():
    node = TaskNode(
        id="task-1",
        title="Feature work",
        priority=1,
        metadata={"phase": "development", "required_role": "features-dev", "role_assignment": "soft"},
    )
    roles = candidate_fallback_roles(
        node=node,
        active_roles={"ui-dev", "core-infra-dev", "qa-dev"},
        current_runtime_role="ui-dev",
    )
    assert roles == ["ui-dev", "core-infra-dev"]


def test_candidate_fallback_roles_rejects_hard_and_non_builder_tasks():
    qa_node = TaskNode(
        id="qa",
        title="QA",
        priority=1,
        metadata={"phase": "qa", "required_role": "qa-dev", "role_assignment": "hard"},
    )
    owner_node = TaskNode(
        id="int",
        title="Integrate",
        priority=1,
        metadata={"phase": "integration", "required_role": "owner", "role_assignment": "soft"},
    )

    assert candidate_fallback_roles(
        node=qa_node,
        active_roles={"ui-dev", "features-dev"},
        current_runtime_role="ui-dev",
    ) == []
    assert candidate_fallback_roles(
        node=owner_node,
        active_roles={"ui-dev", "features-dev"},
        current_runtime_role="ui-dev",
    ) == []


def test_candidate_fallback_roles_obeys_configured_fallback_list():
    node = TaskNode(
        id="task-2",
        title="Feature work",
        priority=1,
        metadata={
            "phase": "development",
            "required_role": "features-dev",
            "role_assignment": "soft",
            "fallback_roles": "ui-dev, team-lead, core-infra-dev",
        },
    )
    roles = candidate_fallback_roles(
        node=node,
        active_roles={"ui-dev", "core-infra-dev", "team-lead"},
        current_runtime_role="owner",
    )
    # Team lead is excluded because fallback policy only permits builder categories.
    assert roles == ["core-infra-dev", "ui-dev"]
