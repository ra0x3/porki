"""Deterministic scheduler policy helpers for role assignment decisions."""

from __future__ import annotations

from .models import TaskNode


VALID_ROLE_ASSIGNMENTS = {"hard", "soft"}


def parse_bool_metadata(value: object) -> bool | None:
    """Parse common string/boolean metadata values into bools."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def role_assignment_mode(node: TaskNode) -> str:
    """Return strict role-assignment mode, rejecting implicit/fuzzy values."""
    mode = str(node.metadata.get("role_assignment", "")).strip().lower()
    if mode not in VALID_ROLE_ASSIGNMENTS:
        raise ValueError(
            f"Task {node.id} missing or invalid role_assignment (expected one of: hard, soft)"
        )
    return mode


def role_assignment_is_hard(node: TaskNode) -> bool:
    """Return whether role assignment should be treated as a hard constraint."""
    return role_assignment_mode(node) == "hard"


def role_category(role: str) -> str:
    """Classify a role into manager/reviewer/builder buckets."""
    normalized = role.strip().lower().replace("_", "-")
    if normalized in {"owner", "team-lead", "lead", "manager"}:
        return "manager"
    if any(token in normalized for token in ("qa", "test", "review")):
        return "reviewer"
    return "builder"


def candidate_fallback_roles(
    *,
    node: TaskNode,
    active_roles: set[str],
    current_runtime_role: str,
) -> list[str]:
    """Return deterministic fallback roles that may execute the task."""
    current_role = str(node.metadata.get("required_role", "")).strip().lower()
    if not current_role or role_assignment_is_hard(node):
        return []
    if role_category(current_role) != "builder":
        return []

    configured = str(node.metadata.get("fallback_roles", "")).strip()
    if configured:
        allowed = {
            candidate.strip().lower()
            for candidate in configured.split(",")
            if candidate.strip()
        }
        allowed = {role for role in allowed if role_category(role) == "builder"}
    else:
        allowed = {role for role in active_roles if role_category(role) == "builder"}

    allowed.discard(current_role)
    if current_runtime_role in allowed:
        ordered = [current_runtime_role]
        ordered.extend(sorted(role for role in allowed if role != current_runtime_role))
        return ordered
    return sorted(allowed)
