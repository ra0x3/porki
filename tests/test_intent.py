"""Tests for intent validation and policy normalization."""

from __future__ import annotations

from porki.intent import PolicySource, normalize_policy, validate_payload


def _base_payload() -> dict:
    return {
        "instruction_schema_version": "4",
        "goal_typing": {
            "primary_goal_class": "Outreach",
            "secondary_goal_classes": ["Retrieve/Research"],
        },
        "goal": {
            "statement": "Send approved partner outreach update",
            "requested_effects": [
                {
                    "primitive": "net.send",
                    "data_class": "business",
                    "recipient_class": "trusted_partner",
                    "consent_class": "explicit",
                    "legal_basis_class": "contract",
                    "reversibility_class": "irreversible",
                }
            ],
        },
        "inputs": {"campaign_id": "c-42"},
        "policy": {
            "profile": "guarded",
            "macros": ["allow_outreach_to_trusted_partners_with_consent"],
            "allow_effects": [
                {
                    "primitive": "net.send",
                    "data_class": "business",
                    "recipient_class": "trusted_partner",
                    "consent_class": "explicit",
                    "legal_basis_class": "contract",
                    "reversibility_class": "irreversible",
                }
            ],
        },
        "success": {"rubric": "delivery and acknowledgement"},
        "assumptions": [
            {
                "id": "a-consent",
                "statement": "Contact consent remains valid for this campaign",
                "status": "confirmed",
            }
        ],
        "confidence": {
            "target_statement": "Plan will satisfy outreach success rubric",
            "calibration_source": "historical_runs",
            "interval": {"low": 0.71, "high": 0.89},
            "assumption_sensitivity": [
                {"assumption_id": "a-consent", "rank": 1, "breaks_if_false": True}
            ],
            "evidence_gap": ["Recipient suppression reconciliation report"],
            "last_updated_at_stage": "capture",
        },
    }


def test_validate_payload_accepts_valid_outreach_payload():
    payload = _base_payload()

    report = validate_payload(payload)

    assert report.valid is True
    assert report.schema_version == "4"
    assert report.canonical_policy is not None
    assert report.canonical_policy.profile == "guarded"
    assert report.canonical_policy.lane == "guarded"


def test_validate_payload_rejects_incompatible_secondary_goal_class():
    payload = _base_payload()
    payload["goal_typing"]["secondary_goal_classes"] = ["Operate"]

    report = validate_payload(payload)

    assert report.valid is False
    assert any(diag.code == "goal_typing.secondary.incompatible" for diag in report.diagnostics)


def test_validate_payload_blocks_unknown_legal_basis_for_net_send():
    payload = _base_payload()
    payload["goal"]["requested_effects"][0]["legal_basis_class"] = "unknown"

    report = validate_payload(payload)

    assert report.valid is False
    assert any(diag.code == "effects.legal_basis.unknown" for diag in report.diagnostics)


def test_validate_payload_blocks_unresolved_assumptions_for_high_risk_effects():
    payload = _base_payload()
    payload["assumptions"][0]["status"] = "unresolved"

    report = validate_payload(payload)

    assert report.valid is False
    assert any(diag.code == "assumptions.unresolved.high_risk" for diag in report.diagnostics)


def test_policy_normalization_is_deterministic_for_same_input():
    policy = PolicySource.model_validate(_base_payload()["policy"])
    canonical_a, diags_a = normalize_policy(source=policy)
    canonical_b, diags_b = normalize_policy(source=policy)

    assert diags_a == diags_b
    assert canonical_a is not None
    assert canonical_b is not None
    assert canonical_a.model_dump(mode="json") == canonical_b.model_dump(mode="json")
