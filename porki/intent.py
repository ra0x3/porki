"""Strict intent source validation and policy normalization."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

INSTRUCTION_SCHEMA_VERSION = "4"


class GoalClass(StrEnum):
    """Goal classes supported by intent typing."""

    TRANSFORM = "Transform"
    DECIDE = "Decide"
    RETRIEVE_RESEARCH = "Retrieve/Research"
    OUTREACH = "Outreach"
    OPERATE = "Operate"
    MONITOR = "Monitor"


class AssumptionStatus(StrEnum):
    """Assumption resolution states."""

    CONFIRMED = "confirmed"
    BOUNDED = "bounded"
    UNRESOLVED = "unresolved"


class CalibrationSource(StrEnum):
    """Confidence calibration sources."""

    HISTORICAL_RUNS = "historical_runs"
    BENCHMARKS = "benchmarks"
    HELD_OUT_EVALS = "held_out_evals"
    NONE = "none"


class ConfidenceStage(StrEnum):
    """Runtime stage where confidence was updated."""

    CAPTURE = "capture"
    COMPILE = "compile"
    EXECUTE = "execute"
    REPLAN = "replan"
    VERIFY = "verify"


class Severity(StrEnum):
    """Diagnostic severity levels."""

    ERROR = "error"
    WARNING = "warning"


class EffectTuple(BaseModel):
    """Typed effect tuple used for validation and normalization."""

    model_config = ConfigDict(extra="forbid")

    primitive: str
    data_class: str
    recipient_class: str
    consent_class: str
    legal_basis_class: str
    reversibility_class: str

    def stable_key(self) -> str:
        """Return deterministic string key for sorting and dedupe."""
        return (
            f"{self.primitive}|{self.data_class}|{self.recipient_class}|"
            f"{self.consent_class}|{self.legal_basis_class}|{self.reversibility_class}"
        )


class GoalTyping(BaseModel):
    """Primary and secondary goal class typing."""

    model_config = ConfigDict(extra="forbid")

    primary_goal_class: GoalClass
    secondary_goal_classes: list[GoalClass] = Field(default_factory=list)


class AssumptionItem(BaseModel):
    """One assumption ledger record."""

    model_config = ConfigDict(extra="forbid")

    id: str
    statement: str
    status: AssumptionStatus


class ConfidenceInterval(BaseModel):
    """Confidence interval bounds."""

    model_config = ConfigDict(extra="forbid")

    low: float = Field(ge=0.0, le=1.0)
    high: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> ConfidenceInterval:
        """Ensure interval bounds are ordered."""
        if self.low > self.high:
            raise ValueError("confidence.interval.low must be <= confidence.interval.high")
        return self


class AssumptionSensitivityItem(BaseModel):
    """Sensitivity entry for one assumption."""

    model_config = ConfigDict(extra="forbid")

    assumption_id: str
    rank: int = Field(ge=1)
    breaks_if_false: bool


class ConfidenceObject(BaseModel):
    """Structured confidence metadata required by schema."""

    model_config = ConfigDict(extra="forbid")

    target_statement: str
    calibration_source: CalibrationSource
    interval: ConfidenceInterval
    assumption_sensitivity: list[AssumptionSensitivityItem] = Field(default_factory=list)
    evidence_gap: list[str] = Field(default_factory=list)
    last_updated_at_stage: ConfidenceStage


class PolicySource(BaseModel):
    """Source policy input for canonical normalization."""

    model_config = ConfigDict(extra="forbid")

    profile: str = "guarded"
    inherits: list[str] = Field(default_factory=list)
    macros: list[str] = Field(default_factory=list)
    allow_effects: list[EffectTuple] = Field(default_factory=list)
    deny_effects: list[EffectTuple] = Field(default_factory=list)
    approval_requirements: dict[str, str] = Field(default_factory=dict)
    data_handling_rules: dict[str, str] = Field(default_factory=dict)
    suppression_list: list[str] = Field(default_factory=list)
    version_pin: str | None = None
    primary_override_justification: str | None = None


class GoalSpec(BaseModel):
    """Goal intent payload."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    requested_effects: list[EffectTuple] = Field(default_factory=list)


class SourceV4(BaseModel):
    """Canonical strict source schema."""

    model_config = ConfigDict(extra="forbid")

    instruction_schema_version: str
    goal_typing: GoalTyping
    goal: GoalSpec
    inputs: dict[str, Any]
    policy: PolicySource
    success: dict[str, Any]
    assumptions: list[AssumptionItem]
    confidence: ConfidenceObject

    @model_validator(mode="after")
    def validate_schema_version(self) -> SourceV4:
        """Ensure explicit schema version."""
        if self.instruction_schema_version != INSTRUCTION_SCHEMA_VERSION:
            raise ValueError(f"instruction_schema_version must be {INSTRUCTION_SCHEMA_VERSION}")
        return self


class NormalizationExplanation(BaseModel):
    """One normalization decision explanation record."""

    source: str
    decision: str
    effect_key: str


class CanonicalPolicy(BaseModel):
    """Canonical policy generated by normalizer."""

    model_config = ConfigDict(extra="forbid")

    profile: str
    lane: str
    lane_constraints: dict[str, Any]
    allow_effects: list[EffectTuple]
    deny_effects: list[EffectTuple]
    approval_requirements: dict[str, str]
    data_handling_rules: dict[str, str]
    suppression_list: list[str]
    version_pin: str
    explanations: list[NormalizationExplanation]


class ValidationDiagnostic(BaseModel):
    """Machine-readable validation diagnostic."""

    code: str
    severity: Severity
    path: str
    message: str


class ValidationReport(BaseModel):
    """Validation outcome with deterministic diagnostics and policy output."""

    valid: bool
    schema_version: str | None
    diagnostics: list[ValidationDiagnostic]
    canonical_policy: CanonicalPolicy | None = None

    def as_json(self) -> str:
        """Serialize deterministic JSON for CLI output."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=2)


PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "lane": "fast",
        "lane_constraints": {
            "read_only": True,
            "require_multi_verifier": False,
            "require_approval_for_irreversible": True,
        },
        "approval_requirements": {
            "net.send": "human_required",
            "payment.charge": "human_required",
            "destructive": "human_required",
        },
        "data_handling_rules": {},
    },
    "guarded": {
        "lane": "guarded",
        "lane_constraints": {
            "read_only": False,
            "require_multi_verifier": True,
            "require_approval_for_irreversible": True,
        },
        "approval_requirements": {
            "net.send": "human_required",
            "payment.charge": "human_required",
            "destructive": "human_required",
            "regulated_data": "compliance_required",
        },
        "data_handling_rules": {
            "regulated": "encrypt_and_restrict",
        },
    },
    "regulatory": {
        "lane": "guarded",
        "lane_constraints": {
            "read_only": False,
            "require_multi_verifier": True,
            "require_approval_for_irreversible": True,
        },
        "approval_requirements": {
            "net.send": "human_required",
            "payment.charge": "human_required",
            "destructive": "human_required",
            "regulated_data": "compliance_required",
            "override": "compliance_required",
        },
        "data_handling_rules": {
            "regulated": "encrypt_and_restrict",
            "sensitive": "mask_and_log_access",
        },
    },
}


MACRO_PRESETS: dict[str, dict[str, Any]] = {
    "allow_outreach_to_trusted_partners_with_consent": {
        "allow_effects": [
            {
                "primitive": "net.send",
                "data_class": "business",
                "recipient_class": "trusted_partner",
                "consent_class": "explicit",
                "legal_basis_class": "contract",
                "reversibility_class": "irreversible",
            }
        ]
    }
}


SECONDARY_COMPATIBILITY: dict[GoalClass, set[GoalClass]] = {
    GoalClass.TRANSFORM: {GoalClass.DECIDE, GoalClass.RETRIEVE_RESEARCH},
    GoalClass.DECIDE: {GoalClass.RETRIEVE_RESEARCH, GoalClass.TRANSFORM},
    GoalClass.RETRIEVE_RESEARCH: {GoalClass.TRANSFORM, GoalClass.DECIDE},
    GoalClass.OUTREACH: {GoalClass.RETRIEVE_RESEARCH, GoalClass.TRANSFORM},
    GoalClass.OPERATE: {GoalClass.RETRIEVE_RESEARCH, GoalClass.TRANSFORM},
    GoalClass.MONITOR: {GoalClass.RETRIEVE_RESEARCH, GoalClass.TRANSFORM, GoalClass.DECIDE},
}

IRREVERSIBLE_PRIMITIVES = {"net.send", "payment.charge", "db.delete", "ticket.close"}
HIGH_RISK_PRIMITIVES = {"net.send", "payment.charge"}
UNGROUNDED_TOKENS = {"unknown", "tbd", "the usual", "usual", "same as before"}


def _diag(
    code: str, path: str, message: str, severity: Severity = Severity.ERROR
) -> ValidationDiagnostic:
    """Build a typed diagnostic record."""
    return ValidationDiagnostic(code=code, path=path, message=message, severity=severity)


def _sorted_unique_effects(effects: list[EffectTuple]) -> list[EffectTuple]:
    """Return stable sorted unique effect tuples."""
    by_key: dict[str, EffectTuple] = {effect.stable_key(): effect for effect in effects}
    return [by_key[key] for key in sorted(by_key)]


def _build_default_denies() -> list[EffectTuple]:
    """Return default deny tuples for protected effect classes."""
    rows = [
        ("net.send", "*", "*", "*", "*", "*"),
        ("payment.charge", "*", "*", "*", "*", "*"),
        ("db.delete", "*", "*", "*", "*", "*"),
        ("ticket.close", "*", "*", "*", "*", "*"),
    ]
    return [
        EffectTuple(
            primitive=row[0],
            data_class=row[1],
            recipient_class=row[2],
            consent_class=row[3],
            legal_basis_class=row[4],
            reversibility_class=row[5],
        )
        for row in rows
    ]


def normalize_policy(
    source: PolicySource,
) -> tuple[CanonicalPolicy | None, list[ValidationDiagnostic]]:
    """Normalize policy source into canonical deterministic policy."""
    diagnostics: list[ValidationDiagnostic] = []
    profile_key = source.profile.strip().lower()
    if profile_key not in PROFILE_PRESETS:
        diagnostics.append(
            _diag(
                "policy.profile.unknown",
                "policy.profile",
                f"Unknown policy profile: {source.profile}",
            )
        )
        return None, diagnostics

    merged_approval = dict(PROFILE_PRESETS[profile_key]["approval_requirements"])
    merged_data_rules = dict(PROFILE_PRESETS[profile_key]["data_handling_rules"])
    lane = PROFILE_PRESETS[profile_key]["lane"]
    lane_constraints = dict(PROFILE_PRESETS[profile_key]["lane_constraints"])

    allow_effects = list(source.allow_effects)
    deny_effects = _build_default_denies() + list(source.deny_effects)
    explanations: list[NormalizationExplanation] = []

    inherits = sorted({item.strip().lower() for item in source.inherits if item.strip()})
    for inherited in inherits:
        preset = PROFILE_PRESETS.get(inherited)
        if not preset:
            diagnostics.append(
                _diag(
                    "policy.inherits.unknown",
                    "policy.inherits",
                    f"Unknown inherited policy profile: {inherited}",
                )
            )
            continue
        merged_approval.update(preset["approval_requirements"])
        merged_data_rules.update(preset["data_handling_rules"])

    macros = sorted({item.strip() for item in source.macros if item.strip()})
    for macro in macros:
        preset = MACRO_PRESETS.get(macro)
        if not preset:
            diagnostics.append(
                _diag("policy.macro.unknown", "policy.macros", f"Unknown policy macro: {macro}")
            )
            continue
        macro_effects = [EffectTuple.model_validate(item) for item in preset["allow_effects"]]
        allow_effects.extend(macro_effects)
        for effect in macro_effects:
            explanations.append(
                NormalizationExplanation(
                    source=f"macro:{macro}",
                    decision="allow",
                    effect_key=effect.stable_key(),
                )
            )

    merged_approval.update(source.approval_requirements)
    merged_data_rules.update(source.data_handling_rules)

    final_allow = _sorted_unique_effects(allow_effects)
    final_deny = _sorted_unique_effects(deny_effects)

    allow_keys = {effect.stable_key() for effect in final_allow}
    deny_keys = {effect.stable_key() for effect in final_deny}
    overlaps = sorted(allow_keys.intersection(deny_keys))
    for overlap in overlaps:
        diagnostics.append(
            _diag(
                "policy.allow_deny.conflict",
                "policy.allow_effects",
                f"Effect is present in both allow and deny sets: {overlap}",
            )
        )

    for effect in final_allow:
        explanations.append(
            NormalizationExplanation(
                source="policy.allow_effects",
                decision="allow",
                effect_key=effect.stable_key(),
            )
        )
    for effect in final_deny:
        explanations.append(
            NormalizationExplanation(
                source="policy.deny_effects",
                decision="deny",
                effect_key=effect.stable_key(),
            )
        )

    canonical = CanonicalPolicy(
        profile=profile_key,
        lane=lane,
        lane_constraints=lane_constraints,
        allow_effects=final_allow,
        deny_effects=final_deny,
        approval_requirements=dict(sorted(merged_approval.items())),
        data_handling_rules=dict(sorted(merged_data_rules.items())),
        suppression_list=sorted({item.strip() for item in source.suppression_list if item.strip()}),
        version_pin=source.version_pin or "policy.latest",
        explanations=sorted(
            explanations,
            key=lambda item: (item.decision, item.effect_key, item.source),
        ),
    )
    return canonical, diagnostics


def _validate_goal_typing(source: SourceV4) -> list[ValidationDiagnostic]:
    """Validate primary-secondary goal typing composition and effect constraints."""
    diagnostics: list[ValidationDiagnostic] = []
    primary = source.goal_typing.primary_goal_class
    allowed_secondary = SECONDARY_COMPATIBILITY[primary]

    for secondary in source.goal_typing.secondary_goal_classes:
        if secondary not in allowed_secondary:
            diagnostics.append(
                _diag(
                    "goal_typing.secondary.incompatible",
                    "goal_typing.secondary_goal_classes",
                    f"Secondary class {secondary} is incompatible with primary {primary}",
                )
            )

    has_net_send = any(effect.primitive == "net.send" for effect in source.goal.requested_effects)
    has_destructive_or_payment = any(
        effect.primitive in {"payment.charge", "db.delete", "ticket.close"}
        for effect in source.goal.requested_effects
    )

    if (
        has_net_send
        and primary != GoalClass.OUTREACH
        and not source.policy.primary_override_justification
    ):
        diagnostics.append(
            _diag(
                "goal_typing.primary.required_outreach",
                "goal_typing.primary_goal_class",
                "Primary goal class must be Outreach when net.send is requested",
            )
        )

    if (
        has_destructive_or_payment
        and primary != GoalClass.OPERATE
        and not source.policy.primary_override_justification
    ):
        diagnostics.append(
            _diag(
                "goal_typing.primary.required_operate",
                "goal_typing.primary_goal_class",
                "Primary goal class must be Operate for destructive or payment effects",
            )
        )

    return diagnostics


def _validate_assumptions(source: SourceV4) -> list[ValidationDiagnostic]:
    """Validate assumption ledger and sensitivity links."""
    diagnostics: list[ValidationDiagnostic] = []
    if not source.assumptions:
        diagnostics.append(
            _diag("assumptions.missing", "assumptions", "Assumption ledger must not be empty")
        )
        return diagnostics

    assumption_ids = {item.id for item in source.assumptions}
    unresolved_ids = {
        item.id for item in source.assumptions if item.status == AssumptionStatus.UNRESOLVED
    }

    for item in source.confidence.assumption_sensitivity:
        if item.assumption_id not in assumption_ids:
            diagnostics.append(
                _diag(
                    "confidence.assumption_sensitivity.unknown_assumption",
                    "confidence.assumption_sensitivity",
                    f"Unknown assumption_id: {item.assumption_id}",
                )
            )

    has_high_risk_effect = any(
        effect.primitive in HIGH_RISK_PRIMITIVES for effect in source.goal.requested_effects
    )
    if has_high_risk_effect and unresolved_ids:
        diagnostics.append(
            _diag(
                "assumptions.unresolved.high_risk",
                "assumptions",
                "High-risk effects require all assumptions to be resolved or bounded",
            )
        )

    return diagnostics


def _contains_ungrounded(text: str) -> bool:
    """Return whether text appears ungrounded for policy-relevant fields."""
    lowered = text.strip().lower()
    return any(token in lowered for token in UNGROUNDED_TOKENS)


def _validate_effects(source: SourceV4) -> list[ValidationDiagnostic]:
    """Validate effect tuples and required legal-basis safety gates."""
    diagnostics: list[ValidationDiagnostic] = []

    for index, effect in enumerate(source.goal.requested_effects):
        prefix = f"goal.requested_effects[{index}]"
        if _contains_ungrounded(effect.recipient_class):
            diagnostics.append(
                _diag(
                    "effects.ungrounded.recipient",
                    f"{prefix}.recipient_class",
                    "Recipient class is ungrounded for policy-sensitive effect",
                )
            )
        if _contains_ungrounded(effect.data_class):
            diagnostics.append(
                _diag(
                    "effects.ungrounded.data",
                    f"{prefix}.data_class",
                    "Data class is ungrounded for policy-sensitive effect",
                )
            )

        requires_strict_fields = (
            effect.primitive in IRREVERSIBLE_PRIMITIVES
            or effect.reversibility_class.lower() == "irreversible"
        )
        if requires_strict_fields:
            missing_paths: list[str] = []
            if _contains_ungrounded(effect.recipient_class):
                missing_paths.append("recipient_class")
            if _contains_ungrounded(effect.consent_class):
                missing_paths.append("consent_class")
            if _contains_ungrounded(effect.legal_basis_class):
                missing_paths.append("legal_basis_class")
            if missing_paths:
                diagnostics.append(
                    _diag(
                        "effects.required_fields.unresolved",
                        prefix,
                        "Irreversible effects require grounded recipient/consent/legal basis fields",
                    )
                )

        if effect.primitive == "net.send" and effect.legal_basis_class.strip().lower() == "unknown":
            diagnostics.append(
                _diag(
                    "effects.legal_basis.unknown",
                    f"{prefix}.legal_basis_class",
                    "net.send is blocked when legal_basis_class is unknown",
                )
            )

    return diagnostics


def _validate_policy_compatibility(
    source: SourceV4,
    canonical_policy: CanonicalPolicy | None,
) -> list[ValidationDiagnostic]:
    """Validate requested effects against canonical policy allow/deny rules."""
    if canonical_policy is None:
        return []

    diagnostics: list[ValidationDiagnostic] = []
    allow_keys = {item.stable_key() for item in canonical_policy.allow_effects}
    deny_keys = {item.stable_key() for item in canonical_policy.deny_effects}

    for index, effect in enumerate(source.goal.requested_effects):
        key = effect.stable_key()
        if key in deny_keys and key not in allow_keys:
            diagnostics.append(
                _diag(
                    "effects.policy.denied",
                    f"goal.requested_effects[{index}]",
                    f"Requested effect is denied by normalized policy: {key}",
                )
            )

    if (
        source.goal_typing.primary_goal_class == GoalClass.OUTREACH
        and source.policy.suppression_list
        and any(effect.primitive == "net.send" for effect in source.goal.requested_effects)
    ):
        diagnostics.append(
            _diag(
                "outreach.suppression.review",
                "policy.suppression_list",
                "Suppression list present; runtime must resolve do-not-contact before net.send",
                severity=Severity.WARNING,
            )
        )

    return diagnostics


def _normalize_diagnostics(items: list[ValidationDiagnostic]) -> list[ValidationDiagnostic]:
    """Return diagnostics in deterministic order."""
    return sorted(items, key=lambda item: (item.severity, item.code, item.path, item.message))


def validate_payload(payload: dict[str, Any]) -> ValidationReport:
    """Validate raw payload and return typed report."""
    diagnostics: list[ValidationDiagnostic] = []
    canonical_policy: CanonicalPolicy | None = None

    try:
        source = SourceV4.model_validate(payload)
    except ValidationError as exc:
        for error in exc.errors():
            path = ".".join(str(piece) for piece in error["loc"])
            diagnostics.append(
                _diag(
                    "schema.validation_error",
                    path,
                    error["msg"],
                )
            )
        return ValidationReport(
            valid=False,
            schema_version=str(payload.get("instruction_schema_version") or ""),
            diagnostics=_normalize_diagnostics(diagnostics),
            canonical_policy=None,
        )

    canonical_policy, policy_diags = normalize_policy(source.policy)
    diagnostics.extend(policy_diags)
    diagnostics.extend(_validate_goal_typing(source))
    diagnostics.extend(_validate_assumptions(source))
    diagnostics.extend(_validate_effects(source))
    diagnostics.extend(_validate_policy_compatibility(source, canonical_policy))

    errors = [item for item in diagnostics if item.severity == Severity.ERROR]
    return ValidationReport(
        valid=not errors,
        schema_version=source.instruction_schema_version,
        diagnostics=_normalize_diagnostics(diagnostics),
        canonical_policy=canonical_policy,
    )


def validate_file(path: Path) -> ValidationReport:
    """Validate source file from disk."""
    raw_text = path.read_text(encoding="utf-8")
    if "```" in raw_text:
        return ValidationReport(
            valid=False,
            schema_version=None,
            diagnostics=[
                _diag(
                    "source.markdown.unsupported",
                    "$",
                    "Instruction files must be strict YAML documents",
                )
            ],
            canonical_policy=None,
        )

    try:
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        return ValidationReport(
            valid=False,
            schema_version=None,
            diagnostics=[
                _diag(
                    "source.yaml.invalid",
                    "$",
                    f"Invalid YAML instructions document: {exc}",
                )
            ],
            canonical_policy=None,
        )

    if not isinstance(payload, dict):
        return ValidationReport(
            valid=False,
            schema_version=None,
            diagnostics=[
                _diag(
                    "source.shape.invalid",
                    "$",
                    "Instructions must be a YAML mapping with top-level keys",
                )
            ],
            canonical_policy=None,
        )

    return validate_payload(payload)


__all__ = [
    "INSTRUCTION_SCHEMA_VERSION",
    "GoalClass",
    "AssumptionStatus",
    "CalibrationSource",
    "ConfidenceStage",
    "Severity",
    "EffectTuple",
    "GoalTyping",
    "AssumptionItem",
    "ConfidenceInterval",
    "AssumptionSensitivityItem",
    "ConfidenceObject",
    "PolicySource",
    "GoalSpec",
    "SourceV4",
    "NormalizationExplanation",
    "CanonicalPolicy",
    "ValidationDiagnostic",
    "ValidationReport",
    "PROFILE_PRESETS",
    "MACRO_PRESETS",
    "SECONDARY_COMPATIBILITY",
    "IRREVERSIBLE_PRIMITIVES",
    "HIGH_RISK_PRIMITIVES",
    "UNGROUNDED_TOKENS",
    "normalize_policy",
    "validate_payload",
    "validate_file",
]
