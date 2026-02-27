"""Verifier tiers, no-self-grading, and proxy-risk controls."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class VerifierTier(StrEnum):
    """Supported verifier tiers."""

    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"


class VerifierSignal(BaseModel):
    """One verifier signal record."""

    tier: VerifierTier
    actor_id: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool


class VerificationOutcome(BaseModel):
    """Aggregated verification result with proxy risk."""

    passed: bool
    proxy_risk: float = Field(ge=0.0, le=1.0)
    escalation_required: bool
    reason: str


def compute_proxy_risk(signals: list[VerifierSignal]) -> float:
    """Compute proxy-risk from diversity and agreement profile."""
    if not signals:
        return 1.0
    tiers = {signal.tier for signal in signals}
    actor_ids = {signal.actor_id for signal in signals}
    mean_score = sum(signal.score for signal in signals) / len(signals)
    diversity_penalty = 0.0 if len(tiers) >= 2 else 0.25
    actor_penalty = 0.0 if len(actor_ids) >= 2 else 0.25
    calibration_penalty = 0.2 if mean_score > 0.95 else 0.0
    risk = min(1.0, diversity_penalty + actor_penalty + calibration_penalty)
    return round(risk, 4)


def enforce_no_self_grading(
    task_actor_id: str,
    signals: list[VerifierSignal],
    *,
    high_risk: bool,
    proxy_risk_threshold: float = 0.5,
) -> VerificationOutcome:
    """Enforce tier policy and no-self-grading constraints."""
    if not signals:
        return VerificationOutcome(
            passed=False,
            proxy_risk=1.0,
            escalation_required=True,
            reason="No verifier signals available",
        )

    if any(signal.actor_id == task_actor_id for signal in signals):
        return VerificationOutcome(
            passed=False,
            proxy_risk=1.0,
            escalation_required=True,
            reason="No-self-grading violation",
        )

    proxy_risk = compute_proxy_risk(signals)
    tier_set = {signal.tier for signal in signals}
    passed = all(signal.passed for signal in signals)

    if high_risk and VerifierTier.TIER1 not in tier_set and VerifierTier.TIER2 not in tier_set:
        return VerificationOutcome(
            passed=False,
            proxy_risk=proxy_risk,
            escalation_required=True,
            reason="High-risk flows require Tier1 or Tier2 verification",
        )

    if proxy_risk >= proxy_risk_threshold:
        return VerificationOutcome(
            passed=False,
            proxy_risk=proxy_risk,
            escalation_required=True,
            reason="Proxy-risk above threshold",
        )

    return VerificationOutcome(
        passed=passed,
        proxy_risk=proxy_risk,
        escalation_required=not passed,
        reason="ok" if passed else "Verification failed",
    )


__all__ = [
    "VerifierTier",
    "VerifierSignal",
    "VerificationOutcome",
    "compute_proxy_risk",
    "enforce_no_self_grading",
]
