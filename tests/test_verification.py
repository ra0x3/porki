"""Tests for verifier policy enforcement and proxy-risk controls."""

from __future__ import annotations

from porki.verification import VerifierSignal, VerifierTier, enforce_no_self_grading


def test_enforce_no_self_grading_rejects_same_actor():
    outcome = enforce_no_self_grading(
        task_actor_id="agent-a",
        signals=[
            VerifierSignal(tier=VerifierTier.TIER3, actor_id="agent-a", score=0.7, passed=True)
        ],
        high_risk=False,
    )

    assert outcome.passed is False
    assert outcome.escalation_required is True
    assert "No-self-grading" in outcome.reason


def test_high_risk_requires_tier1_or_tier2():
    outcome = enforce_no_self_grading(
        task_actor_id="agent-a",
        signals=[
            VerifierSignal(tier=VerifierTier.TIER3, actor_id="verifier-a", score=0.8, passed=True),
            VerifierSignal(tier=VerifierTier.TIER3, actor_id="verifier-b", score=0.82, passed=True),
        ],
        high_risk=True,
    )

    assert outcome.passed is False
    assert outcome.escalation_required is True


def test_proxy_risk_escalates_when_diversity_is_low():
    outcome = enforce_no_self_grading(
        task_actor_id="agent-a",
        signals=[
            VerifierSignal(tier=VerifierTier.TIER2, actor_id="verifier-a", score=0.99, passed=True)
        ],
        high_risk=False,
    )

    assert outcome.passed is False
    assert outcome.proxy_risk >= 0.5
    assert outcome.escalation_required is True


def test_verification_passes_with_diverse_non_self_signals():
    outcome = enforce_no_self_grading(
        task_actor_id="agent-a",
        signals=[
            VerifierSignal(tier=VerifierTier.TIER1, actor_id="verifier-a", score=0.8, passed=True),
            VerifierSignal(tier=VerifierTier.TIER2, actor_id="verifier-b", score=0.83, passed=True),
        ],
        high_risk=True,
    )

    assert outcome.passed is True
    assert outcome.escalation_required is False
