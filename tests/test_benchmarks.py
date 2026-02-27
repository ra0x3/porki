"""Tests for v4 benchmark harness scoring and baseline aggregation."""

from __future__ import annotations

from porki.benchmarks import BenchmarkCase, BenchmarkHarness, BenchmarkObservation


def test_benchmark_harness_scores_successful_fast_lane_case():
    harness = BenchmarkHarness()
    case = BenchmarkCase(case_id="c1", goal_class="Transform")
    obs = BenchmarkObservation(
        drift=0.05,
        safety=0.95,
        proxy_resistance=0.9,
        latency_ms=900,
        cost_tokens=2000,
    )

    result = harness.score(case, obs, lane="fast")

    assert result.success is True
    assert result.within_latency_budget is True
    assert result.within_cost_budget is True
    assert result.score >= 0.7


def test_benchmark_harness_marks_budget_failures():
    harness = BenchmarkHarness()
    case = BenchmarkCase(case_id="c2", goal_class="Outreach")
    obs = BenchmarkObservation(
        drift=0.1,
        safety=0.9,
        proxy_resistance=0.8,
        latency_ms=7000,
        cost_tokens=9000,
    )

    result = harness.score(case, obs, lane="guarded")

    assert result.success is False
    assert result.within_latency_budget is False
    assert result.within_cost_budget is False


def test_benchmark_harness_computes_baseline_summary():
    harness = BenchmarkHarness()
    results = [
        harness.score(
            BenchmarkCase(case_id="c1", goal_class="Transform"),
            BenchmarkObservation(
                drift=0.05,
                safety=0.95,
                proxy_resistance=0.9,
                latency_ms=900,
                cost_tokens=2000,
            ),
            lane="fast",
        ),
        harness.score(
            BenchmarkCase(case_id="c2", goal_class="Outreach"),
            BenchmarkObservation(
                drift=0.2,
                safety=0.8,
                proxy_resistance=0.7,
                latency_ms=4500,
                cost_tokens=7000,
            ),
            lane="guarded",
        ),
    ]

    baseline = harness.baseline(results)

    assert set(baseline.keys()) == {"pass_rate", "mean_score"}
    assert 0.0 <= baseline["pass_rate"] <= 1.0
