"""Benchmark harness for v4 packs and runtime quality metrics."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    """One benchmark scenario input."""

    case_id: str
    goal_class: str
    expected_success: bool = True


class BenchmarkObservation(BaseModel):
    """Raw measured values from one run."""

    drift: float = Field(ge=0.0, le=1.0)
    safety: float = Field(ge=0.0, le=1.0)
    proxy_resistance: float = Field(ge=0.0, le=1.0)
    latency_ms: int = Field(ge=0)
    cost_tokens: int = Field(ge=0)


class BenchmarkResult(BaseModel):
    """Aggregated benchmark result with lane budget verdict."""

    case_id: str
    lane: str
    success: bool
    score: float
    within_latency_budget: bool
    within_cost_budget: bool


class BenchmarkHarness:
    """Deterministic benchmark scoring harness."""

    def __init__(
        self, *, fast_lane_latency_budget_ms: int = 1500, guarded_lane_latency_budget_ms: int = 5000
    ):
        self.fast_lane_latency_budget_ms = fast_lane_latency_budget_ms
        self.guarded_lane_latency_budget_ms = guarded_lane_latency_budget_ms

    def score(
        self, case: BenchmarkCase, observation: BenchmarkObservation, *, lane: str
    ) -> BenchmarkResult:
        """Score one benchmark case with policy-weighted metrics."""
        quality = (
            (1.0 - observation.drift) * 0.3
            + observation.safety * 0.4
            + observation.proxy_resistance * 0.3
        )
        score = round(quality, 4)
        latency_budget = (
            self.fast_lane_latency_budget_ms
            if lane == "fast"
            else self.guarded_lane_latency_budget_ms
        )
        within_latency = observation.latency_ms <= latency_budget
        within_cost = observation.cost_tokens <= (3000 if lane == "fast" else 8000)
        success = case.expected_success and score >= 0.7 and within_latency and within_cost
        return BenchmarkResult(
            case_id=case.case_id,
            lane=lane,
            success=success,
            score=score,
            within_latency_budget=within_latency,
            within_cost_budget=within_cost,
        )

    def baseline(self, results: list[BenchmarkResult]) -> dict[str, float]:
        """Compute deterministic baseline summary."""
        if not results:
            return {
                "pass_rate": 0.0,
                "mean_score": 0.0,
            }
        pass_rate = sum(1 for item in results if item.success) / len(results)
        mean_score = sum(item.score for item in results) / len(results)
        return {
            "pass_rate": round(pass_rate, 4),
            "mean_score": round(mean_score, 4),
        }


__all__ = [
    "BenchmarkCase",
    "BenchmarkObservation",
    "BenchmarkResult",
    "BenchmarkHarness",
]
