"""Tests for v4 domain pack interfaces and reference behavior."""

from __future__ import annotations

from porki.domain_packs import (
    CodingPack,
    OutreachPack,
    PackContext,
    ResearchPack,
    default_domain_packs,
)
from porki.intent import GoalClass


def test_default_domain_packs_include_required_sets():
    packs = default_domain_packs()
    names = {pack.name for pack in packs}

    assert names == {"coding", "outreach", "research"}


def test_outreach_pack_fast_lane_prefers_draft_only():
    pack = OutreachPack()
    tasks = pack.build_tasks(PackContext(goal_statement="Reach out", lane="fast", max_tasks=3))

    assert pack.supports(GoalClass.OUTREACH)
    assert len(tasks) == 1
    assert tasks[0].id == "outreach.draft"


def test_coding_and_research_pack_goal_support_and_task_limits():
    coding = CodingPack()
    research = ResearchPack()

    coding_tasks = coding.build_tasks(
        PackContext(goal_statement="Refactor", lane="guarded", max_tasks=1)
    )
    research_tasks = research.build_tasks(
        PackContext(goal_statement="Investigate", lane="guarded", max_tasks=2)
    )

    assert coding.supports(GoalClass.TRANSFORM)
    assert research.supports(GoalClass.RETRIEVE_RESEARCH)
    assert len(coding_tasks) == 1
    assert len(research_tasks) == 2
