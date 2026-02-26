import logging
from datetime import datetime, timedelta, timezone

from porki.llm import (
    LLMClient,
    RecoveryDecision,
    StubLLMClient,
    TaskExecutionResult,
    TaskSelection,
)
from porki.models import DagModel, TaskEdge, TaskNode, TaskState, TaskStatus
from porki.runtime import AgentRuntime


def _prepare_dag(goal_id: str) -> DagModel:
    """Build a single-node DAG fixture."""
    return DagModel(
        goal_id=goal_id,
        nodes=[TaskNode(id="task-1", title="Initial task", priority=1)],
        edges=[],
    )


class ScriptedLLM(LLMClient):
    """Deterministic client for role-transition tests."""

    def __init__(self) -> None:
        """Initialize scripted QA failure counter."""
        self.qa_failures = 1

    def create_goal_dag(self, instructions: str, *, goal_id: str) -> DagModel:
        """Unused in these runtime tests."""
        raise NotImplementedError

    def select_next_task(
        self, ready_nodes, *, memory, goal_id: str, instructions: str
    ) -> TaskSelection:
        """Select the first ready task deterministically."""
        first = next(iter(ready_nodes), None)
        if first is None:
            return TaskSelection(selected_task_id=None, justification="No tasks", confidence=0.0)
        return TaskSelection(selected_task_id=first.id, justification="Pick first", confidence=1.0)

    def execute_task(
        self, task: TaskNode, *, goal_id: str, instructions: str, memory
    ) -> TaskExecutionResult:
        """Fail first QA run, then return successful execution."""
        phase = task.metadata.get("phase", "development")
        if phase == "qa" and self.qa_failures > 0:
            self.qa_failures -= 1
            return TaskExecutionResult(
                status="failed",
                outputs=[],
                notes=f"QA failed for {task.id}",
                follow_ups=[],
            )
        return TaskExecutionResult(
            status="done",
            outputs=[f"artifact://{task.id}.txt"],
            notes=f"Completed {task.id}",
            follow_ups=[],
        )

    def summarize_task(
        self,
        task: TaskNode,
        execution: TaskExecutionResult,
        *,
        goal_id: str,
        instructions: str,
        memory,
    ) -> str:
        """Return deterministic summary text."""
        return f"{task.id}: {execution.notes}"

    def assess_recovery(
        self,
        task: TaskNode,
        *,
        error: str,
        goal_id: str,
        instructions: str,
        memory,
    ) -> RecoveryDecision:
        """Unused in QA workflow tests."""
        return RecoveryDecision(
            recoverable=False,
            reason="Not used in this test",
            remediation_title="",
            remediation_steps=[],
            confidence=0.0,
        )


class FailingOnceLLM(StubLLMClient):
    """Fail first execution, then succeed."""

    def __init__(self, error_text: str):
        super().__init__()
        self.error_text = error_text
        self.calls = 0

    def execute_task(
        self,
        task: TaskNode,
        *,
        goal_id: str,
        instructions: str,
        memory,
    ) -> TaskExecutionResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError(self.error_text)
        return super().execute_task(task, goal_id=goal_id, instructions=instructions, memory=memory)


class NoSelectLLM(StubLLMClient):
    """Fail fast if selection is attempted."""

    def select_next_task(
        self, ready_nodes, *, memory, goal_id: str, instructions: str
    ) -> TaskSelection:
        raise AssertionError("select_next_task should not run during spending-cap backoff")


class DeclineSelectionLLM(StubLLMClient):
    """Return a deterministic selection decline for role-mismatch tests."""

    def __init__(self, justification: str):
        super().__init__()
        self.justification = justification

    def select_next_task(
        self, ready_nodes, *, memory, goal_id: str, instructions: str
    ) -> TaskSelection:
        return TaskSelection(
            selected_task_id=None,
            justification=self.justification,
            confidence=1.0,
        )


class CountingHeartbeatRuntime(AgentRuntime):
    """Runtime test helper to count heartbeat file reads."""

    def __init__(self, *args, **kwargs):
        """Initialize counter before base runtime setup."""
        self.heartbeat_reads = 0
        super().__init__(*args, **kwargs)

    def poll_heartbeat(self):  # type: ignore[override]
        """Count heartbeat reads and return no directives."""
        self.heartbeat_reads += 1
        return []


class CountingReloadRuntime(AgentRuntime):
    """Runtime test helper to count instruction reloads."""

    def __init__(self, *args, **kwargs):
        """Initialize counter before base runtime setup."""
        self.reload_calls = 0
        super().__init__(*args, **kwargs)

    def reload_instructions(self) -> None:  # type: ignore[override]
        """Count reload calls and run normal behavior."""
        self.reload_calls += 1
        super().reload_instructions()


def test_agent_completes_task(redis_store, tmp_path, assets_dir):
    """Agent should complete a development task."""
    instructions_src = assets_dir / "instructions" / "agent-research.md"
    heartbeat_src = assets_dir / "instructions" / "heartbeat" / "agent-research.md"
    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text(instructions_src.read_text(), encoding="utf-8")
    heartbeat_path.write_text(heartbeat_src.read_text(), encoding="utf-8")

    goal_id = "goal-demo"
    redis_store.write_dag(_prepare_dag(goal_id))

    agent = AgentRuntime(
        agent_name="agent-research",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
        lease_ttl=timedelta(seconds=5),
    )

    agent.run(max_cycles=3)
    state = redis_store.get_task_state("task-1")
    assert state is not None
    assert state.status is TaskStatus.DEV_DONE
    assert state.progress.startswith("Task task-1 completed")


def test_agent_pause_directive(redis_store, tmp_path, assets_dir):
    """Pause directive should skip execution cycle."""
    instructions_src = assets_dir / "instructions" / "agent-research.md"
    instructions_path = tmp_path / "instructions.md"
    instructions_path.write_text(instructions_src.read_text(), encoding="utf-8")
    heartbeat_path = tmp_path / "heartbeat.md"
    heartbeat_path.write_text("PAUSE\n", encoding="utf-8")

    goal_id = "goal-demo"
    redis_store.write_dag(_prepare_dag(goal_id))

    agent = AgentRuntime(
        agent_name="agent-research",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )

    agent.run(max_cycles=1)
    state = redis_store.get_task_state("task-1")
    assert state is not None
    assert state.status is not TaskStatus.DEV_DONE


def test_reload_instructions_warns_only_when_file_missing(redis_store, tmp_path, caplog):
    """Instruction-missing warning should be emitted only for absent files."""
    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="agent-research",
        goal_id="goal-warn-on-missing",
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )

    with caplog.at_level(logging.WARNING):
        agent.reload_instructions()

    existing_path_warnings = [
        rec for rec in caplog.records if "Instructions file missing" in rec.getMessage()
    ]
    assert not existing_path_warnings

    caplog.clear()
    missing_path = tmp_path / "missing-instructions.md"
    agent.instructions_path = missing_path

    with caplog.at_level(logging.WARNING):
        agent.reload_instructions()

    missing_path_warnings = [
        rec for rec in caplog.records if "Instructions file missing" in rec.getMessage()
    ]
    assert len(missing_path_warnings) == 1
    assert str(missing_path) in missing_path_warnings[0].getMessage()


def test_role_restriction_blocks_wrong_agent(redis_store, tmp_path):
    """Agent should not claim task assigned to another role."""
    goal_id = "goal-role-check"
    dag = DagModel(
        goal_id=goal_id,
        nodes=[
            TaskNode(
                id="task-dev",
                title="Build feature",
                priority=1,
                metadata={"phase": "development", "required_role": "features-dev"},
            )
        ],
        edges=[],
    )
    redis_store.write_dag(dag)
    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="qa-dev",
        agent_role="qa-dev",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )
    agent.run(max_cycles=1)

    state = redis_store.get_task_state("task-dev")
    assert state is not None
    assert state.status is TaskStatus.READY


def test_selection_decline_reassigns_explicit_role_mismatch(redis_store, tmp_path):
    """Selection-time role mismatch decline should reassign the single ready task."""
    goal_id = "goal-role-reassign"
    dag = DagModel(
        goal_id=goal_id,
        nodes=[
            TaskNode(
                id="task-004",
                title="Build Dashboard Component",
                priority=1,
                metadata={
                    "phase": "development",
                    "required_role": "core-infra-dev",
                    "dev_role": "core-infra-dev",
                },
            )
        ],
        edges=[],
    )
    redis_store.write_dag(dag)
    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    llm = DeclineSelectionLLM(
        "Task-004 requires creating UI components which belongs to ui-dev role, "
        "not core-infra-dev. As core-infra-dev, I must reject this task."
    )
    agent = AgentRuntime(
        agent_name="core-infra-dev",
        agent_role="core-infra-dev",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=llm,
        loop_interval=0,
    )

    agent.run(max_cycles=1)

    node = redis_store.get_task_node(goal_id, "task-004")
    assert node is not None
    assert node.metadata["required_role"] == "ui-dev"
    assert node.metadata["dev_role"] == "ui-dev"

    state = redis_store.get_task_state("task-004")
    assert state is not None
    assert state.status is TaskStatus.READY


def test_selection_decline_without_role_hint_does_not_reassign(redis_store, tmp_path):
    """Generic selection decline should not mutate task role assignment."""
    goal_id = "goal-no-role-reassign"
    dag = DagModel(
        goal_id=goal_id,
        nodes=[
            TaskNode(
                id="task-004",
                title="Build Dashboard Component",
                priority=1,
                metadata={
                    "phase": "development",
                    "required_role": "core-infra-dev",
                    "dev_role": "core-infra-dev",
                },
            )
        ],
        edges=[],
    )
    redis_store.write_dag(dag)
    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="core-infra-dev",
        agent_role="core-infra-dev",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=DeclineSelectionLLM("No task is safe to select right now."),
        loop_interval=0,
    )

    agent.run(max_cycles=1)

    node = redis_store.get_task_node(goal_id, "task-004")
    assert node is not None
    assert node.metadata["required_role"] == "core-infra-dev"
    assert node.metadata["dev_role"] == "core-infra-dev"


def test_iterative_qa_loop_creates_remediation(redis_store, tmp_path):
    """QA failure should create remediation and pass after fix."""
    goal_id = "goal-iterative"
    dag = DagModel(
        goal_id=goal_id,
        nodes=[
            TaskNode(
                id="task-dev",
                title="Build feature",
                priority=1,
                metadata={
                    "phase": "development",
                    "required_role": "features-dev",
                    "dev_role": "features-dev",
                },
            ),
            TaskNode(
                id="task-dev__qa",
                title="QA review",
                priority=1,
                metadata={
                    "phase": "qa",
                    "required_role": "qa-dev",
                    "parent_task_id": "task-dev",
                    "review_cycle": "0",
                    "dev_role": "features-dev",
                },
            ),
            TaskNode(
                id="task-dev__integrate",
                title="Integrate",
                priority=1,
                metadata={"phase": "integration", "required_role": "team-lead"},
            ),
        ],
        edges=[
            TaskEdge(source="task-dev", target="task-dev__qa"),
            TaskEdge(source="task-dev__qa", target="task-dev__integrate"),
        ],
    )
    redis_store.write_dag(dag)
    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    llm = ScriptedLLM()

    features_agent = AgentRuntime(
        agent_name="features-dev",
        agent_role="features-dev",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=llm,
        loop_interval=0,
    )
    qa_agent = AgentRuntime(
        agent_name="qa-dev",
        agent_role="qa-dev",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=llm,
        loop_interval=0,
    )
    lead_agent = AgentRuntime(
        agent_name="team-lead",
        agent_role="team-lead",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=llm,
        loop_interval=0,
    )

    features_agent.run(max_cycles=1)
    state = redis_store.get_task_state("task-dev")
    assert state is not None
    assert state.status is TaskStatus.DEV_DONE

    qa_agent.run(max_cycles=2)
    qa_state = redis_store.get_task_state("task-dev__qa")
    assert qa_state is not None
    assert qa_state.status is TaskStatus.BLOCKED

    remediation_nodes = [
        node
        for node in redis_store.read_dag(goal_id).nodes  # type: ignore[union-attr]
        if node.id.startswith("task-dev__qa__fix_")
    ]
    assert remediation_nodes
    remediation_id = remediation_nodes[0].id
    assert remediation_nodes[0].metadata["required_role"] == "features-dev"

    features_agent.run(max_cycles=1)
    remediation_state = redis_store.get_task_state(remediation_id)
    assert remediation_state is not None
    assert remediation_state.status is TaskStatus.DEV_DONE

    qa_agent.run(max_cycles=1)
    qa_state = redis_store.get_task_state("task-dev__qa")
    assert qa_state is not None
    assert qa_state.status is TaskStatus.QA_PASSED

    lead_agent.run(max_cycles=1)
    integration_state = redis_store.get_task_state("task-dev__integrate")
    assert integration_state is not None
    assert integration_state.status is TaskStatus.DONE


def test_agent_retries_stale_running_task_after_restart(redis_store, tmp_path):
    """Agent should resume by reclaiming stale running work on startup."""
    goal_id = "goal-resume"
    redis_store.write_dag(_prepare_dag(goal_id))
    redis_store.update_task_state(
        "task-1",
        TaskState(
            status=TaskStatus.RUNNING,
            owner="agent-crashed",
            lease_expires=datetime.now(timezone.utc) - timedelta(seconds=60),
        ),
    )

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="agent-research",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
        lease_ttl=timedelta(seconds=5),
    )
    agent.run(max_cycles=1)

    state = redis_store.get_task_state("task-1")
    assert state is not None
    assert state.status is TaskStatus.DEV_DONE


def test_agent_reads_heartbeat_every_120_seconds_by_default(redis_store, tmp_path):
    """Heartbeat directives should be read on the configured refresh interval."""
    goal_id = "goal-heartbeat-interval"
    redis_store.write_dag(_prepare_dag(goal_id))

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = CountingHeartbeatRuntime(
        agent_name="agent-research",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
        lease_ttl=timedelta(seconds=5),
    )
    agent.run(max_cycles=3)

    assert agent.heartbeat_reads == 1


def test_agent_reads_instructions_every_120_seconds_by_default(redis_store, tmp_path):
    """Instruction file should not be reloaded each cycle under default interval."""
    goal_id = "goal-instruction-interval"
    redis_store.write_dag(_prepare_dag(goal_id))

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = CountingReloadRuntime(
        agent_name="agent-research",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
        lease_ttl=timedelta(seconds=5),
    )
    agent.run(max_cycles=3)

    assert agent.reload_calls == 1


def test_recoverable_runtime_error_creates_recovery_task(redis_store, tmp_path):
    """Recoverable runtime errors should block original task and enqueue remediation."""
    goal_id = "goal-recovery-loop"
    redis_store.write_dag(_prepare_dag(goal_id))

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    llm = FailingOnceLLM("node: command not found")
    agent = AgentRuntime(
        agent_name="features-dev",
        agent_role="features-dev",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=llm,
        loop_interval=0,
    )
    agent.run(max_cycles=1)

    state = redis_store.get_task_state("task-1")
    assert state is not None
    assert state.status is TaskStatus.BLOCKED
    assert "created remediation task" in (state.progress or "")

    dag = redis_store.read_dag(goal_id)
    assert dag is not None
    recovery_nodes = [node for node in dag.nodes if node.id.startswith("task-1__recover_")]
    assert recovery_nodes
    recovery_state = redis_store.get_task_state(recovery_nodes[0].id)
    assert recovery_state is not None
    assert recovery_state.status is TaskStatus.READY


def test_recovery_attempt_cap_marks_terminal_failure(redis_store, tmp_path):
    """Tasks should fail after bounded recovery attempts."""
    goal_id = "goal-recovery-cap"
    dag = DagModel(
        goal_id=goal_id,
        nodes=[
            TaskNode(
                id="task-1",
                title="Initial task",
                priority=1,
                metadata={"recovery_attempts": str(AgentRuntime.MAX_RECOVERY_ATTEMPTS)},
            )
        ],
        edges=[],
    )
    redis_store.write_dag(dag)

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    llm = FailingOnceLLM("node: command not found")
    agent = AgentRuntime(
        agent_name="features-dev",
        agent_role="features-dev",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=llm,
        loop_interval=0,
    )
    agent.run(max_cycles=1)

    state = redis_store.get_task_state("task-1")
    assert state is not None
    assert state.status is TaskStatus.FAILED


def test_goal_spending_cap_backoff_skips_work_cycle(redis_store, tmp_path):
    """Active goal-level spending cap should prevent task selection attempts."""
    goal_id = "goal-cap-backoff"
    redis_store.write_dag(_prepare_dag(goal_id))
    redis_store.set_goal_spending_cap_until(
        goal_id,
        datetime.now(timezone.utc) + timedelta(seconds=60),
    )

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="features-dev",
        agent_role="features-dev",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=NoSelectLLM(),
        loop_interval=0,
    )
    agent.run(max_cycles=1)

    state = redis_store.get_task_state("task-1")
    assert state is not None
    assert state.status is TaskStatus.READY


def test_duplicate_agent_runtime_is_rejected(redis_store, tmp_path):
    """Second runtime with same agent name should refuse startup."""
    goal_id = "goal-duplicate-agent"
    redis_store.write_dag(_prepare_dag(goal_id))
    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    first = AgentRuntime(
        agent_name="agent-research",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )
    second = AgentRuntime(
        agent_name="agent-research",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )

    assert redis_store.acquire_agent_active_slot(
        "agent-research", pid=first._pid, ttl=timedelta(seconds=30)
    )
    second.run(max_cycles=1)
    state = redis_store.get_task_state("task-1")
    assert state is not None
    assert state.status is TaskStatus.READY
    redis_store.release_agent_active_slot("agent-research", pid=first._pid)


def test_recovery_failure_requeues_at_root_without_nested_ids(redis_store, tmp_path):
    """Recovery retries should target the root task instead of nested recovery chains."""
    goal_id = "goal-recovery-root"
    root = TaskNode(
        id="task-1",
        title="Root task",
        priority=1,
        metadata={"recovery_attempts": "1", "required_role": "features-dev"},
    )
    recovery = TaskNode(
        id="task-1__recover_1",
        title="Recover root",
        priority=2,
        metadata={
            "phase": "development",
            "required_role": "features-dev",
            "parent_task_id": "task-1",
            "recovery_for": "task-1",
            "recovery_attempt": "1",
            "recovery_attempts": "1",
        },
    )
    dag = DagModel(
        goal_id=goal_id,
        nodes=[root, recovery],
        edges=[TaskEdge(source=recovery.id, target=root.id)],
    )
    redis_store.write_dag(dag)
    redis_store.update_task_state(
        recovery.id, TaskState(status=TaskStatus.RUNNING, owner="features-dev")
    )

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="features-dev",
        agent_role="features-dev",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )

    agent._record_failure_or_recovery(recovery, "timeout talking to provider")
    dag_after = redis_store.read_dag(goal_id)
    assert dag_after is not None
    ids = {node.id for node in dag_after.nodes}
    assert "task-1__recover_2" in ids
    assert not any("__recover_1__recover_" in node_id for node_id in ids)
    root_after = redis_store.get_task_node(goal_id, "task-1")
    assert root_after is not None
    assert root_after.metadata["recovery_attempts"] == "2"


def test_goal_completion_logs_once_when_no_work_left(redis_store, tmp_path, caplog):
    """Runtime should emit a concise completion summary once per unchanged state."""
    goal_id = "goal-complete-log"
    redis_store.write_dag(_prepare_dag(goal_id))

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="agent-research",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )
    with caplog.at_level(logging.INFO):
        agent.run(max_cycles=3)

    completion_logs = [
        rec
        for rec in caplog.records
        if "Goal goal-complete-log complete: 0 tasks left" in rec.getMessage()
    ]
    assert len(completion_logs) == 1


def test_goal_not_complete_when_failed_tasks_remain(redis_store, tmp_path, caplog):
    """Runtime should not report completion when failed tasks remain."""
    goal_id = "goal-incomplete-failures"
    redis_store.write_dag(_prepare_dag(goal_id))
    redis_store.update_task_state("task-1", TaskState(status=TaskStatus.FAILED, last_error="boom"))

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="qa-dev",
        agent_role="qa-dev",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )
    with caplog.at_level(logging.INFO):
        agent.run(max_cycles=1)

    complete_logs = [
        rec
        for rec in caplog.records
        if "Goal goal-incomplete-failures complete" in rec.getMessage()
    ]
    assert complete_logs == []
    incomplete_logs = [
        rec
        for rec in caplog.records
        if "Goal goal-incomplete-failures not complete" in rec.getMessage()
    ]
    assert len(incomplete_logs) == 1


def test_logs_blocked_when_ready_tasks_require_inactive_roles(redis_store, tmp_path, caplog):
    """Runtime should explain goal-wide blockage when required roles are inactive."""
    goal_id = "goal-inactive-role"
    dag = DagModel(
        goal_id=goal_id,
        nodes=[
                TaskNode(
                    id="task-owner-only",
                    title="Owner work",
                    priority=1,
                    metadata={
                        "phase": "integration",
                        "required_role": "owner",
                        "role_assignment": "hard",
                    },
                )
        ],
        edges=[],
    )
    redis_store.write_dag(dag)

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="team-lead",
        agent_role="team-lead",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )
    with caplog.at_level(logging.WARNING):
        agent.run(max_cycles=1)

    blocked_logs = [
        rec for rec in caplog.records if "Goal goal-inactive-role blocked" in rec.getMessage()
    ]
    assert blocked_logs
    assert "inactive roles owner" in blocked_logs[0].getMessage()


def test_coordinator_reassigns_soft_ready_task_from_inactive_builder_role(
    redis_store, tmp_path, caplog
):
    """Owner/team-lead should deterministically reassign soft builder work when blocked."""
    goal_id = "goal-soft-role-reassign"
    dag = DagModel(
        goal_id=goal_id,
        nodes=[
                TaskNode(
                    id="task-features",
                    title="Feature task",
                    priority=1,
                    metadata={
                        "phase": "development",
                        "required_role": "features-dev",
                        "role_assignment": "soft",
                    },
                )
        ],
        edges=[],
    )
    redis_store.write_dag(dag)
    redis_store.register_agent("ui-dev-worker", pid=11, capabilities={"role": "ui-dev"})
    redis_store.heartbeat_agent("ui-dev-worker", ttl=timedelta(seconds=30))
    redis_store.register_agent("qa-worker", pid=12, capabilities={"role": "qa-dev"})
    redis_store.heartbeat_agent("qa-worker", ttl=timedelta(seconds=30))

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    coordinator = AgentRuntime(
        agent_name="owner",
        agent_role="owner",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )
    with caplog.at_level(logging.INFO):
        coordinator.run(max_cycles=1)

    node = redis_store.get_task_node(goal_id, "task-features")
    assert node is not None
    assert node.metadata["required_role"] == "ui-dev"
    assert node.metadata["original_required_role"] == "features-dev"
    assert node.metadata["last_role_reassignment_reason"] == "inactive-required-role"

    remediation_logs = [
        rec
        for rec in caplog.records
        if "Remediated inactive-role stall by reassigning 1 ready task(s)" in rec.getMessage()
    ]
    assert len(remediation_logs) == 1


def test_coordinator_remediates_orchestrator_style_inactive_builder_deadlock(
    redis_store, tmp_path, caplog
):
    """Coordinator should reassign multiple ready builder tasks when only inactive builder roles own them."""
    goal_id = "goal-orchestrator-ui-deadlock"
    nodes = [
        TaskNode(
            id=f"task-core-{idx}",
            title=f"Core task {idx}",
            priority=10 - idx,
            metadata={
                "phase": "development",
                "required_role": "core-infra-dev",
                "role_assignment": "soft",
            },
        )
        for idx in range(1, 4)
    ] + [
        TaskNode(
            id=f"task-feat-{idx}",
            title=f"Feature task {idx}",
            priority=7 - idx,
            metadata={
                "phase": "development",
                "required_role": "features-dev",
                "role_assignment": "soft",
            },
        )
        for idx in range(1, 4)
    ]
    dag = DagModel(goal_id=goal_id, nodes=nodes, edges=[])
    redis_store.write_dag(dag)

    # Active roles mirror the stalled example: no core-infra-dev / features-dev.
    for pid, (agent_name, role) in enumerate(
        [
            ("owner-1", "owner"),
            ("lead-1", "team-lead"),
            ("ui-1", "ui-dev"),
            ("qa-1", "qa-dev"),
        ],
        start=30,
    ):
        redis_store.register_agent(agent_name, pid=pid, capabilities={"role": role})
        redis_store.heartbeat_agent(agent_name, ttl=timedelta(seconds=30))

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    coordinator = AgentRuntime(
        agent_name="owner",
        agent_role="owner",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )
    with caplog.at_level(logging.INFO):
        coordinator.run(max_cycles=1)

    reassigned_to_ui = 0
    for node_id in [f"task-core-{i}" for i in range(1, 4)] + [f"task-feat-{i}" for i in range(1, 4)]:
        node = redis_store.get_task_node(goal_id, node_id)
        assert node is not None
        assert node.metadata["role_assignment"] == "soft"
        assert node.metadata["required_role"] == "ui-dev"
        assert node.metadata["original_required_role"] in {"core-infra-dev", "features-dev"}
        reassigned_to_ui += 1
    assert reassigned_to_ui == 6

    remediation_logs = [
        rec
        for rec in caplog.records
        if "Remediated inactive-role stall by reassigning 6 ready task(s)" in rec.getMessage()
    ]
    assert len(remediation_logs) == 1


def test_coordinator_does_not_reassign_hard_integration_task(redis_store, tmp_path):
    """Integration tasks remain hard-pinned even when the required role is inactive."""
    goal_id = "goal-hard-role-remains"
    dag = DagModel(
        goal_id=goal_id,
        nodes=[
            TaskNode(
                id="task-integrate",
                title="Integration step",
                priority=1,
                metadata={"phase": "integration", "required_role": "owner"},
            )
        ],
        edges=[],
    )
    redis_store.write_dag(dag)
    redis_store.register_agent("ui-dev-worker", pid=21, capabilities={"role": "ui-dev"})
    redis_store.heartbeat_agent("ui-dev-worker", ttl=timedelta(seconds=30))

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    coordinator = AgentRuntime(
        agent_name="team-lead",
        agent_role="team-lead",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )
    coordinator.run(max_cycles=1)

    node = redis_store.get_task_node(goal_id, "task-integrate")
    assert node is not None
    assert node.metadata["required_role"] == "owner"


def test_no_eligible_log_is_deduplicated_for_unchanged_queue(redis_store, tmp_path, caplog):
    """Repeated unchanged no-eligible cycles should not spam INFO logs."""
    goal_id = "goal-no-eligible-dedupe"
    dag = DagModel(
        goal_id=goal_id,
        nodes=[
            TaskNode(
                id="task-features",
                title="Features work",
                priority=1,
                metadata={"phase": "development", "required_role": "features-dev"},
            )
        ],
        edges=[],
    )
    redis_store.write_dag(dag)

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="qa-dev",
        agent_role="qa-dev",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )
    with caplog.at_level(logging.INFO):
        agent.run(max_cycles=3)

    no_eligible_logs = [
        rec
        for rec in caplog.records
        if "No eligible ready tasks for role qa-dev" in rec.getMessage()
    ]
    assert len(no_eligible_logs) == 1


def test_idle_heartbeat_logs_periodically_when_still_idle(redis_store, tmp_path, caplog):
    """Runtime should emit recurring idle summaries instead of going fully quiet."""
    goal_id = "goal-idle-heartbeat"
    dag = DagModel(
        goal_id=goal_id,
        nodes=[
            TaskNode(
                id="task-features",
                title="Features work",
                priority=1,
                metadata={"phase": "development", "required_role": "features-dev"},
            )
        ],
        edges=[],
    )
    redis_store.write_dag(dag)

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="qa-dev",
        agent_role="qa-dev",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
        idle_log_interval=timedelta(seconds=0),
    )
    with caplog.at_level(logging.INFO):
        agent.run(max_cycles=3)

    idle_logs = [rec for rec in caplog.records if "Idle (no-eligible-tasks)" in rec.getMessage()]
    assert len(idle_logs) >= 3


def test_startup_logs_stale_recovery_summary(redis_store, tmp_path, caplog):
    """Runtime should report stale-task recovery status at startup."""
    goal_id = "goal-startup-recovery-log"
    redis_store.write_dag(_prepare_dag(goal_id))
    redis_store.update_task_state(
        "task-1",
        TaskState(
            status=TaskStatus.RUNNING,
            owner="dead-agent",
            lease_expires=datetime.now(timezone.utc) - timedelta(seconds=60),
        ),
    )

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("PAUSE\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="agent-research",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )
    with caplog.at_level(logging.INFO):
        agent.run(max_cycles=1)

    resume_logs = [
        rec for rec in caplog.records if "Startup recovered 1 stale task(s)" in rec.getMessage()
    ]
    assert len(resume_logs) == 1


def test_auto_unblocks_when_blocked_by_failed_dependency(redis_store, tmp_path, caplog):
    """Runtime should create remediation work for failed prerequisites that block progress."""
    goal_id = "goal-auto-unblock"
    dag = DagModel(
        goal_id=goal_id,
        nodes=[
            TaskNode(
                id="task-failed",
                title="Failed prerequisite",
                priority=3,
                metadata={"phase": "development", "required_role": "features-dev"},
            ),
            TaskNode(
                id="task-blocked",
                title="Downstream work",
                priority=2,
                metadata={"phase": "development", "required_role": "features-dev"},
            ),
        ],
        edges=[TaskEdge(source="task-failed", target="task-blocked")],
    )
    redis_store.write_dag(dag)
    redis_store.update_task_state(
        "task-failed",
        TaskState(status=TaskStatus.FAILED, last_error="compile error"),
    )
    redis_store.update_task_state("task-blocked", TaskState(status=TaskStatus.BLOCKED))

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="owner",
        agent_role="owner",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )
    with caplog.at_level(logging.INFO):
        agent.run(max_cycles=1)

    dag_after = redis_store.read_dag(goal_id)
    assert dag_after is not None
    recovery_ids = sorted(
        node.id for node in dag_after.nodes if node.id.startswith("task-failed__recover_")
    )
    assert len(recovery_ids) == 1

    failed_state = redis_store.get_task_state("task-failed")
    assert failed_state is not None
    assert failed_state.status is TaskStatus.BLOCKED

    triggered_logs = [
        rec
        for rec in caplog.records
        if "Auto-unblock for failed dependency task-failed" in rec.getMessage()
    ]
    assert len(triggered_logs) == 1


def test_non_coordinator_does_not_run_auto_unblock(redis_store, tmp_path):
    """Non-owner/team-lead runtimes should not mutate failed dependency deadlocks."""
    goal_id = "goal-auto-unblock-non-coordinator"
    dag = DagModel(
        goal_id=goal_id,
        nodes=[
            TaskNode(id="task-failed", title="Failed prerequisite", priority=3),
            TaskNode(id="task-blocked", title="Downstream work", priority=2),
        ],
        edges=[TaskEdge(source="task-failed", target="task-blocked")],
    )
    redis_store.write_dag(dag)
    redis_store.update_task_state("task-failed", TaskState(status=TaskStatus.FAILED))
    redis_store.update_task_state("task-blocked", TaskState(status=TaskStatus.BLOCKED))

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="features-dev",
        agent_role="features-dev",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )
    agent.run(max_cycles=1)

    failed_state = redis_store.get_task_state("task-failed")
    assert failed_state is not None
    assert failed_state.status is TaskStatus.FAILED
    dag_after = redis_store.read_dag(goal_id)
    assert dag_after is not None
    recovery_ids = [
        node.id for node in dag_after.nodes if node.id.startswith("task-failed__recover_")
    ]
    assert recovery_ids == []


def test_coordinator_auto_unblocks_failed_dead_end(redis_store, tmp_path, caplog):
    """Coordinator should create remediation for failed dead-end tasks when queue is empty."""
    goal_id = "goal-auto-unblock-dead-end"
    dag = DagModel(
        goal_id=goal_id,
        nodes=[TaskNode(id="task-failed", title="Failed leaf", priority=3)],
        edges=[],
    )
    redis_store.write_dag(dag)
    redis_store.update_task_state(
        "task-failed", TaskState(status=TaskStatus.FAILED, last_error="boom")
    )

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="owner",
        agent_role="owner",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )
    with caplog.at_level(logging.INFO):
        agent.run(max_cycles=1)

    dag_after = redis_store.read_dag(goal_id)
    assert dag_after is not None
    recovery_ids = sorted(
        node.id for node in dag_after.nodes if node.id.startswith("task-failed__recover_")
    )
    assert len(recovery_ids) == 1
    triggered_logs = [
        rec
        for rec in caplog.records
        if "Auto-unblock for failed dependency task-failed" in rec.getMessage()
    ]
    assert len(triggered_logs) == 1


def test_exhausted_auto_unblock_escalates_once(redis_store, tmp_path, caplog):
    """Exhausted failed dependencies should create one escalation remediation task."""
    goal_id = "goal-auto-unblock-escalation"
    dag = DagModel(
        goal_id=goal_id,
        nodes=[
            TaskNode(
                id="task-failed",
                title="Failed prerequisite",
                priority=3,
                metadata={"recovery_attempts": "3"},
            ),
            TaskNode(id="task-blocked", title="Downstream work", priority=2),
        ],
        edges=[TaskEdge(source="task-failed", target="task-blocked")],
    )
    redis_store.write_dag(dag)
    redis_store.update_task_state("task-failed", TaskState(status=TaskStatus.FAILED))
    redis_store.update_task_state("task-blocked", TaskState(status=TaskStatus.BLOCKED))

    instructions_path = tmp_path / "instructions.md"
    heartbeat_path = tmp_path / "heartbeat.md"
    instructions_path.write_text("test", encoding="utf-8")
    heartbeat_path.write_text("RESUME\n", encoding="utf-8")

    agent = AgentRuntime(
        agent_name="owner",
        agent_role="owner",
        goal_id=goal_id,
        instructions_path=instructions_path,
        heartbeat_path=heartbeat_path,
        redis_store=redis_store,
        llm_client=StubLLMClient(),
        loop_interval=0,
    )
    with caplog.at_level(logging.WARNING):
        agent.run(max_cycles=2)

    dag_after = redis_store.read_dag(goal_id)
    assert dag_after is not None
    escalation_ids = sorted(
        node.id for node in dag_after.nodes if node.id.startswith("task-failed__recover_4")
    )
    assert len(escalation_ids) == 1

    failed_state = redis_store.get_task_state("task-failed")
    assert failed_state is not None
    assert failed_state.status is TaskStatus.BLOCKED

    escalated_logs = [
        rec
        for rec in caplog.records
        if "Escalated failed dependency task-failed" in rec.getMessage()
    ]
    assert len(escalated_logs) == 1
    exhausted_logs = [
        rec for rec in caplog.records if "exhausted auto-recovery attempts" in rec.getMessage()
    ]
    assert exhausted_logs == []
