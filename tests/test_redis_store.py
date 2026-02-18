from datetime import datetime, timedelta, timezone

from porki.models import DagModel, TaskEdge, TaskNode, TaskState, TaskStatus


def test_write_and_list_ready_tasks(redis_store):
    """RedisStore should surface ready tasks by dependency status."""
    dag = DagModel(
        goal_id="goal-demo",
        nodes=[
            TaskNode(id="task-1", title="Step 1", priority=1),
            TaskNode(id="task-2", title="Step 2", priority=0),
        ],
        edges=[TaskEdge(source="task-1", target="task-2")],
    )
    redis_store.write_dag(dag)
    ready = redis_store.list_ready_tasks("goal-demo")
    assert ready == ["task-1"], ready

    state = redis_store.get_task_state("task-1")
    redis_store.update_task_state("task-1", state.model_copy(update={"status": TaskStatus.DONE}))
    ready_after = redis_store.list_ready_tasks("goal-demo")
    assert ready_after[0] == "task-2"


def test_lock_cycle(redis_store):
    """RedisStore lock lifecycle should acquire, renew, and release."""
    dag = DagModel(goal_id="g", nodes=[TaskNode(id="t", title="T", priority=0)], edges=[])
    redis_store.write_dag(dag)
    assert redis_store.acquire_lock("t", "agent-a", timedelta(seconds=5)) is True
    assert redis_store.acquire_lock("t", "agent-b", timedelta(seconds=5)) is False
    assert redis_store.lock_owner("t") == "agent-a"
    assert redis_store.renew_lock("t", "agent-a", timedelta(seconds=5)) is True
    redis_store.release_lock("t", "agent-a")
    assert redis_store.lock_owner("t") is None


def test_list_ready_tasks_recovers_stale_running(redis_store):
    """Stale running tasks should be reset to ready for retry."""
    dag = DagModel(
        goal_id="goal-recover", nodes=[TaskNode(id="t1", title="T1", priority=1)], edges=[]
    )
    redis_store.write_dag(dag)
    redis_store.update_task_state(
        "t1",
        TaskState(
            status=TaskStatus.RUNNING,
            owner="agent-crashed",
            lease_expires=datetime.now(timezone.utc) - timedelta(seconds=60),
        ),
    )

    ready = redis_store.list_ready_tasks("goal-recover")
    assert ready == ["t1"]

    recovered_state = redis_store.get_task_state("t1")
    assert recovered_state is not None
    assert recovered_state.status is TaskStatus.READY
    assert recovered_state.owner is None
    assert recovered_state.lease_expires is None


def test_goal_spending_cap_roundtrip(redis_store):
    """Goal spending-cap deadline should round-trip while active."""
    goal_id = "goal-cap"
    until = datetime.now(timezone.utc) + timedelta(seconds=45)
    redis_store.set_goal_spending_cap_until(goal_id, until)

    fetched = redis_store.get_goal_spending_cap_until(goal_id)
    assert fetched is not None
    assert fetched >= datetime.now(timezone.utc)


def test_recover_stale_skips_active_owner_with_recent_heartbeat(redis_store):
    """Running task should not be recovered when owner heartbeat is still fresh."""
    dag = DagModel(
        goal_id="goal-active-owner", nodes=[TaskNode(id="t1", title="T1", priority=1)], edges=[]
    )
    redis_store.write_dag(dag)
    redis_store.update_task_state(
        "t1",
        TaskState(
            status=TaskStatus.RUNNING,
            owner="agent-live",
            lease_expires=datetime.now(timezone.utc) - timedelta(seconds=30),
        ),
    )
    redis_store.heartbeat_agent("agent-live", ttl=timedelta(seconds=60))

    ready = redis_store.list_ready_tasks("goal-active-owner")
    assert ready == []
    state = redis_store.get_task_state("t1")
    assert state is not None
    assert state.status is TaskStatus.RUNNING


def test_recover_stale_skips_running_with_execution_guard(redis_store):
    """Running task with active execution guard should not be recovered."""
    dag = DagModel(
        goal_id="goal-exec-guard", nodes=[TaskNode(id="t1", title="T1", priority=1)], edges=[]
    )
    redis_store.write_dag(dag)
    redis_store.update_task_state(
        "t1",
        TaskState(
            status=TaskStatus.RUNNING,
            owner="agent-live",
            lease_expires=datetime.now(timezone.utc) - timedelta(seconds=60),
        ),
    )
    assert redis_store.acquire_execution_guard("t1", "agent-live:token", timedelta(seconds=120))

    ready = redis_store.list_ready_tasks("goal-exec-guard")
    assert ready == []

    state = redis_store.get_task_state("t1")
    assert state is not None
    assert state.status is TaskStatus.RUNNING


def test_recover_stale_respects_lease_grace_without_heartbeat(redis_store):
    """Recent lease expiry should not recover immediately when heartbeat is absent."""
    dag = DagModel(
        goal_id="goal-lease-grace", nodes=[TaskNode(id="t1", title="T1", priority=1)], edges=[]
    )
    redis_store.write_dag(dag)
    redis_store.update_task_state(
        "t1",
        TaskState(
            status=TaskStatus.RUNNING,
            owner="agent-missing-heartbeat",
            lease_expires=datetime.now(timezone.utc) - timedelta(seconds=1),
        ),
    )

    assert redis_store.list_ready_tasks("goal-lease-grace") == []

    redis_store.update_task_state(
        "t1",
        TaskState(
            status=TaskStatus.RUNNING,
            owner="agent-missing-heartbeat",
            lease_expires=datetime.now(timezone.utc) - timedelta(seconds=60),
        ),
    )
    assert redis_store.list_ready_tasks("goal-lease-grace") == ["t1"]


def test_agent_active_slot_lifecycle(redis_store):
    """Agent active-slot lock should be single-owner and renewable."""
    ttl = timedelta(seconds=5)
    assert redis_store.acquire_agent_active_slot("agent-x", pid=101, ttl=ttl) is True
    assert redis_store.acquire_agent_active_slot("agent-x", pid=202, ttl=ttl) is False
    assert redis_store.renew_agent_active_slot("agent-x", pid=101, ttl=ttl) is True
    redis_store.release_agent_active_slot("agent-x", pid=101)
    assert redis_store.acquire_agent_active_slot("agent-x", pid=202, ttl=ttl) is True


def test_active_roles_only_include_recent_heartbeats(redis_store):
    """Only roles with fresh heartbeats should be considered active."""
    redis_store.register_agent("owner-agent", pid=1, capabilities={"role": "owner"})
    redis_store.register_agent("qa-agent", pid=2, capabilities={"role": "qa-dev"})
    redis_store.heartbeat_agent("qa-agent", ttl=timedelta(seconds=30))

    roles = redis_store.active_roles(stale_after=timedelta(seconds=5))
    assert "qa-dev" in roles
    assert "owner" not in roles
