"""Implements the single-process agent runtime."""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .cache import RedisStore
from .constants import AGENT_LOOP_INTERVAL, HEARTBEAT_REFRESH_INTERVAL, INSTRUCTION_REFRESH_INTERVAL
from .heartbeat import HeartbeatController, HeartbeatDirective
from .llm import LLMClient, RecoveryDecision, TaskExecutionResult
from .memory import Memory
from .models import DEFAULT_LEASE_TTL, TaskNode, TaskState, TaskStatus
from .version import BaseLogger, InstructionStore


class AgentRuntime(BaseLogger):
    """Coordinates local memory, heartbeat control, and Redis state."""

    MAX_RECOVERY_ATTEMPTS = 3
    MAX_SAME_SIGNATURE_RECOVERY_ATTEMPTS = 2
    MIN_RECOVERY_CONFIDENCE = 0.75
    _RECOVERABLE_ERROR_PATTERNS = (
        re.compile(r"spending cap reached", re.IGNORECASE),
        re.compile(r"rate limit", re.IGNORECASE),
        re.compile(r"timed out", re.IGNORECASE),
        re.compile(r"timeout", re.IGNORECASE),
        re.compile(r"temporar(?:y|ily)", re.IGNORECASE),
        re.compile(r"network", re.IGNORECASE),
        re.compile(r"econnreset|enotfound|eai_again", re.IGNORECASE),
        re.compile(r"command not found|not found", re.IGNORECASE),
        re.compile(r"unsupported engine|requires node|node\\s+version", re.IGNORECASE),
    )

    def __init__(
        self,
        *,
        agent_name: str,
        agent_role: str | None = None,
        goal_id: str,
        instructions_path: Path,
        heartbeat_path: Path,
        redis_store: RedisStore,
        llm_client: LLMClient | None = None,
        loop_interval: float = AGENT_LOOP_INTERVAL,
        lease_ttl: timedelta = DEFAULT_LEASE_TTL,
        heartbeat_refresh_interval: timedelta = HEARTBEAT_REFRESH_INTERVAL,
        instructions_refresh_interval: timedelta = INSTRUCTION_REFRESH_INTERVAL,
    ) -> None:
        """Initialize agent runtime and restore persisted memory."""
        super().__init__(f"{self.__class__.__name__}[{agent_name}]")
        self.agent_name = agent_name
        self.agent_role = agent_role or agent_name
        self.goal_id = goal_id
        self.instructions_path = instructions_path
        self.heartbeat_controller = HeartbeatController(heartbeat_path)
        self.redis_store = redis_store
        if llm_client is None:
            raise ValueError("AgentRuntime requires an LLM client instance")
        self.llm_client = llm_client
        self.loop_interval = loop_interval
        self.lease_ttl = lease_ttl
        self.heartbeat_refresh_interval = heartbeat_refresh_interval
        self.instructions_refresh_interval = instructions_refresh_interval

        self.memory = Memory()
        self.instructions_text = ""
        self.instruction_store = InstructionStore(redis_store.client)
        self.instruction_id: str | None = None
        self._active = True
        self._paused = False
        self._in_spending_cap_backoff = False
        self._last_reload = datetime.fromtimestamp(0, tz=timezone.utc)
        self._last_heartbeat_poll = datetime.fromtimestamp(0, tz=timezone.utc)
        self._pid = os.getpid()
        self._last_queue_snapshot: tuple[int, int, tuple[tuple[str, int], ...]] | None = None
        self._last_goal_state_signature: str | None = None

        self._hydrate_memory()
        self.llm_client.set_spending_cap_callback(self._on_spending_cap)
        self.llm_client.set_progress_callback(self._make_progress_callback())

    def _hydrate_memory(self) -> None:
        """Restore memory snapshot from Redis if present."""
        snapshot = self.redis_store.load_memory_snapshot(self.agent_name)
        if snapshot:
            self.logger.debug("Hydrating memory with %d entries", len(snapshot))
            self.memory.hydrate(snapshot)

    def reload_instructions(self) -> None:
        """Reload and version instructions from disk."""
        if self.instructions_path.exists():
            file_instructions = self.instructions_path.read_text(encoding="utf-8")

            if not self.instruction_id:
                self.instruction_id = f"{self.agent_name}:{self.goal_id}"

            latest_version = self.instruction_store.get_latest(self.instruction_id)
            if not latest_version or latest_version.instructions != file_instructions:
                self.instruction_store.push_version(file_instructions, self.instruction_id)
                self.logger.info("Pushed new instruction version for ID: %s", self.instruction_id)

            version = self.instruction_store.get_latest(self.instruction_id)
            if version:
                self.instructions_text = version.instructions
                self.memory.append(
                    f"Loaded instruction version {version.hash[:8]} at {datetime.now(timezone.utc).isoformat()}"
                )
                self.logger.info(
                    "Instructions loaded from store (ID: %s, hash: %s)",
                    self.instruction_id,
                    version.hash[:8],
                )
            else:
                self.instructions_text = file_instructions
                self.logger.warning("Failed to load from store, using file directly")
        else:
            self.instructions_text = ""
            self.logger.warning("Instructions file missing: %s", self.instructions_path)
        self._last_reload = datetime.now(timezone.utc)

    def poll_heartbeat(self) -> list[HeartbeatDirective]:
        """Read current heartbeat directives."""
        self.logger.info("Reading heartbeat at %s", self.heartbeat_controller.heartbeat_path)
        directives = self.heartbeat_controller.consume()
        return directives

    def apply_directives(self, directives: Iterable[HeartbeatDirective]) -> None:
        """Apply heartbeat directives to local runtime state."""
        for directive in directives:
            if directive.command == "PAUSE":
                self._paused = True
            elif directive.command == "RESUME":
                self._paused = False
            elif directive.command == "REPARSE":
                self.reload_instructions()
            elif directive.command == "DROP-TASK" and directive.args:
                self._drop_task(directive.args[0])
            elif directive.command == "ELEVATE" and len(directive.args) >= 2:
                self._elevate_task(directive.args[0], directive.args[1])
            elif directive.command == "FLUSH-MEMORY":
                self.memory.hydrate([])
                self.redis_store.store_memory_snapshot(self.agent_name, [])

    def _drop_task(self, task_id: str) -> None:
        """Requeue a task and release any held lock."""
        self.logger.info("Dropping task %s on operator request", task_id)
        state = self.redis_store.get_task_state(task_id)
        if not state:
            return
        self.redis_store.update_task_state(
            task_id, state.model_copy(update={"status": TaskStatus.READY, "owner": None})
        )
        self.redis_store.release_lock(task_id, self.agent_name)

    def _elevate_task(self, task_id: str, priority: str) -> None:
        """Adjust task priority in DAG metadata."""
        dag = self.redis_store.read_dag(self.goal_id)
        if not dag:
            return
        for node in dag.nodes:
            if node.id == task_id:
                try:
                    node.priority = int(priority)
                except ValueError:
                    self.logger.warning("Invalid priority %s", priority)
                break
        self.redis_store.write_dag(dag)

    def heartbeat(self) -> None:
        """Publish agent heartbeat to Redis."""
        self.redis_store.heartbeat_agent(self.agent_name, ttl=self.lease_ttl)
        self.redis_store.renew_agent_active_slot(self.agent_name, self._pid, self.lease_ttl)

    def _on_spending_cap(self, wait_seconds: float) -> None:
        """Publish goal-wide spending-cap backoff window."""
        if wait_seconds <= 0:
            return
        until = datetime.now(timezone.utc) + timedelta(seconds=wait_seconds)
        self.redis_store.set_goal_spending_cap_until(self.goal_id, until)
        self.log_event(
            logging.WARNING,
            "BACKOFF_PUBLISH",
            "[%s] Published goal-wide spending-cap backoff until %s",
            self.agent_name,
            until.isoformat(timespec="seconds"),
            goal=self.goal_id,
            role=self.agent_role,
            state="backoff",
            next_retry=until.isoformat(timespec="seconds"),
        )

    def run(self, *, max_cycles: int | None = None) -> None:
        """Run agent loop until stopped or cycle limit reached."""
        pid = self._pid
        self.log_event(
            logging.INFO,
            "AGENT_START",
            "[%s] Starting agent (PID=%d) for goal %s",
            self.agent_name,
            pid,
            self.goal_id,
            goal=self.goal_id,
            role=self.agent_role,
            state="active",
        )
        if not self.redis_store.acquire_agent_active_slot(self.agent_name, pid, self.lease_ttl):
            self.log_event(
                logging.ERROR,
                "AGENT_DUPLICATE",
                "[%s] Active runtime already exists; refusing duplicate agent start",
                self.agent_name,
                goal=self.goal_id,
                role=self.agent_role,
                state="error",
            )
            return
        self.redis_store.register_agent(
            self.agent_name, pid, capabilities={"role": self.agent_role}
        )
        self.reload_instructions()
        cycles = 0
        self.log_event(
            logging.INFO,
            "AGENT_LOOP_ENTER",
            "[%s] Agent initialized, entering main loop (max_cycles=%s)",
            self.agent_name,
            max_cycles if max_cycles else "unlimited",
            goal=self.goal_id,
            role=self.agent_role,
            state="active",
        )

        while self._active and (max_cycles is None or cycles < max_cycles):
            now = datetime.now(timezone.utc)
            if now - self._last_heartbeat_poll >= self.heartbeat_refresh_interval:
                directives = self.poll_heartbeat()
                self.apply_directives(directives)
                self._last_heartbeat_poll = now
            self.heartbeat()

            cap_until = self.redis_store.get_goal_spending_cap_until(self.goal_id)
            in_cap_backoff = bool(cap_until and cap_until > now)
            if in_cap_backoff and not self._in_spending_cap_backoff:
                wait_seconds = (cap_until - now).total_seconds() if cap_until else 0
                self.log_event(
                    logging.WARNING,
                    "BACKOFF_ACTIVE",
                    "[%s] Goal %s under spending-cap backoff for %.0fs (until %s); skipping work",
                    self.agent_name,
                    self.goal_id,
                    max(0, wait_seconds),
                    cap_until.isoformat(timespec="seconds") if cap_until else "unknown",
                    goal=self.goal_id,
                    role=self.agent_role,
                    state="backoff",
                    next_retry=cap_until.isoformat(timespec="seconds") if cap_until else "unknown",
                )
            elif not in_cap_backoff and self._in_spending_cap_backoff:
                self.log_event(
                    logging.INFO,
                    "BACKOFF_CLEARED",
                    "[%s] Goal %s spending-cap backoff cleared; resuming work",
                    self.agent_name,
                    self.goal_id,
                    goal=self.goal_id,
                    role=self.agent_role,
                    state="active",
                )
            self._in_spending_cap_backoff = in_cap_backoff

            if not self._paused and not in_cap_backoff:
                if now - self._last_reload >= self.instructions_refresh_interval:
                    self.reload_instructions()
                self._run_cycle()
            else:
                if self._paused:
                    self.logger.debug(
                        "[%s] Agent paused, skipping cycle %d", self.agent_name, cycles
                    )
            cycles += 1
            if self.loop_interval > 0:
                time.sleep(self.loop_interval)

        self.log_event(
            logging.INFO,
            "AGENT_STOP",
            "[%s] Agent shutting down after %d cycles",
            self.agent_name,
            cycles,
            goal=self.goal_id,
            role=self.agent_role,
            state="stopped",
        )
        memory_snapshot = self.memory.snapshot()
        self.logger.info(
            "[%s] Storing final memory snapshot (%d entries)", self.agent_name, len(memory_snapshot)
        )
        self.redis_store.store_memory_snapshot(self.agent_name, memory_snapshot)
        self.redis_store.deregister_agent(self.agent_name)
        self.redis_store.release_agent_active_slot(self.agent_name, pid)
        self.logger.info("[%s] Agent shutdown complete", self.agent_name)

    def stop(self) -> None:
        """Request runtime shutdown."""
        self._active = False

    def _run_cycle(self) -> None:
        """Select, claim, execute, and summarize one task."""
        ready_task_ids = self.redis_store.list_ready_tasks(self.goal_id)
        ready_role_counts = self._ready_role_counts(ready_task_ids)
        if not ready_task_ids:
            self._emit_goal_state(ready_task_ids, ready_role_counts, eligible_count=0)
            if self._last_queue_snapshot != (0, 0, ()):
                self.log_event(
                    logging.INFO,
                    "QUEUE_EMPTY",
                    "[%s] No ready tasks for goal %s",
                    self.agent_name,
                    self.goal_id,
                    goal=self.goal_id,
                    role=self.agent_role,
                    state="idle",
                )
            else:
                self.logger.debug("[%s] No ready tasks for goal %s", self.agent_name, self.goal_id)
            self._last_queue_snapshot = (0, 0, ())
            return

        ready_nodes = []
        for task_id in ready_task_ids:
            node = self.redis_store.get_task_node(self.goal_id, task_id)
            if node and self._is_node_eligible(node):
                ready_nodes.append(node)
                self.logger.debug(
                    "[%s]   Ready task: %s (priority=%d)", self.agent_name, node.id, node.priority
                )

        queue_snapshot = (
            len(ready_task_ids),
            len(ready_nodes),
            tuple(sorted(ready_role_counts.items())),
        )
        snapshot_changed = self._last_queue_snapshot != queue_snapshot
        if snapshot_changed:
            self.log_event(
                logging.INFO,
                "QUEUE_SNAPSHOT",
                "[%s] Found %d ready tasks for goal %s",
                self.agent_name,
                len(ready_task_ids),
                self.goal_id,
                goal=self.goal_id,
                role=self.agent_role,
                state="active",
            )
        else:
            self.logger.debug(
                "[%s] Found %d ready tasks for goal %s",
                self.agent_name,
                len(ready_task_ids),
                self.goal_id,
            )
        self._last_queue_snapshot = queue_snapshot
        self._emit_goal_state(ready_task_ids, ready_role_counts, eligible_count=len(ready_nodes))

        if not ready_nodes:
            if snapshot_changed:
                self.log_event(
                    logging.INFO,
                    "QUEUE_NO_ELIGIBLE",
                    "[%s] No eligible ready tasks for role %s (global_ready=%d) on goal %s",
                    self.agent_name,
                    self.agent_role,
                    len(ready_task_ids),
                    self.goal_id,
                    goal=self.goal_id,
                    role=self.agent_role,
                    state="idle",
                )
            else:
                self.logger.debug(
                    "[%s] No eligible ready tasks for role %s (global_ready=%d) on goal %s",
                    self.agent_name,
                    self.agent_role,
                    len(ready_task_ids),
                    self.goal_id,
                )
            return

        self.log_event(
            logging.INFO,
            "TASK_SELECT_REQUEST",
            "[%s] Asking LLM to select from %d tasks",
            self.agent_name,
            len(ready_nodes),
            goal=self.goal_id,
            role=self.agent_role,
            state="active",
        )
        self.llm_client.set_progress_callback(self._make_progress_callback())
        selection = self.llm_client.select_next_task(
            ready_nodes,
            memory=self.memory,
            goal_id=self.goal_id,
            instructions=self.instructions_text,
        )
        if not selection.selected_task_id:
            self.log_event(
                logging.INFO,
                "TASK_SELECT_NONE",
                "[%s] LLM declined to select a task: %s",
                self.agent_name,
                selection.justification,
                goal=self.goal_id,
                role=self.agent_role,
                state="idle",
            )
            return

        self.log_event(
            logging.INFO,
            "TASK_SELECTED",
            "[%s] LLM selected task %s (confidence=%.2f): %s",
            self.agent_name,
            selection.selected_task_id,
            selection.confidence,
            selection.justification,
            goal=self.goal_id,
            role=self.agent_role,
            task=selection.selected_task_id,
            state="active",
        )

        node = self.redis_store.get_task_node(self.goal_id, selection.selected_task_id)
        if not node:
            self.logger.warning(
                "[%s] Selected task %s missing from DAG",
                self.agent_name,
                selection.selected_task_id,
            )
            return
        if not self._is_node_eligible(node):
            return

        if not self.redis_store.acquire_lock(node.id, self.agent_name, self.lease_ttl):
            self.log_event(
                logging.INFO,
                "TASK_LOCK_MISS",
                "[%s] Could not acquire lock for %s (another agent has it)",
                self.agent_name,
                node.id,
                goal=self.goal_id,
                role=self.agent_role,
                task=node.id,
                state="active",
            )
            return
        try:
            self.log_event(
                logging.INFO,
                "TASK_LOCK_ACQUIRED",
                "[%s] Acquired lock for task %s: %s",
                self.agent_name,
                node.id,
                node.title,
                goal=self.goal_id,
                role=self.agent_role,
                task=node.id,
                state="running",
            )
            guard_token = f"{self.agent_name}:{uuid.uuid4().hex}"
            if not self.redis_store.acquire_execution_guard(
                node.id, guard_token, self._execution_guard_ttl()
            ):
                self.logger.warning(
                    "[%s] Execution guard already active for %s; skipping duplicate run",
                    self.agent_name,
                    node.id,
                )
                return
            state = self.redis_store.get_task_state(node.id) or TaskState(status=TaskStatus.READY)
            lease_expires = datetime.now(timezone.utc) + self.lease_ttl
            state = state.as_running(owner=self.agent_name, lease_expires=lease_expires)
            self.redis_store.update_task_state(node.id, state)
            self.llm_client.set_progress_callback(
                self._make_progress_callback(task_id=node.id, guard_token=guard_token)
            )

            self.log_event(
                logging.INFO,
                "TASK_EXECUTE_START",
                "[%s] Executing task %s with LLM",
                self.agent_name,
                node.id,
                goal=self.goal_id,
                role=self.agent_role,
                task=node.id,
                state="running",
            )
            execution = self.llm_client.execute_task(
                node,
                goal_id=self.goal_id,
                instructions=self.instructions_text,
                memory=self.memory,
            )
            self.log_event(
                logging.INFO,
                "TASK_EXECUTE_RESULT",
                "[%s] Task %s execution result: status=%s, outputs=%s, notes=%s",
                self.agent_name,
                node.id,
                execution.status,
                execution.outputs,
                execution.notes[:200] if execution.notes else "None",
                goal=self.goal_id,
                role=self.agent_role,
                task=node.id,
                state=execution.status.strip().lower() or "running",
            )

            if execution.follow_ups:
                self.logger.info(
                    "[%s] Task %s suggested follow-ups: %s",
                    self.agent_name,
                    node.id,
                    execution.follow_ups,
                )

            self.log_event(
                logging.INFO,
                "TASK_SUMMARY_START",
                "[%s] Generating summary for task %s",
                self.agent_name,
                node.id,
                goal=self.goal_id,
                role=self.agent_role,
                task=node.id,
                state="running",
            )
            summary = self.llm_client.summarize_task(
                node,
                execution,
                goal_id=self.goal_id,
                instructions=self.instructions_text,
                memory=self.memory,
            )
            self.log_event(
                logging.INFO,
                "TASK_SUMMARY_DONE",
                "[%s] Task %s summary: %s",
                self.agent_name,
                node.id,
                summary[:200],
                goal=self.goal_id,
                role=self.agent_role,
                task=node.id,
                state="running",
            )
            self._record_result(node, execution, summary)
        except Exception as exc:
            self.log_event(
                logging.ERROR,
                "TASK_FAILED",
                "[%s] Task %s failed: %s",
                self.agent_name,
                node.id,
                exc,
                goal=self.goal_id,
                role=self.agent_role,
                task=node.id,
                state="error",
            )
            self.logger.exception("Full exception details:")
            self._record_failure_or_recovery(node, str(exc))
        finally:
            self.llm_client.set_progress_callback(self._make_progress_callback())
            self.redis_store.release_execution_guard(node.id, guard_token)
            self.log_event(
                logging.INFO,
                "TASK_LOCK_RELEASED",
                "[%s] Releasing lock for task %s",
                self.agent_name,
                node.id,
                goal=self.goal_id,
                role=self.agent_role,
                task=node.id,
                state="active",
            )
            self.redis_store.release_lock(node.id, self.agent_name)

    def _record_result(self, node: TaskNode, execution: TaskExecutionResult, summary: str) -> None:
        """Apply role-aware task transitions based on execution output."""
        execution_status = execution.status.strip().lower()
        phase = node.metadata.get("phase", "development")
        if phase == "qa" and execution_status in {"failed", "qa_failed"}:
            self._record_qa_failure(node, summary)
            return
        if execution_status in {"failed", "blocked"}:
            self._record_failure_or_recovery(node, execution.notes or execution_status)
            return
        if phase == "development":
            state = TaskState(
                status=TaskStatus.DEV_DONE, progress=summary, artifacts=execution.outputs
            )
        elif phase == "qa":
            state = TaskState(
                status=TaskStatus.QA_PASSED, progress=summary, artifacts=execution.outputs
            )
        elif phase == "integration":
            state = TaskState(status=TaskStatus.DONE, progress=summary, artifacts=execution.outputs)
        else:
            state = TaskState(status=TaskStatus.DONE, progress=summary, artifacts=execution.outputs)
        self.redis_store.update_task_state(node.id, state)
        self.memory.append(f"Completed {node.id}: {summary}")
        self.redis_store.store_memory_snapshot(self.agent_name, self.memory.snapshot())

    def _record_failure(self, node: TaskNode, error: str) -> None:
        """Persist failure state and update memory snapshot."""
        state = TaskState(status=TaskStatus.FAILED, last_error=error)
        self.redis_store.update_task_state(node.id, state)
        self.memory.append(f"Failed {node.id}: {error}")
        self.redis_store.store_memory_snapshot(self.agent_name, self.memory.snapshot())

    def _record_failure_or_recovery(self, node: TaskNode, error: str) -> None:
        """Create remediation task for recoverable errors; otherwise fail."""
        root_node = self._resolve_recovery_root(node)
        attempts = max(
            int(root_node.metadata.get("recovery_attempts", "0")),
            int(root_node.metadata.get("recovery_attempt", "0")),
        )
        signature = self._error_signature(error)
        previous_signature = root_node.metadata.get("last_recovery_signature", "")
        same_signature_attempts = (
            int(root_node.metadata.get("same_recovery_signature_attempts", "0"))
            if signature == previous_signature
            else 0
        )
        same_signature_attempts += 1
        if same_signature_attempts > self.MAX_SAME_SIGNATURE_RECOVERY_ATTEMPTS:
            self.logger.error(
                "[%s] Recovery halted for %s due to repeated signature (%s)",
                self.agent_name,
                root_node.id,
                signature,
            )
            self._record_failure(root_node, error)
            if node.id != root_node.id:
                self._record_failure(node, error)
            return
        if attempts >= self.MAX_RECOVERY_ATTEMPTS:
            self.logger.error(
                "[%s] Recovery exhausted for task %s after %d attempts",
                self.agent_name,
                root_node.id,
                attempts,
            )
            self._record_failure(root_node, error)
            if node.id != root_node.id:
                self._record_failure(node, error)
            return

        decision = self._build_recovery_decision(node, error)
        if not decision.recoverable:
            self._record_failure(root_node, error)
            if node.id != root_node.id:
                self._record_failure(node, error)
            return

        attempt = attempts + 1
        root_node.metadata["recovery_attempts"] = str(attempt)
        root_node.metadata["last_recovery_signature"] = signature
        root_node.metadata["same_recovery_signature_attempts"] = str(same_signature_attempts)
        if decision.reason:
            root_node.metadata["last_recovery_reason"] = decision.reason[:300]
        self.redis_store.update_task_node(self.goal_id, root_node)

        remediation_title = decision.remediation_title.strip()
        if not remediation_title:
            remediation_title = f"Recover blocked task {root_node.id}"
        recovery_id = self.redis_store.create_recovery_task(
            goal_id=self.goal_id,
            blocked_task_id=root_node.id,
            recovery_root_id=root_node.id,
            owner_role=root_node.metadata.get("required_role", self.agent_role),
            recovery_attempt=attempt,
            priority=max(0, root_node.priority + 2),
            title=remediation_title,
        )
        progress = (
            f"Recoverable failure on {root_node.id}; created remediation task {recovery_id} "
            f"(attempt {attempt}/{self.MAX_RECOVERY_ATTEMPTS})."
        )
        self.redis_store.update_task_state(
            root_node.id,
            TaskState(
                status=TaskStatus.BLOCKED,
                progress=progress,
                last_error=error[:600],
            ),
        )
        if node.id != root_node.id:
            self.redis_store.update_task_state(
                node.id,
                TaskState(
                    status=TaskStatus.FAILED,
                    progress=f"Superseded by {recovery_id}",
                    last_error=error[:600],
                ),
            )
        self.memory.append(progress)
        self.redis_store.store_memory_snapshot(self.agent_name, self.memory.snapshot())
        self.logger.warning(
            "[%s] Recoverable failure for %s. remediation=%s attempt=%d/%d reason=%s",
            self.agent_name,
            root_node.id,
            recovery_id,
            attempt,
            self.MAX_RECOVERY_ATTEMPTS,
            decision.reason or "unspecified",
        )

    def _build_recovery_decision(self, node: TaskNode, error: str) -> RecoveryDecision:
        """Classify whether a task failure should produce remediation work."""
        deterministic = self._deterministic_recovery_decision(error)
        if deterministic:
            self.logger.info(
                "[%s] Deterministic recovery classification for %s: %s",
                self.agent_name,
                node.id,
                deterministic.reason,
            )
            return deterministic

        try:
            llm_decision = self.llm_client.assess_recovery(
                node,
                error=error,
                goal_id=self.goal_id,
                instructions=self.instructions_text,
                memory=self.memory,
            )
        except Exception as exc:
            self.logger.warning(
                "[%s] Recovery classification failed for %s: %s",
                self.agent_name,
                node.id,
                exc,
            )
            return RecoveryDecision(
                recoverable=False,
                reason="Recovery classification unavailable",
                remediation_title="",
                remediation_steps=[],
                confidence=0.0,
            )

        recoverable = llm_decision.recoverable and (
            llm_decision.confidence >= self.MIN_RECOVERY_CONFIDENCE
        )
        reason = llm_decision.reason or (
            f"LLM classified recoverable={llm_decision.recoverable} "
            f"confidence={llm_decision.confidence:.2f}"
        )
        if not recoverable:
            reason = (
                f"{reason}; confidence {llm_decision.confidence:.2f} "
                f"below threshold {self.MIN_RECOVERY_CONFIDENCE:.2f}"
            )
        return RecoveryDecision(
            recoverable=recoverable,
            reason=reason,
            remediation_title=llm_decision.remediation_title,
            remediation_steps=llm_decision.remediation_steps,
            confidence=llm_decision.confidence,
        )

    def _deterministic_recovery_decision(self, error: str) -> RecoveryDecision | None:
        """Return recovery decision for known transient/environment error patterns."""
        for pattern in self._RECOVERABLE_ERROR_PATTERNS:
            if pattern.search(error):
                return RecoveryDecision(
                    recoverable=True,
                    reason=f"Matched recoverable error pattern: {pattern.pattern}",
                    remediation_title="Repair environment/runtime prerequisites",
                    remediation_steps=[],
                    confidence=1.0,
                )
        return None

    def _record_qa_failure(self, node: TaskNode, summary: str) -> None:
        """Create remediation work and block QA until fixes are delivered."""
        cycle = int(node.metadata.get("review_cycle", "0")) + 1
        dev_role = node.metadata.get("dev_role") or self.agent_role
        manager_role = node.metadata.get("manager_role")
        if cycle > 3:
            if manager_role:
                dev_role = manager_role
        node.metadata["review_cycle"] = str(cycle)
        self.redis_store.update_task_node(self.goal_id, node)
        self.redis_store.create_remediation_task(
            goal_id=self.goal_id,
            qa_task_id=node.id,
            dev_role=dev_role,
            review_cycle=cycle,
            priority=max(0, node.priority + 1),
        )
        self.redis_store.update_task_state(
            node.id,
            TaskState(status=TaskStatus.BLOCKED, progress=summary, artifacts=[]),
        )
        self.memory.append(f"QA failed {node.id}: cycle {cycle}")
        self.redis_store.store_memory_snapshot(self.agent_name, self.memory.snapshot())

    def _execution_guard_ttl(self) -> timedelta:
        """Return TTL for per-task in-flight execution guards."""
        return timedelta(seconds=max(300, int(self.lease_ttl.total_seconds() * 4)))

    def _make_progress_callback(
        self,
        *,
        task_id: str | None = None,
        guard_token: str | None = None,
    ):
        """Build periodic callback for long-running LLM requests."""

        def _callback() -> None:
            self.heartbeat()
            if not task_id:
                return
            if self.redis_store.renew_lock(task_id, self.agent_name, self.lease_ttl):
                state = self.redis_store.get_task_state(task_id)
                if (
                    state
                    and state.status in {TaskStatus.RUNNING, TaskStatus.CLAIMED}
                    and state.owner == self.agent_name
                ):
                    self.redis_store.update_task_state(
                        task_id,
                        state.model_copy(
                            update={"lease_expires": datetime.now(timezone.utc) + self.lease_ttl}
                        ),
                    )
            if guard_token:
                self.redis_store.renew_execution_guard(
                    task_id, guard_token, self._execution_guard_ttl()
                )

        return _callback

    def _resolve_recovery_root(self, node: TaskNode) -> TaskNode:
        """Resolve the canonical task node that owns recovery attempts."""
        root_id = node.metadata.get("recovery_for") or node.id
        root_node = self.redis_store.get_task_node(self.goal_id, root_id)
        return root_node or node

    @staticmethod
    def _error_signature(error: str) -> str:
        """Normalize an error into a compact signature for loop detection."""
        normalized = re.sub(r"\s+", " ", error.lower()).strip()
        normalized = re.sub(r"\d+", "#", normalized)
        return normalized[:180]

    def _is_node_eligible(self, node: TaskNode) -> bool:
        """Return whether this agent may claim and run the node now."""
        required_role = node.metadata.get("required_role")
        if required_role and required_role != self.agent_role:
            return False
        state = self.redis_store.get_task_state(node.id)
        if not state:
            return False
        if state.status in {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.INTEGRATED}:
            return False
        return True

    def _ready_role_counts(self, ready_task_ids: list[str]) -> dict[str, int]:
        """Return role histogram for currently ready tasks."""
        counts: dict[str, int] = {}
        for task_id in ready_task_ids:
            node = self.redis_store.get_task_node(self.goal_id, task_id)
            if not node:
                continue
            role = node.metadata.get("required_role", "unassigned")
            counts[role] = counts.get(role, 0) + 1
        return counts

    def _emit_goal_state(
        self,
        ready_task_ids: list[str],
        ready_role_counts: dict[str, int],
        *,
        eligible_count: int,
    ) -> None:
        """Emit one-shot goal completion/blocked summaries when state changes."""
        counts = self.redis_store.goal_status_counts(self.goal_id)
        ready_count = len(ready_task_ids)
        running_count = counts.get(TaskStatus.RUNNING, 0) + counts.get(TaskStatus.CLAIMED, 0)
        blocked_count = counts.get(TaskStatus.BLOCKED, 0)
        done_count = counts.get(TaskStatus.DONE, 0)
        failed_count = counts.get(TaskStatus.FAILED, 0)

        if ready_count == 0 and running_count == 0 and blocked_count == 0:
            signature = f"complete:{done_count}:{failed_count}"
            if signature != self._last_goal_state_signature:
                self.log_event(
                    logging.INFO,
                    "GOAL_COMPLETE",
                    "[%s] Goal %s complete: 0 tasks left (done=%d, failed=%d)",
                    self.agent_name,
                    self.goal_id,
                    done_count,
                    failed_count,
                    goal=self.goal_id,
                    role=self.agent_role,
                    state="complete",
                )
            self._last_goal_state_signature = signature
            return

        if ready_count > 0 and eligible_count == 0:
            active_roles = self.redis_store.active_roles()
            required_roles = set(ready_role_counts)
            inactive_required_roles = sorted(required_roles - active_roles)
            if inactive_required_roles:
                signature = f"blocked-inactive:{ready_count}:{','.join(inactive_required_roles)}"
                if signature != self._last_goal_state_signature:
                    self.log_event(
                        logging.WARNING,
                        "GOAL_BLOCKED_INACTIVE_ROLE",
                        "[%s] Goal %s blocked: %d ready tasks require inactive roles %s",
                        self.agent_name,
                        self.goal_id,
                        ready_count,
                        ", ".join(inactive_required_roles),
                        goal=self.goal_id,
                        role=self.agent_role,
                        state="blocked",
                    )
                self._last_goal_state_signature = signature
                return

        self._last_goal_state_signature = (
            f"active:{ready_count}:{eligible_count}:{blocked_count}:{running_count}"
        )
