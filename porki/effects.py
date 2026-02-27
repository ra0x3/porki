"""Effect governance and idempotency lifecycle for runtime."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel

from .intent import EffectTuple


class EffectState(StrEnum):
    """Effect lifecycle states."""

    PLANNED = "planned"
    PREFLIGHT_PASSED = "preflight_passed"
    EXECUTED = "executed"
    RECORDED = "recorded"
    COMPENSATED = "compensated"


class EffectRecord(BaseModel):
    """Persisted effect execution state."""

    idempotency_key: str
    run_id: str
    task_id: str
    effect: EffectTuple
    state: EffectState
    backend_receipt: str | None = None


class EffectBackend(Protocol):
    """Backend interface for effect execution."""

    supports_idempotency: bool

    def execute(self, effect: EffectTuple, payload: dict[str, Any], idempotency_key: str) -> str:
        """Execute effect and return backend receipt."""


class EffectEngine:
    """Idempotency-first effect execution engine."""

    def __init__(self) -> None:
        self.records: dict[str, EffectRecord] = {}
        self.dedupe_receipts: dict[str, str] = {}

    @staticmethod
    def build_idempotency_key(
        run_id: str,
        task_id: str,
        effect: EffectTuple,
        payload: dict[str, Any],
    ) -> str:
        """Build deterministic idempotency key from effect tuple and payload hash."""
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        base = f"{run_id}:{task_id}:{effect.stable_key()}:{digest}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def preflight(
        self, run_id: str, task_id: str, effect: EffectTuple, payload: dict[str, Any]
    ) -> str:
        """Record preflight-passed effect state and return idempotency key."""
        key = self.build_idempotency_key(run_id, task_id, effect, payload)
        self.records[key] = EffectRecord(
            idempotency_key=key,
            run_id=run_id,
            task_id=task_id,
            effect=effect,
            state=EffectState.PREFLIGHT_PASSED,
        )
        return key

    def execute(
        self,
        backend: EffectBackend,
        run_id: str,
        task_id: str,
        effect: EffectTuple,
        payload: dict[str, Any],
    ) -> EffectRecord:
        """Execute one effect with exactly-once intent or dedupe fallback."""
        key = self.preflight(run_id, task_id, effect, payload)

        existing_receipt = self.dedupe_receipts.get(key)
        if existing_receipt is not None:
            record = self.records[key]
            record.state = EffectState.RECORDED
            record.backend_receipt = existing_receipt
            return record

        if backend.supports_idempotency:
            receipt = backend.execute(effect, payload, key)
        else:
            receipt = backend.execute(effect, payload, key)
            self.dedupe_receipts[key] = receipt

        record = self.records[key]
        record.state = EffectState.EXECUTED
        record.backend_receipt = receipt
        record.state = EffectState.RECORDED
        return record

    def compensate(self, idempotency_key: str) -> EffectRecord:
        """Mark an effect as compensated."""
        record = self.records[idempotency_key]
        record.state = EffectState.COMPENSATED
        return record


__all__ = [
    "EffectState",
    "EffectRecord",
    "EffectBackend",
    "EffectEngine",
]
