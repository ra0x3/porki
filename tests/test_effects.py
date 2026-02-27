"""Tests for effect idempotency and lifecycle transitions."""

from __future__ import annotations

from porki.effects import EffectEngine, EffectState
from porki.intent import EffectTuple


class BackendWithIdempotency:
    supports_idempotency = True

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, effect: EffectTuple, payload: dict, idempotency_key: str) -> str:
        self.calls += 1
        return f"receipt:{idempotency_key}"


class BackendWithoutIdempotency:
    supports_idempotency = False

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, effect: EffectTuple, payload: dict, idempotency_key: str) -> str:
        self.calls += 1
        return f"receipt:{self.calls}"


def _effect() -> EffectTuple:
    return EffectTuple(
        primitive="net.send",
        data_class="business",
        recipient_class="trusted_partner",
        consent_class="explicit",
        legal_basis_class="contract",
        reversibility_class="irreversible",
    )


def test_effect_engine_generates_stable_idempotency_key():
    key_a = EffectEngine.build_idempotency_key("run", "task", _effect(), {"msg": "x"})
    key_b = EffectEngine.build_idempotency_key("run", "task", _effect(), {"msg": "x"})

    assert key_a == key_b


def test_effect_engine_records_and_dedupes_without_backend_idempotency():
    engine = EffectEngine()
    backend = BackendWithoutIdempotency()
    payload = {"msg": "hello"}

    first = engine.execute(backend, "run-1", "task-1", _effect(), payload)
    second = engine.execute(backend, "run-1", "task-1", _effect(), payload)

    assert backend.calls == 1
    assert first.idempotency_key == second.idempotency_key
    assert second.state == EffectState.RECORDED


def test_effect_engine_supports_compensation_transition():
    engine = EffectEngine()
    backend = BackendWithIdempotency()
    record = engine.execute(backend, "run-1", "task-1", _effect(), {"msg": "hello"})

    compensated = engine.compensate(record.idempotency_key)

    assert compensated.state == EffectState.COMPENSATED
