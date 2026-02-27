# porki

Compiler-backed intent runtime with typed schema-v4 source, policy normalization, legalizer gates, deterministic execution, and auditable checkpoints.

## Requirements

- Python 3.12+
- Optional: `claude` or `codex` CLI for `porki run --prompt`

## Install

```bash
pip install -e .
```

## Commands

```bash
porki --help
porki run --help
porki instructions --help
```

### Prompt mode

```bash
porki run --prompt "Summarize this repo in 5 bullets"
```

### Create a v4 template

```bash
porki instructions create --name "Create a python calculator program" --path ./instructions
```

### Validate / compile / execute

```bash
porki instructions validate --path ./instructions/CREATE_A_PYTHON_CALCULATOR_PROGRAM.yaml
porki instructions compile --path ./instructions/CREATE_A_PYTHON_CALCULATOR_PROGRAM.yaml --out ./artifacts/compile.json
porki instructions execute --path ./instructions/CREATE_A_PYTHON_CALCULATOR_PROGRAM.yaml --run-id calc-1 --checkpoint ./artifacts/calc-1.json
```

### Unified run path (v4 source)

```bash
porki run --instructions ./instructions/CREATE_A_PYTHON_CALCULATOR_PROGRAM.yaml --run-id calc-1 --checkpoint ./artifacts/calc-1.json
```

## Local integration tests (non-CI)

These are intentionally skipped in CI.

```bash
RUN_LOCAL_INTEGRATION=1 pytest -q tests/test_integration_local.py
```

## Test

```bash
pytest -q
```
