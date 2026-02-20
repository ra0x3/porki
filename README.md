# porki

<img src="https://i.imgur.com/A9d5wWF.png" width="300" height="300" />

<div display="flex" align-items="center">
    <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" />
    <img src="https://img.shields.io/badge/Claude-D97757?style=for-the-badge&logo=anthropic&logoColor=white" />
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=apple&logoColor=white" />
    <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black" />
</div>

## Table of Contents

1. [Description](#description)
2. [Quickstart](#quickstart)
   - 2.1 [Install](#install)
   - 2.2 [Simple Command](#simple-command)
3. [Installation](#installation)
   - 3.1 [Prerequisites](#prerequisites)
   - 3.2 [PyPI](#pypi)
   - 3.3 [From Source](#from-source)
   - 3.4 [Development Only (uv)](#development-only-uv)
4. [Test](#test)
5. [CLI Reference](#cli-reference)

---

## Description

`porki` is an agentic orchestration runtime for multi-agent workflows. It coordinates orchestrator and agent processes, persists shared state in Redis, and uses an LLM CLI (`claude` or `codex`) to plan and execute tasks.

It leverages markdown-based control files: a primary `INSTRUCTIONS.md` for orchestrator/agent directives and per-agent heartbeat markdown files for live control signals (for example pause/resume-style runtime directives).

Example usage: [Systemg Orchestrator](https://github.com/ra0x3/systemg/tree/main/examples/orchestrator)

> [!WARNING]
> Designed to run with [systemg](https://github.com/ra0x3/systemg).

## Quickstart

### Install

```bash
pip install porki
```

### Simple Command

```bash
porki --help
```

```bash
porki run --help
```

```bash
porki instructions --help
```

Minimal runtime example (orchestrator + agent) with bundled test assets:

```bash
porki run \
  --role orchestrator \
  --instructions tests/assets/INSTRUCTIONS.md \
  --llm-provider codex \
  --llm-cli codex
```

```bash
porki run \
  --role agent \
  --instructions tests/assets/instructions/agent-research.md \
  --heartbeat tests/assets/instructions/heartbeat/agent-research.md \
  --agent-name agent-research \
  --goal-id goal-demo \
  --llm-provider codex \
  --llm-cli codex
```

One-shot prompt mode:

```bash
porki run --prompt "Draft a concise architecture summary for this repo."
```

Enable colored logging output (similar to cargo):

```bash
porki run --role orchestrator --instructions INSTRUCTIONS.md --color
```

The `--color` flag enables ANSI color codes for log levels:
- **INFO**: Green
- **DEBUG**: Light Blue (Cyan)
- **WARNING**: Yellow
- **ERROR**: Red

Create a template instruction file:

```bash
porki instructions create --name "Core infra dev" --path ./instructions
```

Generated templates now include canonical JSON schema examples for goals, DAG tasks, task state, finished tasks, and LLM response payloads.
Each generated file also includes explicit version metadata (`porki_instruction_template_version` and `porki_schema_version`) so upgrades are trackable.

Example output path from the command above:

```bash
./instructions/CORE_INFRA_DEV.md
```

By default, `--redis-url` is `fakeredis://` for local/demo usage.

## Installation

### Prerequisites

- `python` 3.10+
- `systemg` (`sysg` CLI available on PATH)
- `redis` (server reachable by `--redis-url`)
- an LLM CLI: `claude` or `codex`

### PyPI

```bash
pip install porki
```

### From Source

```bash
git clone https://github.com/ra0x3/porki.git
cd porki
pip install -e .
```

### Development Only (uv)

`uv` commands are for development workflows (not required for normal runtime use):

```bash
uv sync
```

## Test

From repository checkout:

```bash
uv run pytest
```

Without `uv`:

```bash
python -m pip install pytest
python -m pytest
```

## CLI Reference

```text
usage: porki [-h] [--version] {run,instructions} ...

Porki agent/orchestrator entrypoint (version X.Y.Z)

positional arguments:
  {run,instructions}
    run               Run orchestrator/agent loops or submit a one-shot prompt
                      to the configured LLM
    instructions      Instruction file utilities

options:
  -h, --help          show this help message and exit
  --version           Show version and exit
```

```text
usage: porki run [-h] [--role {agent,orchestrator}]
                 [--instructions INSTRUCTIONS] [-p [PROMPT]]
                 [--redis-url REDIS_URL] [--log-level LOG_LEVEL]
                 [--color] [--agent-name AGENT_NAME] [--agent-role AGENT_ROLE]
                 [--goal-id GOAL_ID] [--heartbeat HEARTBEAT]
                 [--loop-interval LOOP_INTERVAL] [--lease-ttl LEASE_TTL]
                 [--poll-interval POLL_INTERVAL]
                 [--heartbeat-interval HEARTBEAT_INTERVAL]
                 [--instruction-interval INSTRUCTION_INTERVAL]
                 [--claude-cli CLAUDE_CLI]
                 [--claude-extra-arg CLAUDE_EXTRA_ARG] [--claude-use-sysg]
                 [--llm-provider {claude,codex}] [--llm-cli LLM_CLI]
                 [--llm-extra-arg LLM_EXTRA_ARG] [--llm-use-sysg]
```

```text
usage: porki instructions [-h] [--log-level LOG_LEVEL] [--color] {create} ...
```

```text
usage: porki instructions create [-h] -n NAME -p PATH [--force]
                                 [--log-level LOG_LEVEL]
```
