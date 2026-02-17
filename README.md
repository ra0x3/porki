# porki

<img src="https://i.imgur.com/VQ0Uk3g.png" width="250" height="250" />

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

Minimal runtime example (orchestrator + agent) with bundled test assets:

```bash
porki \
  --role orchestrator \
  --instructions tests/assets/INSTRUCTIONS.md \
  --llm-provider codex \
  --llm-cli codex
```

```bash
porki \
  --role agent \
  --instructions tests/assets/instructions/agent-research.md \
  --heartbeat tests/assets/instructions/heartbeat/agent-research.md \
  --agent-name agent-research \
  --goal-id goal-demo \
  --llm-provider codex \
  --llm-cli codex
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
usage: porki [-h] --role {agent,orchestrator} --instructions INSTRUCTIONS
             [--redis-url REDIS_URL] [--log-level LOG_LEVEL]
             [--agent-name AGENT_NAME] [--agent-role AGENT_ROLE]
             [--goal-id GOAL_ID] [--heartbeat HEARTBEAT]
             [--loop-interval LOOP_INTERVAL] [--lease-ttl LEASE_TTL]
             [--poll-interval POLL_INTERVAL]
             [--heartbeat-interval HEARTBEAT_INTERVAL]
             [--instruction-interval INSTRUCTION_INTERVAL]
             [--claude-cli CLAUDE_CLI] [--claude-extra-arg CLAUDE_EXTRA_ARG]
             [--claude-use-sysg] [--llm-provider {claude,codex}]
             [--llm-cli LLM_CLI] [--llm-extra-arg LLM_EXTRA_ARG]
             [--llm-use-sysg]

Porki agent/orchestrator entrypoint

options:
  -h, --help            show this help message and exit
  --role {agent,orchestrator}
                        Process role
  --instructions INSTRUCTIONS
                        Primary instructions file
  --redis-url REDIS_URL
                        Redis connection URL
  --log-level LOG_LEVEL
                        Python logging level
  --agent-name AGENT_NAME
                        Agent identifier when running in agent mode
  --agent-role AGENT_ROLE
                        Agent role identifier when running in agent mode
  --goal-id GOAL_ID     Goal identifier for the active DAG
  --heartbeat HEARTBEAT
                        Heartbeat file path for agent role
  --loop-interval LOOP_INTERVAL
                        Agent loop interval in seconds
  --lease-ttl LEASE_TTL
                        Lease TTL in seconds
  --poll-interval POLL_INTERVAL
                        Orchestrator poll interval in seconds
  --heartbeat-interval HEARTBEAT_INTERVAL
                        Agent heartbeat file read interval in seconds
  --instruction-interval INSTRUCTION_INTERVAL
                        Agent instructions reload interval in seconds
  --claude-cli CLAUDE_CLI
                        Path to the Claude CLI executable
  --claude-extra-arg CLAUDE_EXTRA_ARG
                        Additional arguments for the Claude CLI
  --claude-use-sysg     Invoke Claude through `sysg spawn --ttl` to capture
                        stdout/stderr
  --llm-provider {claude,codex}
                        LLM provider used for orchestration and agents
  --llm-cli LLM_CLI     Path to provider CLI executable
  --llm-extra-arg LLM_EXTRA_ARG
                        Additional arguments for the selected LLM CLI
  --llm-use-sysg        Invoke LLM CLI through `sysg spawn` for output capture
```
