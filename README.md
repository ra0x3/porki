# porki

Compiler-backed intent runtime with typed schema source, policy normalization, legalizer gates, deterministic execution, and auditable checkpoints.

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
```

```
usage: porki [-h] [--version] {run,instructions} ...

Porki agent/orchestrator entrypoint (version 0.7.1)

positional arguments:
  {run,instructions}
    run               Run orchestrator/agent loops or submit a one-shot prompt
                      to the configured LLM
    instructions      Instruction file utilities

options:
  -h, --help          show this help message and exit
  --version           Show version and exit
```

### Run command

```bash
porki run --help
```

```
usage: porki run [-h] [--role {agent,orchestrator}]
                 [--instructions INSTRUCTIONS] [-p [PROMPT]]
                 [--redis-url REDIS_URL] [--log-level LOG_LEVEL]
                 [--log-style {concise,event}] [--color]
                 [--agent-name AGENT_NAME] [--agent-role AGENT_ROLE]
                 [--goal-id GOAL_ID] [--heartbeat HEARTBEAT]
                 [--loop-interval LOOP_INTERVAL] [--lease-ttl LEASE_TTL]
                 [--poll-interval POLL_INTERVAL]
                 [--heartbeat-interval HEARTBEAT_INTERVAL]
                 [--instruction-interval INSTRUCTION_INTERVAL]
                 [--idle-log-interval IDLE_LOG_INTERVAL]
                 [--claude-cli CLAUDE_CLI]
                 [--claude-extra-arg CLAUDE_EXTRA_ARG] [--claude-use-sysg]
                 [--llm-provider {claude,codex}] [--llm-cli LLM_CLI]
                 [--llm-extra-arg LLM_EXTRA_ARG] [--llm-use-sysg]

options:
  -h, --help            show this help message and exit
  --role {agent,orchestrator}
                        Process role
  --instructions INSTRUCTIONS
                        Primary instructions file
  -p [PROMPT], --prompt [PROMPT]
                        Simple one-shot prompt sent directly to the configured
                        LLM
  --redis-url REDIS_URL
                        Redis connection URL
  --log-level LOG_LEVEL
                        Python logging level
  --log-style {concise,event}
                        Terminal log style: concise (default) or event (full
                        context)
  --color               Enable colored logging output
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
  --idle-log-interval IDLE_LOG_INTERVAL
                        Agent idle-state summary log interval in seconds
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

### Instructions command

```bash
porki instructions --help
```

```
usage: porki instructions [-h] [--log-level LOG_LEVEL]
                          [--log-style {concise,event}] [--color]
                          {create} ...

positional arguments:
  {create}
    create              Create a structured instruction template

options:
  -h, --help            show this help message and exit
  --log-level LOG_LEVEL
                        Python logging level
  --log-style {concise,event}
                        Terminal log style: concise (default) or event (full
                        context)
  --color               Enable colored logging output
```

### Instructions create subcommand

```bash
porki instructions create --help
```

```
usage: porki instructions create [-h] -n NAME -p PATH [--force]
                                 [--log-level LOG_LEVEL]

options:
  -h, --help            show this help message and exit
  -n NAME, --name NAME  Instruction goal/name used to generate filename and
                        heading
  -p PATH, --path PATH  Directory where the instruction file should be created
  --force               Overwrite the target file if it already exists
  --log-level LOG_LEVEL
                        Python logging level
```

## Examples

### Prompt mode

```bash
porki run --prompt "Summarize this repo in 5 bullets"
```

### Create a template

```bash
porki instructions create --name "Create a python calculator program" --path ./instructions
```

### Validate / compile / execute

```bash
porki instructions validate --path ./instructions/CREATE_A_PYTHON_CALCULATOR_PROGRAM.yaml
porki instructions compile --path ./instructions/CREATE_A_PYTHON_CALCULATOR_PROGRAM.yaml --out ./artifacts/compile.json
porki instructions execute --path ./instructions/CREATE_A_PYTHON_CALCULATOR_PROGRAM.yaml --run-id calc-1 --checkpoint ./artifacts/calc-1.json
```

### Unified run path

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
