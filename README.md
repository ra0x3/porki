# porki

Agentic orchestration.

`porki` is an orchestration runtime designed to run with
[systemg](https://github.com/ra0x3/systemg).

## Requirements

- Python 3.10+
- `systemg` available on your machine (`sysg` CLI), since `porki` uses it for supervised process spawning.

## Install

```bash
pip install porki
```

## CLI

```bash
porki --help
```

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
