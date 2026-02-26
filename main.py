"""CLI entrypoint for porki orchestrator/agent roles."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path

import redis

try:
    import fakeredis
except ImportError:
    fakeredis = None

from porki.cache import RedisStore
from porki.instructions import InstructionParser
from porki.instruction_schemas import (
    INSTRUCTION_SCHEMA_VERSION,
    INSTRUCTION_TEMPLATE_VERSION,
    render_instruction_schema_reference,
)
from porki.llm import LLMRuntimeConfig, create_llm_client
from porki.logging_utils import CompactingHandler, EventContextFilter, EventFormatter
from porki.orchestrator import Orchestrator, RealSpawnAdapter
from porki.runtime import AgentRuntime

LOGGER = logging.getLogger(__name__)


def _get_version() -> str:
    """Get the current version of porki."""
    try:
        return metadata.version("porki")
    except metadata.PackageNotFoundError:
        return "unknown"


def _add_llm_flags(parser: argparse.ArgumentParser) -> None:
    """Add provider selection and CLI invocation flags."""
    parser.add_argument("--claude-cli", default="claude", help="Path to the Claude CLI executable")
    parser.add_argument(
        "--claude-extra-arg",
        action="append",
        default=[],
        help="Additional arguments for the Claude CLI",
    )
    parser.add_argument(
        "--claude-use-sysg",
        action="store_true",
        help="Invoke Claude through `sysg spawn --ttl` to capture stdout/stderr",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["claude", "codex"],
        help="LLM provider used for orchestration and agents",
    )
    parser.add_argument("--llm-cli", help="Path to provider CLI executable")
    parser.add_argument(
        "--llm-extra-arg",
        action="append",
        default=[],
        help="Additional arguments for the selected LLM CLI",
    )
    parser.add_argument(
        "--llm-use-sysg",
        action="store_true",
        help="Invoke LLM CLI through `sysg spawn` for output capture",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    version = _get_version()
    parser = argparse.ArgumentParser(
        prog="porki", description=f"Porki agent/orchestrator entrypoint (version {version})"
    )
    parser.add_argument(
        "--version", action="version", version=f"porki {version}", help="Show version and exit"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run orchestrator/agent loops or submit a one-shot prompt to the configured LLM",
    )
    run_parser.add_argument("--role", choices=["agent", "orchestrator"], help="Process role")
    run_parser.add_argument("--instructions", type=Path, help="Primary instructions file")
    run_parser.add_argument(
        "-p",
        "--prompt",
        nargs="?",
        const="",
        help="Simple one-shot prompt sent directly to the configured LLM",
    )
    run_parser.add_argument("--redis-url", default="fakeredis://", help="Redis connection URL")
    run_parser.add_argument("--log-level", default="INFO", help="Python logging level")
    run_parser.add_argument(
        "--log-style",
        choices=["concise", "event"],
        default="concise",
        help="Terminal log style: concise (default) or event (full context)",
    )
    run_parser.add_argument("--color", action="store_true", help="Enable colored logging output")
    run_parser.add_argument("--agent-name", help="Agent identifier when running in agent mode")
    run_parser.add_argument("--agent-role", help="Agent role identifier when running in agent mode")
    run_parser.add_argument("--goal-id", help="Goal identifier for the active DAG")
    run_parser.add_argument("--heartbeat", type=Path, help="Heartbeat file path for agent role")
    run_parser.add_argument(
        "--loop-interval", type=float, default=1.0, help="Agent loop interval in seconds"
    )
    run_parser.add_argument("--lease-ttl", type=float, default=30.0, help="Lease TTL in seconds")
    run_parser.add_argument(
        "--poll-interval", type=float, default=5.0, help="Orchestrator poll interval in seconds"
    )
    run_parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=300.0,
        help="Agent heartbeat file read interval in seconds",
    )
    run_parser.add_argument(
        "--instruction-interval",
        type=float,
        default=300.0,
        help="Agent instructions reload interval in seconds",
    )
    run_parser.add_argument(
        "--idle-log-interval",
        type=float,
        default=60.0,
        help="Agent idle-state summary log interval in seconds",
    )
    _add_llm_flags(run_parser)

    instructions_parser = subparsers.add_parser("instructions", help="Instruction file utilities")
    instructions_parser.add_argument("--log-level", default="INFO", help="Python logging level")
    instructions_parser.add_argument(
        "--log-style",
        choices=["concise", "event"],
        default="concise",
        help="Terminal log style: concise (default) or event (full context)",
    )
    instructions_parser.add_argument(
        "--color", action="store_true", help="Enable colored logging output"
    )
    instructions_subparsers = instructions_parser.add_subparsers(
        dest="instructions_command", required=True
    )
    create_parser = instructions_subparsers.add_parser(
        "create",
        help="Create a structured instruction template",
    )
    create_parser.add_argument(
        "-n",
        "--name",
        required=True,
        help="Instruction goal/name used to generate filename and heading",
    )
    create_parser.add_argument(
        "-p",
        "--path",
        type=Path,
        required=True,
        help="Directory where the instruction file should be created",
    )
    create_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target file if it already exists",
    )
    create_parser.add_argument("--log-level", default="INFO", help="Python logging level")

    validate_parser = instructions_subparsers.add_parser(
        "validate",
        help="Validate a strict YAML orchestrator instruction file",
    )
    validate_parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Instruction YAML file to validate",
    )
    validate_parser.add_argument("--log-level", default="INFO", help="Python logging level")

    return parser


def _redis_client_from_url(url: str):
    """Construct Redis client from connection URL."""
    if url.startswith("fakeredis://"):
        if not fakeredis:
            raise RuntimeError("fakeredis is not installed")
        return fakeredis.FakeRedis(decode_responses=False)
    return redis.Redis.from_url(url)


def _configure_logging(level: str, use_color: bool = False, log_style: str = "concise") -> None:
    """Configure root logging format and level."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    stream = logging.StreamHandler()
    if use_color and log_style == "event":
        from porki.logging_utils import ColoredEventFormatter

        formatter = ColoredEventFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    elif use_color and log_style == "concise":
        from porki.logging_utils import ColoredConciseEventFormatter

        formatter = ColoredConciseEventFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    elif log_style == "event":
        formatter = EventFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    else:
        from porki.logging_utils import ConciseEventFormatter

        formatter = ConciseEventFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream.setFormatter(formatter)
    handler = CompactingHandler(stream)
    handler.addFilter(EventContextFilter())
    root.addHandler(handler)


def _resolve_llm_config(args: argparse.Namespace) -> LLMRuntimeConfig:
    """Resolve provider config from generic flags and legacy Claude aliases."""
    provider = (args.llm_provider or "claude").strip().lower()
    executable = (args.llm_cli or "").strip()
    extra_args = tuple(args.llm_extra_arg or [])
    use_sysg_spawn = bool(args.llm_use_sysg)

    if args.llm_provider is None:
        if args.claude_cli != "claude" or args.claude_extra_arg or args.claude_use_sysg:
            provider = "claude"
    if not executable:
        provider_binary = "claude" if provider == "claude" else "codex"
        if provider == "claude" and args.claude_cli != "claude":
            provider_binary = args.claude_cli
        executable = shutil.which(provider_binary) or provider_binary
    if not extra_args and args.claude_extra_arg and provider == "claude":
        extra_args = tuple(args.claude_extra_arg)
    if not use_sysg_spawn and args.claude_use_sysg and provider == "claude":
        use_sysg_spawn = True

    return LLMRuntimeConfig(
        provider=provider,
        executable=executable,
        extra_args=extra_args,
        use_sysg_spawn=use_sysg_spawn,
    )


def _normalize_prompt(value: str | None) -> str | None:
    """Normalize optional prompt text to None when blank."""
    if value is None:
        return None
    text = value.strip()
    return text or None


def _instruction_filename(name: str) -> str:
    """Convert user-provided instruction name into uppercase snake-case filename."""
    tokens = re.findall(r"[A-Za-z0-9]+", name.upper())
    if not tokens:
        raise ValueError("Instruction name must contain at least one alphanumeric character")
    return f"{'_'.join(tokens)}.md"


def _default_instruction_template(name: str) -> str:
    """Return a starter markdown template for role instructions."""
    display_name = " ".join(name.strip().split()) or "Role"
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = (
        "---\n"
        f'porki_instruction_template_version: "{INSTRUCTION_TEMPLATE_VERSION}"\n'
        f'porki_schema_version: "{INSTRUCTION_SCHEMA_VERSION}"\n'
        f'generated_at_utc: "{generated_at}"\n'
        'generated_by: "porki instructions create"\n'
        "---\n\n"
    )
    body = f"""# {display_name} Instructions

## Role
Describe the role clearly. Focus on responsibilities and boundaries.

## Working Directory
`<project-root-or-subdir>/`

## Prerequisites
List what must exist before this role starts.

## Responsibilities
- Primary responsibility 1
- Primary responsibility 2
- Primary responsibility 3

## Deliverables
- Deliverable 1
- Deliverable 2
- Deliverable 3
"""
    return f"{metadata}{body}\n{render_instruction_schema_reference()}"


def _handle_instructions_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Handle `porki instructions` subcommands."""
    if args.instructions_command == "validate":
        parser_obj = InstructionParser(args.path.expanduser().resolve())
        agents = parser_obj.parse_agents()
        LOGGER.info(
            "Validated instruction file %s (schema v2, agents=%d)",
            parser_obj.instructions_path,
            len(agents),
        )
        print(f"valid: {parser_obj.instructions_path} agents={len(agents)}")
        return 0

    if args.instructions_command != "create":
        parser.error(f"Unsupported instructions command: {args.instructions_command}")

    output_dir = args.path.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"Path must be a directory: {output_dir}")

    filename = _instruction_filename(args.name)
    target = output_dir / filename
    if target.exists() and not args.force:
        parser.error(f"Target already exists: {target} (use --force to overwrite)")

    content = _default_instruction_template(args.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    LOGGER.info("Created instruction template at %s", target)
    print(str(target))
    return 0


def _handle_run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Handle `porki run` command."""
    prompt = _normalize_prompt(args.prompt)
    if prompt:
        if args.role or args.instructions:
            parser.error("Use either --prompt or (--role and --instructions), not both")
        llm_config = _resolve_llm_config(args)
        llm_client = create_llm_client(llm_config)
        print(llm_client.run_prompt(prompt))
        return 0

    if not args.role:
        parser.error("`porki run` requires --role when --prompt is not provided")
    if not args.instructions:
        parser.error("`porki run` requires --instructions when --prompt is not provided")

    client = _redis_client_from_url(args.redis_url)
    store = RedisStore(client)
    llm_config = _resolve_llm_config(args)

    if args.role == "agent":
        if not args.agent_name or not args.goal_id or not args.heartbeat:
            parser.error("Agent role requires --agent-name, --goal-id, and --heartbeat")
        llm_client = create_llm_client(llm_config, redis_url=args.redis_url)
        runtime = AgentRuntime(
            agent_name=args.agent_name,
            agent_role=args.agent_role or args.agent_name,
            goal_id=args.goal_id,
            instructions_path=args.instructions,
            heartbeat_path=args.heartbeat,
            redis_store=store,
            llm_client=llm_client,
            loop_interval=args.loop_interval,
            lease_ttl=timedelta(seconds=args.lease_ttl),
            heartbeat_refresh_interval=timedelta(seconds=args.heartbeat_interval),
            instructions_refresh_interval=timedelta(seconds=args.instruction_interval),
            idle_log_interval=timedelta(seconds=args.idle_log_interval),
        )
        try:
            runtime.run()
        except KeyboardInterrupt:
            LOGGER.info("Shutdown requested (Ctrl+C). Stopping gracefully...")
            runtime.stop()
        return 0

    llm_client = create_llm_client(llm_config, redis_url=args.redis_url)
    orchestrator = Orchestrator(
        instructions_path=args.instructions,
        redis_store=store,
        redis_url=args.redis_url,
        llm_client=llm_client,
        spawn_adapter=RealSpawnAdapter(),
        poll_interval=args.poll_interval,
        heartbeat_interval=args.heartbeat_interval,
        instruction_interval=args.instruction_interval,
        idle_log_interval=args.idle_log_interval,
        llm_config=llm_config,
    )
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        LOGGER.info("Shutdown requested (Ctrl+C). Stopping gracefully...")
        orchestrator.stop()
    return 0


def run_cli(argv: list[str] | None = None) -> int:
    """Execute CLI entrypoint logic and return process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(
        args.log_level,
        use_color=getattr(args, "color", False),
        log_style=getattr(args, "log_style", "concise"),
    )

    if args.command == "instructions":
        return _handle_instructions_command(args, parser)
    if args.command == "run":
        return _handle_run_command(args, parser)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(run_cli())
