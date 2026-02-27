"""CLI entrypoint for porki."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import yaml

from porki.compiler import compile_source, emit_compile_result, execute_source
from porki.intent import validate_file
from porki.llm import LLMRuntimeConfig, create_llm_client
from porki.logging_utils import CompactingHandler, EventContextFilter, EventFormatter

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
        prog="porki", description=f"Porki typed intent runtime (version {version})"
    )
    parser.add_argument(
        "--version", action="version", version=f"porki {version}", help="Show version and exit"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Execute schema-v4 instructions or submit a one-shot prompt to the configured LLM",
    )
    run_parser.add_argument("--instructions", type=Path, help="Primary instructions file")
    run_parser.add_argument(
        "-p",
        "--prompt",
        nargs="?",
        const="",
        help="Simple one-shot prompt sent directly to the configured LLM",
    )
    run_parser.add_argument("--log-level", default="INFO", help="Python logging level")
    run_parser.add_argument(
        "--log-style",
        choices=["concise", "event"],
        default="concise",
        help="Terminal log style: concise (default) or event (full context)",
    )
    run_parser.add_argument("--color", action="store_true", help="Enable colored logging output")
    run_parser.add_argument(
        "--run-id",
        help="Run identifier used by schema-v4 execution pipeline",
    )
    run_parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint output path used by schema-v4 execution pipeline",
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
        help="Validate a strict schema-v4 YAML instruction file",
    )
    validate_parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Instruction YAML file to validate",
    )
    validate_parser.add_argument("--log-level", default="INFO", help="Python logging level")

    compile_parser = instructions_subparsers.add_parser(
        "compile",
        help="Compile a schema-v4 source file into typed IR and legalize it",
    )
    compile_parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Instruction YAML file to compile",
    )
    compile_parser.add_argument(
        "--out",
        type=Path,
        help="Optional JSON output path for compile artifact",
    )
    compile_parser.add_argument("--log-level", default="INFO", help="Python logging level")

    execute_parser = instructions_subparsers.add_parser(
        "execute",
        help="Compile+legalize+execute a schema-v4 source with deterministic runtime",
    )
    execute_parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Instruction YAML file to execute",
    )
    execute_parser.add_argument(
        "--run-id",
        required=True,
        help="Stable run identifier used in runtime and checkpoints",
    )
    execute_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint JSON output path",
    )
    execute_parser.add_argument("--log-level", default="INFO", help="Python logging level")

    return parser


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


def _read_instruction_schema_version(path: Path) -> str:
    """Read instruction schema version from a YAML source path."""
    if not path.exists():
        return ""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("instruction_schema_version", "")).strip()


def _instruction_filename(name: str) -> str:
    """Convert user-provided instruction name into uppercase snake-case filename."""
    tokens = re.findall(r"[A-Za-z0-9]+", name.upper())
    if not tokens:
        raise ValueError("Instruction name must contain at least one alphanumeric character")
    return f"{'_'.join(tokens)}.yaml"


def _default_instruction_template(name: str) -> str:
    """Return a starter schema-v4 YAML template."""
    display_name = " ".join(name.strip().split()) or "Goal"
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    return (
        'instruction_schema_version: "4"\n'
        "goal_typing:\n"
        "  primary_goal_class: Transform\n"
        "  secondary_goal_classes: [Decide]\n"
        "goal:\n"
        f'  statement: "{display_name}"\n'
        "  requested_effects:\n"
        "    - primitive: fs.write\n"
        "      data_class: internal\n"
        "      recipient_class: internal\n"
        "      consent_class: na\n"
        "      legal_basis_class: na\n"
        "      reversibility_class: reversible\n"
        "inputs:\n"
        "  artifact_path: artifacts/output.txt\n"
        "policy:\n"
        "  profile: fast\n"
        "success:\n"
        "  rubric: output artifact exists\n"
        "assumptions:\n"
        "  - id: a1\n"
        "    statement: filesystem is writable\n"
        "    status: confirmed\n"
        "confidence:\n"
        f'  target_statement: "Plan can satisfy {display_name}"\n'
        "  calibration_source: none\n"
        "  interval:\n"
        "    low: 0.4\n"
        "    high: 0.8\n"
        "  assumption_sensitivity:\n"
        "    - assumption_id: a1\n"
        "      rank: 1\n"
        "      breaks_if_false: true\n"
        "  evidence_gap: []\n"
        "  last_updated_at_stage: capture\n"
        "_meta:\n"
        f'  generated_at_utc: "{generated_at}"\n'
        '  generated_by: "porki instructions create"\n'
    )


def _handle_instructions_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Handle `porki instructions` subcommands."""
    if args.instructions_command == "validate":
        source_path = args.path.expanduser().resolve()
        if not source_path.exists():
            report = {
                "valid": False,
                "schema_version": None,
                "diagnostics": [
                    {
                        "code": "source.file.missing",
                        "severity": "error",
                        "path": "$",
                        "message": f"Instruction file does not exist: {source_path}",
                    }
                ],
            }
            print(json.dumps(report, indent=2, sort_keys=True))
            return 1

        report = validate_file(source_path)
        LOGGER.info(
            "Validated instruction file %s (schema=%s, valid=%s)",
            source_path,
            report.schema_version,
            report.valid,
        )
        print(report.as_json())
        return 0 if report.valid else 1

    if args.instructions_command == "compile":
        source_path = args.path.expanduser().resolve()
        result = compile_source(source_path)
        if args.out:
            written = emit_compile_result(result, path=args.out.expanduser().resolve())
            LOGGER.info("Wrote compile artifact to %s", written)
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if result.valid else 1

    if args.instructions_command == "execute":
        source_path = args.path.expanduser().resolve()
        checkpoint_path = args.checkpoint.expanduser().resolve()
        result = execute_source(source_path, run_id=args.run_id, checkpoint_path=checkpoint_path)
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
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
        if args.instructions:
            parser.error("Use either --prompt or --instructions, not both")
        llm_config = _resolve_llm_config(args)
        llm_client = create_llm_client(llm_config)
        print(llm_client.run_prompt(prompt))
        return 0

    instruction_schema_version = ""
    instructions_path = args.instructions.expanduser().resolve() if args.instructions else None
    if instructions_path is not None:
        instruction_schema_version = _read_instruction_schema_version(instructions_path)

    if instruction_schema_version == "4":
        run_id = args.run_id or "run-default"
        checkpoint = args.checkpoint or Path(".porki") / "checkpoints" / f"{run_id}.json"
        checkpoint_path = checkpoint.expanduser().resolve()
        try:
            result = execute_source(
                instructions_path,
                run_id=run_id,
                checkpoint_path=checkpoint_path,
            )
        except ValueError:
            compiled = compile_source(instructions_path)
            print(json.dumps(compiled.model_dump(mode="json"), indent=2, sort_keys=True))
            return 1
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0

    parser.error(
        "`porki run` now requires schema-v4 instructions when not using --prompt; "
        "use `porki instructions execute --path ... --run-id ... --checkpoint ...`"
    )
    return 2


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
