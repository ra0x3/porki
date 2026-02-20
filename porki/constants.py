"""Constants for orchestrator configuration."""

import logging
from datetime import timedelta

PROMPT_TIMEOUT_SECONDS = 900
"""Default timeout in seconds for each LLM prompt operation."""

HEARTBEAT_INTERVAL_SECONDS = 120
"""Default interval in seconds for reading heartbeat directive files."""

HEARTBEAT_REFRESH_INTERVAL = timedelta(seconds=HEARTBEAT_INTERVAL_SECONDS)
"""Timedelta form of heartbeat refresh cadence."""

INSTRUCTION_REFRESH_INTERVAL = timedelta(seconds=120)
"""Default interval as timedelta for reloading instruction files."""

DEFAULT_POLL_INTERVAL = 5.0
"""Default orchestrator reconcile-loop sleep interval in seconds."""

AGENT_LOOP_INTERVAL = 1.0
"""Default agent main-loop sleep interval in seconds."""

DEFAULT_LOGGING_ENABLED = False
"""Default logging enabled state."""

COLOR_DEBUG = "\033[36m"
"""ANSI color code for DEBUG level logs (light blue/cyan)."""

COLOR_INFO = "\033[32m"
"""ANSI color code for INFO level logs (green)."""

COLOR_WARNING = "\033[33m"
"""ANSI color code for WARNING level logs (yellow)."""

COLOR_ERROR = "\033[31m"
"""ANSI color code for ERROR level logs (red)."""

COLOR_CRITICAL = "\033[31m"
"""ANSI color code for CRITICAL level logs (red)."""

COLOR_RESET = "\033[0m"
"""ANSI color reset code."""

LOG_LEVEL_COLORS = {
    logging.DEBUG: COLOR_DEBUG,
    logging.INFO: COLOR_INFO,
    logging.WARNING: COLOR_WARNING,
    logging.ERROR: COLOR_ERROR,
    logging.CRITICAL: COLOR_CRITICAL,
}
"""Mapping of logging levels to their ANSI color codes."""
