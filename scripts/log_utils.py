"""Shared logging helpers for pipeline-wide run history."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LOGS_DIR = PROJECT_DIR / "logs"
PROJECT_LOG_PATH = LOGS_DIR / "project_log.txt"


def append_project_log(stage: str, status: str, lines: list[str]) -> None:
    """Append one stage summary to the cumulative project log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with PROJECT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Stage: {stage}\n")
        f.write(f"Status: {status}\n")
        for line in lines:
            f.write(f"{line}\n")
        f.write("\n")
