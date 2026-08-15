"""Consistent console-and-file logging for experiment entry points."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from io import TextIOBase
from pathlib import Path
from typing import Any


def configure_experiment_logging(
    command_name: str,
    log_dir: str | Path = "logs",
) -> tuple[logging.Logger, Path]:
    """Create a fresh timestamped log file while retaining console output."""
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = directory / f"{command_name}_{timestamp}.log"
    logger = logging.getLogger(f"fuzzy_dynamics.{command_name}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("Log file: %s", log_path.resolve())

    def log_uncaught_exception(exc_type, exc_value, traceback) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, traceback)
            return
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, traceback),
        )

    sys.excepthook = log_uncaught_exception
    return logger, log_path


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Replace a JSON file only after its new contents have been written."""
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    temporary.replace(destination)


class _TeeStream(TextIOBase):
    def __init__(self, console: Any, file_handle: Any) -> None:
        self.console = console
        self.file_handle = file_handle

    def write(self, value: str) -> int:
        self.console.write(value)
        self.file_handle.write(value)
        return len(value)

    def flush(self) -> None:
        self.console.flush()
        self.file_handle.flush()

    def isatty(self) -> bool:
        return False


@contextmanager
def tee_output_to_log(command_name: str, log_dir: str | Path = "logs"):
    """Mirror legacy print/tqdm output to a timestamped file under ``logs/``."""
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = directory / f"{command_name}_{timestamp}.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        sys.stdout = _TeeStream(original_stdout, handle)
        sys.stderr = _TeeStream(original_stderr, handle)
        try:
            print(f"Log file: {log_path.resolve()}")
            yield log_path
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = original_stdout, original_stderr
