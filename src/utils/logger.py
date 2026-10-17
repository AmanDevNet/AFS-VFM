"""
AFS-VFM Centralized Logging and Performance Profiling Utilities
===============================================================

Provides structured, timestamped logging and latency benchmark timers
for Vision Foundation Model inference, degradation rendering, and statistical analysis.

Usage:
------
>>> from src.utils.logger import get_logger, BenchmarkTimer
>>> logger = get_logger("Inference")
>>> logger.info("Starting VFM evaluation pass...")
>>> with BenchmarkTimer("CLIP Zero-Shot Forward Pass", logger=logger):
...     # inference code
"""

import sys
import time
import logging
from typing import Optional, Any, Sequence, Callable


def get_logger(
    name: str = "AFS-VFM",
    level: int = logging.INFO,
    stream: Optional[Any] = None,
) -> logging.Logger:
    """Create or retrieve a preconfigured timestamped logger.

    Parameters
    ----------
    name : str
        Logger identifier (default: 'AFS-VFM').
    level : int
        Logging threshold (default: logging.INFO).
    stream : optional
        Output stream (default: sys.stdout).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger was already created
    if not logger.handlers:
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


class BenchmarkTimer:
    """Context manager and profiling timer for benchmarking execution latency.

    Attributes
    ----------
    task_name : str
        Descriptive label for the profiled section.
    logger : Optional[logging.Logger]
        If provided, logs start and elapsed timing automatically.
    elapsed_seconds : float
        Elapsed wall-clock time in seconds.
    elapsed_ms : float
        Elapsed time in milliseconds.
    """

    def __init__(
        self,
        task_name: str = "Task",
        logger: Optional[logging.Logger] = None,
        log_level: int = logging.INFO,
    ) -> None:
        self.task_name = task_name
        self.logger = logger
        self.log_level = log_level
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self.elapsed_seconds: float = 0.0

    def __enter__(self) -> "BenchmarkTimer":
        self._start_time = time.perf_counter()
        if self.logger:
            self.logger.log(self.log_level, f"Starting {self.task_name} ...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._end_time = time.perf_counter()
        self.elapsed_seconds = self._end_time - self._start_time
        if self.logger:
            if exc_type is not None:
                self.logger.error(
                    f"{self.task_name} failed after {self.elapsed_ms:.2f} ms ({exc_val})"
                )
            else:
                self.logger.log(
                    self.log_level,
                    f"Completed {self.task_name} in {self.elapsed_ms:.2f} ms ({self.elapsed_seconds:.4f} s)",
                )

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        return self.elapsed_seconds * 1000.0


def format_metric_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Format structured rows into a clean, dependency-free ASCII table.

    Parameters
    ----------
    headers : Sequence[str]
        Column headers.
    rows : Sequence[Sequence[Any]]
        Data rows matching column count.

    Returns
    -------
    str
        Formatted ASCII grid.
    """
    if not headers:
        return ""

    str_rows = [[str(cell) for cell in row] for row in rows]
    col_widths = [len(h) for h in headers]

    for row in str_rows:
        for idx, cell in enumerate(row):
            if idx < len(col_widths):
                col_widths[idx] = max(col_widths[idx], len(cell))

    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    lines = [separator, header_line, separator]
    for row in str_rows:
        row_line = "| " + " | ".join(
            (row[i] if i < len(row) else "").ljust(col_widths[i]) for i in range(len(col_widths))
        ) + " |"
        lines.append(row_line)
    lines.append(separator)

    return "\n".join(lines)
