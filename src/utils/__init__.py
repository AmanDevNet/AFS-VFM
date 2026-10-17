"""
AFS-VFM Common System Utilities
===============================

Public API
----------
- ``get_logger``          — formatted, timestamped logger factory
- ``BenchmarkTimer``      — latency measurement context manager and decorator
- ``format_metric_table`` — dependency-free ASCII tabular formatter
"""

from .logger import get_logger, BenchmarkTimer, format_metric_table

__all__ = [
    "get_logger",
    "BenchmarkTimer",
    "format_metric_table",
]
