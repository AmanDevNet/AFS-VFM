"""
AFS-VFM Degradation Engine
===========================

Public API
----------
- ``DegradationPipeline``           — class for fine-grained control
- ``generate_degradation_sequence`` — one-call convenience function
- ``SeverityScheduler``             — non-linear progression curves
"""

from .degradation import DegradationPipeline, generate_degradation_sequence
from .scheduler import SeverityScheduler

__all__ = [
    "DegradationPipeline",
    "generate_degradation_sequence",
    "SeverityScheduler",
]
