"""
AFS-VFM Degradation Engine
===========================

Public API
----------
- ``DegradationPipeline``           — class for fine-grained control
- ``generate_degradation_sequence`` — one-call convenience function
- ``SeverityScheduler``             — non-linear progression curves
- ``CompoundDegradation``           — multi-corruption composition
- ``COMPOUND_RECIPES``              — canonical benchmark recipes
"""

from .degradation import DegradationPipeline, generate_degradation_sequence
from .scheduler import SeverityScheduler
from .compound import CompoundDegradation, COMPOUND_RECIPES

__all__ = [
    "DegradationPipeline",
    "generate_degradation_sequence",
    "SeverityScheduler",
    "CompoundDegradation",
    "COMPOUND_RECIPES",
]
