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
- ``load_image`` / ``save_image``   — image I/O helpers
"""

from .degradation import DegradationPipeline, generate_degradation_sequence
from .scheduler import SeverityScheduler
from .compound import CompoundDegradation, COMPOUND_RECIPES
from .utils import load_image, save_image

__all__ = [
    "DegradationPipeline",
    "generate_degradation_sequence",
    "SeverityScheduler",
    "CompoundDegradation",
    "COMPOUND_RECIPES",
    "load_image",
    "save_image",
]
