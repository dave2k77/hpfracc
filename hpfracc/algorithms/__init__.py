"""
Algorithms Module
================

Core fractional calculus algorithms with unified backend dispatch.

This module provides high-performance implementations of fractional derivatives and integrals,
automatically selecting the best available backend (NumPy, JAX, CUDA) for execution.

Classes
-------
RiemannLiouville
    Unified Riemann-Liouville derivative calculator.
Caputo
    Unified Caputo derivative calculator.
GrunwaldLetnikov
    Unified Grünwald-Letnikov derivative calculator.

Submodules
----------
advanced_methods
    Advanced derivatives (Weyl, Hadamard, etc.).
gpu_optimized_methods
    (Legacy) Direct GPU implementations.
integral_methods
    Fractional integrals.
novel_derivatives
    Novel derivatives (Caputo-Fabrizio, etc.).
"""

from .derivatives import (
    RiemannLiouville,
    Caputo,
    GrunwaldLetnikov,
    UnifiedFractionalOperator,
)

# Import legacy/specialized modules to keep them accessible
from . import advanced_methods
from . import integral_methods
from . import novel_derivatives

# Export the Unified API as primary
__all__ = [
    "RiemannLiouville",
    "Caputo",
    "GrunwaldLetnikov",
    "UnifiedFractionalOperator",
    "advanced_methods",
    "integral_methods",
    "novel_derivatives",
]
