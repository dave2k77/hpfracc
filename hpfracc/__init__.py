"""
High-Performance Fractional Calculus Library (hpfracc)

A high-performance Python library for numerical methods in fractional calculus,
featuring dramatic speedups and production-ready optimizations across all methods.

This library provides optimized implementations of:
- Core fractional derivatives: Caputo, Riemann-Liouville, Grünwald-Letnikov
- Advanced methods: Weyl, Marchaud, Hadamard, Reiz-Feller derivatives
- Special methods: Fractional Laplacian, Fractional Fourier Transform, Fractional Z-Transform, Fractional Mellin Transform
- GPU acceleration via JAX, PyTorch, and CuPy
- Parallel computing via NUMBA
"""

__version__ = "3.0.2"
__author__ = "Davian R. Chin"
__email__ = "d.r.chin@pgr.reading.ac.uk"
__affiliation__ = "Department of Biomedical Engineering, University of Reading"

# Keep the top-level package import extremely lightweight to avoid importing
# optional heavy dependencies (e.g., torch, jax, numba) during package import.
# Users should import symbols from submodules explicitly, e.g.:
#   from hpfracc.algorithms.optimized_methods import OptimizedCaputo

# Initialize JAX configuration early to prevent conflicts
# This must happen before any other module imports JAX
try:
    from .core.jax_config import initialize_jax_once
    initialize_jax_once()
except ImportError:
    pass  # core.jax_config may not be available in all installs

# Core definitions
from .core.definitions import FractionalOrder

# Optimized methods
from .algorithms.optimized_methods import (
    OptimizedRiemannLiouville,
    OptimizedCaputo,
    OptimizedGrunwaldLetnikov,
    optimized_riemann_liouville,
)

# Advanced methods
from .algorithms.advanced_methods import (
    WeylDerivative,
    MarchaudDerivative,
    HadamardDerivative,
    ReizFellerDerivative,
)

# Special methods
from .algorithms.special_methods import (
    FractionalLaplacian,
    FractionalFourierTransform,
    FractionalZTransform,
    FractionalMellinTransform,
)

# Integrals
from .core.integrals import (
    RiemannLiouvilleIntegral,
    CaputoIntegral,
)

# Novel derivatives
from .core.fractional_implementations import (
    CaputoFabrizioDerivative,
    AtanganaBaleanuDerivative,
    CaputoDerivative,
)

# Alias for backward compatibility
Caputo = CaputoDerivative

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__affiliation__",
    "OptimizedRiemannLiouville",
    "OptimizedCaputo",
    "OptimizedGrunwaldLetnikov",
    "optimized_riemann_liouville",
    "FractionalOrder",
    "WeylDerivative",
    "MarchaudDerivative",
    "FractionalLaplacian",
    "FractionalFourierTransform",
    "RiemannLiouvilleIntegral",
    "CaputoIntegral",
    "CaputoFabrizioDerivative",
    "AtanganaBaleanuDerivative",
    "Caputo",
    "CaputoDerivative",
    "HadamardDerivative",
    "ReizFellerDerivative",
    "FractionalZTransform",
    "FractionalMellinTransform",
]
