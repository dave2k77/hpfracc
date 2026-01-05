"""
Backend dispatch logic for fractional calculus algorithms.
"""

import warnings
import numpy as np
from ..core.jax_config import is_jax_available
# Check for CuPy availability
try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class BackendDispatcher:
    """
    Determines the appropriate backend for execution.
    """
    
    @staticmethod
    def get_backend(requested_backend: str, data_size: int = 0):
        """
        Selects the best backend based on request and availability.
        
        Args:
            requested_backend: 'auto', 'numpy', 'jax', or 'cuda'
            data_size: Number of data points (hint for 'auto')
            
        Returns:
            str: Selected backend name
        """
        if requested_backend != "auto":
            if requested_backend == "jax" and not is_jax_available():
                warnings.warn("JAX requested but not available. Falling back to NumPy.")
                return "numpy"
            if requested_backend == "cuda" and not CUPY_AVAILABLE:
                warnings.warn("CUDA (CuPy) requested but not available. Falling back to NumPy.")
                return "numpy"
            return requested_backend

        # Auto-selection logic
        # 1. Prefer JAX if available (generally fastest for compiled kernels)
        if is_jax_available():
            return "jax"
        
        # 2. Prefer CUDA if JAX not available but CuPy is (and data is large enough?)
        # For now, if CuPy is present, use it for potential speedup on large arrays, 
        # but overhead might be high for small. Let's strictly prefer JAX > CUDA > NumPy for now.
        if CUPY_AVAILABLE:
             # Heuristic: only use GPU for N > 1000 to avoid overhead?
             # But simplicity suggests availability-First.
             if data_size > 1000:
                 return "cuda"
        
        return "numpy"
