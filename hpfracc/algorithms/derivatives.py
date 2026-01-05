"""
Unified Fractional Derivative Interfaces.

These classes provide a single entry point for computing fractional derivatives,
automatically dispatching to the best available backend (NumPy, JAX, CUDA).
"""

import numpy as np
from typing import Union, Callable, Optional
from ..core.definitions import FractionalOrder
from .base import FractionalOperator
from .dispatch import BackendDispatcher
from .impls import numpy_backend, jax_backend, cuda_backend


class UnifiedFractionalOperator(FractionalOperator):
    """
    Base class for unified operators with backend dispatch.
    """
    def __init__(self, order: Union[float, FractionalOrder], backend: str = "auto"):
        super().__init__(order)
        self.backend_request = backend

    def _dispatch(self, f_arr, h):
        """
        Dispatches computation to the appropriate backend implementation.
        """
        N = len(f_arr)
        selected_backend = BackendDispatcher.get_backend(self.backend_request, N)
        
        return selected_backend


class RiemannLiouville(UnifiedFractionalOperator):
    """
    Riemann-Liouville fractional derivative.
    D^α f(t) = (d/dt)^n I^(n-α) f(t)
    """
    def __init__(self, order: Union[float, FractionalOrder], backend: str = "auto"):
        super().__init__(order, backend)
        self.n = int(np.ceil(self.alpha.alpha))

    def compute(self, f: Union[Callable, np.ndarray], t: Union[float, np.ndarray], h: float = None) -> np.ndarray:
        if h is not None and h <= 0:
            raise ValueError("Step size h must be positive")
        # Prepare inputs (common logic from base)
        # We assume base.FracOp.compute structure, but we override to control dispatch
        # Actually, base.compute does too much (it assumes JIT/NumPy split).
        # We should use helper methods from base but orchestrate here.
        
        t_array = self._get_t_array(f, t, h)
        f_array = self._prepare_f(f, t_array)
        
        if len(f_array) == 0:
            return np.array([])
            
        step_size = self._get_step_size(t_array, h)
        backend = self._dispatch(f_array, step_size)
        
        # Handle alpha=0, alpha=int cases broadly?
        if self.alpha.alpha == 0:
            return f_array
            
        if backend == "jax":
            try:
                # Assuming f is array-like, JAX backend handles conversion
                return np.asarray(jax_backend._riemann_liouville_jax(
                    f_array, self.alpha.alpha, self.n, step_size
                ))
            except Exception:
                # Fallback handled by re-dispatch logic? 
                # For now, strict or manual fallback
                pass
                
        elif backend == "cuda":
            try:
                return cuda_backend._riemann_liouville_cuda(
                    f_array, self.alpha.alpha, self.n, step_size
                )
            except NotImplementedError:
                pass
        
        # Default / Fallback to NumPy
        return numpy_backend._riemann_liouville_numpy(
            f_array, self.alpha.alpha, self.n, step_size
        )


class Caputo(UnifiedFractionalOperator):
    """
    Caputo fractional derivative.
    D^α f(t) = I^(n-α) f^(n)(t)
    """
    def compute(self, f: Union[Callable, np.ndarray], t: Union[float, np.ndarray], h: float = None) -> np.ndarray:
        if h is not None and h <= 0:
            raise ValueError("Step size h must be positive")
        t_array = self._get_t_array(f, t, h)
        f_array = self._prepare_f(f, t_array)
        
        if len(f_array) == 0: return np.array([])
        step_size = self._get_step_size(t_array, h)
        
        backend = self._dispatch(f_array, step_size)
        
        if backend == "jax":
            try:
                return np.asarray(jax_backend._caputo_jax(
                    f_array, self.alpha.alpha, step_size
                ))
            except Exception:
                pass
        elif backend == "cuda":
            # Caputo CUDA not implemented yet
            pass
            
        return numpy_backend._caputo_numpy(
            f_array, self.alpha.alpha, step_size
        )


class GrunwaldLetnikov(UnifiedFractionalOperator):
    """
    Grünwald-Letnikov fractional derivative.
    """
    def compute(self, f: Union[Callable, np.ndarray], t: Union[float, np.ndarray], h: float = None) -> np.ndarray:
        if h is not None and h <= 0:
            raise ValueError("Step size h must be positive")
        t_array = self._get_t_array(f, t, h)
        f_array = self._prepare_f(f, t_array)
        
        if len(f_array) == 0: return np.array([])
        step_size = self._get_step_size(t_array, h)
        
        backend = self._dispatch(f_array, step_size)
        
        if backend == "jax":
            try:
                return np.asarray(jax_backend._grunwald_letnikov_jax(
                    f_array, self.alpha.alpha, step_size
                ))
            except Exception:
                pass
        elif backend == "cuda":
            try:
                return cuda_backend._grunwald_letnikov_cuda(
                    f_array, self.alpha.alpha, step_size
                )
            except NotImplementedError:
                pass
                
        return numpy_backend._grunwald_letnikov_numpy(
            f_array, self.alpha.alpha, step_size
        )
