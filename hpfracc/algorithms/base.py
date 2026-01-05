"""
Base classes for fractional calculus algorithms.
"""

import numpy as np
from typing import Union, Callable
import warnings
from ..core.definitions import FractionalOrder

# Use centralized JAX configuration to prevent conflicts
try:
    from ..core.jax_config import get_jax_safely, is_jax_available
    jax, jnp = get_jax_safely()
    JAX_AVAILABLE = (jax is not None and jnp is not None)
except (ImportError, AttributeError):
    JAX_AVAILABLE = False
    jax = None
    jnp = None


class FractionalOperator:
    """
    Base class for fractional calculus operators.
    
    Handles:
    - Input validation (f, t, h)
    - JAX/NumPy backend selection logic
    - Common utility methods
    """
    
    def __init__(self, order: Union[float, FractionalOrder]):
        if isinstance(order, (int, float)):
            self.alpha = FractionalOrder(order)
        else:
            self.alpha = order

        if self.alpha.alpha < 0:
            raise ValueError(
                "Order must be non-negative for fractional derivative")

        self.fractional_order = self.alpha

    def compute(self, f: Union[Callable, np.ndarray, "jnp.ndarray"], t: Union[float, np.ndarray], h: float = None) -> np.ndarray:
        if h is not None and h <= 0:
            raise ValueError("Step size h must be positive")

        t_array = self._get_t_array(f, t, h)
        f_array = self._prepare_f(f, t_array)

        if len(f_array) == 0:
            return np.array([])

        if np.allclose(f_array, f_array[0]):
            return np.zeros_like(f_array)

        step_size = self._get_step_size(t_array, h)

        if step_size <= 0:
            raise ValueError("Step size must be positive")

        if JAX_AVAILABLE:
            try:
                f_array_jax = jnp.asarray(f_array)
                # This will be overridden in subclasses
                result = self._compute_jit(
                    f_array_jax, self.alpha.alpha, step_size)
                return np.asarray(result)
            except Exception as e:
                # JAX failed (likely due to CuDNN issues), fall back to NumPy
                warnings.warn(f"JAX computation failed ({e}), falling back to NumPy")
                return self._numpy_kernel(f_array, self.alpha.alpha, step_size)
        else:
            # Fallback for non-JAX environments
            return self._numpy_kernel(f_array, self.alpha.alpha, step_size)

    def _get_jax_kernel(self):
        raise NotImplementedError("Subclasses must implement _get_jax_kernel")

    def _numpy_kernel(self, f, alpha, h):
        raise NotImplementedError("Subclasses must implement _numpy_kernel")

    def _get_t_array(self, f: Callable, t: Union[float, np.ndarray], h: float) -> np.ndarray:
        t_is_array = hasattr(t, "__len__")
        if callable(f):
            if not t_is_array:
                # Create a t_array if t is scalar
                return np.arange(0, t + (h or 0.01), (h or 0.01))
            return t
        else:
            if t_is_array:
                return np.asarray(t)
            else:
                # If f is an array but t is not, we can't infer t_array
                raise ValueError(
                    "Time array `t` must be provided for array input `f` if `t` is not an array.")

    def _prepare_f(self, f: Union[Callable, np.ndarray, "jnp.ndarray"], t_array: np.ndarray) -> Union["jnp.ndarray", np.ndarray]:
        if callable(f):
            f_array = np.array([f(ti) for ti in t_array])
        else:
            f_array = np.asarray(f)

        if hasattr(t_array, "__len__") and len(f_array) != len(t_array):
            raise ValueError(
                "Function array and time array must have the same length")

        return f_array

    def _get_step_size(self, t_array: np.ndarray, h: float) -> float:
        if h is not None:
            return h
        elif hasattr(t_array, "__len__") and len(t_array) > 1:
            return t_array[1] - t_array[0]
        else:
            return 1.0

    @property
    def _compute_jit(self):
        if not hasattr(self, '_compute_jit_cache'):
            self._compute_jit_cache = self._get_jax_kernel()
        return self._compute_jit_cache
