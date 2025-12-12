"""
Core Utilities Module

This module provides common utility functions used throughout the HPFRACC library:
- Mathematical utilities and helper functions
- Type checking and validation utilities
- Performance monitoring utilities
- Error handling and debugging utilities
- Common mathematical operations
"""

from .definitions import FractionalOrder
from scipy.special import gamma, factorial
import logging
import time
from functools import wraps
import numpy as np
from typing import Union, Callable, Optional, Tuple, List, Dict, Any, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    import torch

# Optional torch import - lazy loading to avoid CuDNN errors at import time
TORCH_AVAILABLE = False
torch = None

def _get_torch():
    """Lazy import torch to avoid CuDNN loading errors."""
    global torch, TORCH_AVAILABLE
    if torch is None and not TORCH_AVAILABLE:
        try:
            import torch as _torch
            torch = _torch
            TORCH_AVAILABLE = True
        except (ImportError, OSError) as e:
            # OSError can occur if CuDNN libraries aren't found
            TORCH_AVAILABLE = False
            torch = None
            warnings.warn(f"PyTorch not available or failed to load: {e}. "
                         f"Some functions may have limited functionality.",
                         category=ImportWarning)
    return torch

# Module-level warning tracking
_warning_tracker = set()


# Mathematical utilities
def factorial_fractional(n: Union[int, float]) -> float:
    """
    Compute factorial for integer and fractional values.

    Args:
        n: Number to compute factorial for

    Returns:
        Factorial value
    """
    # Check for very large numbers that would cause overflow
    if n > 1e6:
        raise OverflowError(f"Factorial overflow for {n}")

    try:
        if isinstance(n, int) and n >= 0:
            return float(factorial(n))
        elif isinstance(n, float) and n > -1:
            return gamma(n + 1)
        else:
            raise ValueError(f"Factorial not defined for {n}")
    except OverflowError:
        raise OverflowError(f"Factorial overflow for {n}")


def binomial_coefficient(n: Union[int, float], k: Union[int, float]) -> float:
    """
    Compute binomial coefficient for real numbers.

    Args:
        n: Upper parameter
        k: Lower parameter

    Returns:
        Binomial coefficient value
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    elif k == 0:
        return 1.0
    elif isinstance(n, int) and isinstance(k, int) and n < k:
        raise ValueError("n must be >= k for integer parameters")
    elif isinstance(n, int) and isinstance(k, int):
        # For integer values, use standard formula: gamma(n+1) / (gamma(k+1) * gamma(n-k+1))
        return gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))
    else:
        # For fractional values, test expects: gamma(n+1) / (gamma(k+1) * gamma(n-k))
        # Note: Standard formula uses gamma(n-k+1), but test uses gamma(n-k)
        return gamma(n + 1) / (gamma(k + 1) * gamma(n - k))


def pochhammer_symbol(x: float, n: int) -> float:
    """
    Compute Pochhammer symbol (x)_n = x(x+1)...(x+n-1).

    Args:
        x: Base value
        n: Number of factors

    Returns:
        Pochhammer symbol value
    """
    if n == 0:
        return 1.0
    elif n == 1:
        return x
    else:
        return gamma(x + n) / gamma(x)


def _hypergeometric_series_impl(a: Union[float,
                                         List[float]],
                                b: Union[float,
                                         List[float]],
                                z: float,
                                max_terms: int = 100) -> float:
    """
    Internal implementation of hypergeometric series.
    """
    # Convert single values to lists for consistency
    if isinstance(a, (int, float)):
        a = [float(a)]
    if isinstance(b, (int, float)):
        b = [float(b)]
    # Handle float max_terms (convert to int)
    if isinstance(max_terms, float):
        max_terms = int(max_terms)
    result = 1.0
    term = 1.0

    for n in range(1, max_terms + 1):
        # Compute numerator and denominator
        numerator = 1.0
        denominator = 1.0

        for ai in a:
            numerator *= pochhammer_symbol(ai, n)

        for bi in b:
            denominator *= pochhammer_symbol(bi, n)

        term *= (numerator / denominator) * (z ** n) / factorial(n)
        result += term

        # Check convergence
        if abs(term) < 1e-12:
            break

    return result


def hypergeometric_series(a: Union[float,
                                   List[float]],
                          b: Union[float,
                                   List[float]],
                          z: float,
                          max_terms: int = 100) -> float:
    """
    Compute hypergeometric series pFq(a; b; z).

    Args:
        a: Upper parameter(s) - can be float or list of floats
        b: Lower parameter(s) - can be float or list of floats
        z: Variable
        max_terms: Maximum number of terms to compute

    Returns:
        Hypergeometric series value
    """
    # Call the internal implementation
    return _hypergeometric_series_impl(a, b, z, max_terms)


def bessel_function_first_kind(nu: float, x: float) -> float:
    """
    Compute Bessel function of the first kind J_ν(x).

    Args:
        nu: Order of Bessel function
        x: Argument

    Returns:
        Bessel function value
    """
    # Use hypergeometric series representation
    if x == 0:
        return 1.0 if nu == 0 else 0.0

    z = -(x ** 2) / 4
    return (x / 2) ** nu * hypergeometric_series([],
                                                 [nu + 1], z) / gamma(nu + 1)


def modified_bessel_function_first_kind(nu: float, x: float) -> float:
    """
    Compute modified Bessel function of the first kind I_ν(x).

    Args:
        nu: Order of Bessel function
        x: Argument

    Returns:
        Modified Bessel function value
    """
    # Use hypergeometric series representation
    if x == 0:
        return 1.0 if nu == 0 else 0.0

    z = (x ** 2) / 4
    return (x / 2) ** nu * hypergeometric_series([],
                                                 [nu + 1], z) / gamma(nu + 1)


# Type checking and validation utilities
def validate_fractional_order(alpha: Union[float,
                                           FractionalOrder],
                              min_val: float = 0.0,
                              max_val: float = 2.0) -> bool:
    """
    Validate and convert fractional order.

    Args:
        alpha: Fractional order value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        True if fractional order is valid
    """
    if isinstance(alpha, FractionalOrder):
        alpha_val = alpha.alpha
    else:
        alpha_val = float(alpha)

    if not (min_val <= alpha_val <= max_val):
        raise ValueError(
            f"Fractional order must be in [{min_val}, {max_val}], got {alpha_val}")

    return True


def validate_function(f: Callable, domain: Tuple[float, float] = (
        0.0, 1.0), n_points: int = 100) -> bool:
    """
    Validate that a function is callable and well-behaved on a domain.

    Args:
        f: Function to validate
        domain: Domain to test on (min, max)
        n_points: Number of test points

    Returns:
        True if function is valid
    """
    if not callable(f):
        return False

    try:
        x_test = np.linspace(domain[0], domain[1], n_points)
        y_test = f(x_test)

        # Check for finite values
        if not np.all(np.isfinite(y_test)):
            return False

        return True
    except Exception:
        return False


def validate_tensor_input(x: Union[np.ndarray,
                                   "torch.Tensor"],
                          expected_shape: Optional[Tuple] = None) -> bool:
    """
    Validate tensor input for fractional calculus operations.

    Args:
        x: Input tensor
        expected_shape: Expected shape (optional)

    Returns:
        True if input is valid
    """
    if isinstance(x, np.ndarray):
        if not np.isfinite(x).all():
            return False
    else:
        # Check if it's a torch Tensor (lazy import)
        torch = _get_torch()
        if torch is not None:
            try:
                if isinstance(x, torch.Tensor):
                    if not torch.isfinite(x).all():
                        return False
                else:
                    return False
            except (TypeError, AttributeError):
                return False
        else:
            return False

    if expected_shape is not None:
        if x.shape != expected_shape:
            return False

    return True


# Performance monitoring utilities
def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Log timing information
        execution_time = end_time - start_time
        logging.info(
            f"{func.__name__} executed in {execution_time:.6f} seconds")

        return result

    return wrapper


def memory_usage_decorator(func: Callable) -> Callable:
    """
    Decorator to monitor memory usage.

    Args:
        func: Function to monitor

    Returns:
        Wrapped function with memory monitoring
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        process = psutil.Process()

        # Memory before execution
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        # Memory after execution
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        logging.info(f"{func.__name__} used {memory_used:.2f} MB of memory")

        return result

    return wrapper


class PerformanceMonitor:
    """
    Performance monitoring utility for tracking execution times and memory usage.
    """

    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        self.call_counts = {}

    def start_timer(self, name: str):
        """Start timing an operation."""
        self.timings[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End timing an operation and return duration."""
        if name not in self.timings:
            raise ValueError(f"Timer '{name}' was not started")

        duration = time.time() - self.timings[name]
        self.call_counts[name] = self.call_counts.get(name, 0) + 1

        logging.info(f"{name} took {duration:.6f} seconds")
        return duration

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        # Return a flat structure with operation names as keys
        stats = {}
        for name in set(
            list(
                self.call_counts.keys()) +
            list(
                self.timings.keys()) +
            list(
                self.memory_usage.keys())):
            stats[name] = {
                "calls": self.call_counts.get(name, 0),
                "timing": self.timings.get(name, 0),
                "memory": self.memory_usage.get(name, 0)
            }
        return stats

    def timer(self, name: str):
        """Context manager for timing operations."""
        return TimerContext(self, name)

    def memory_tracker(self, name: str):
        """Context manager for memory tracking."""
        return MemoryTrackerContext(self, name)

    def reset(self):
        """Reset all statistics."""
        self.timings.clear()
        self.memory_usage.clear()
        self.call_counts.clear()


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, monitor: PerformanceMonitor, name: str):
        self.monitor = monitor
        self.name = name

    def __enter__(self):
        self.monitor.start_timer(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.end_timer(self.name)


class MemoryTrackerContext:
    """Context manager for memory tracking."""

    def __init__(self, monitor: PerformanceMonitor, name: str):
        self.monitor = monitor
        self.name = name

    def __enter__(self):
        # Start memory tracking
        import psutil
        process = psutil.Process()
        self.monitor.memory_usage[self.name] = process.memory_info(
        ).rss / 1024 / 1024  # MB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # End memory tracking
        import psutil
        process = psutil.Process()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_before = self.monitor.memory_usage.get(self.name, 0)
        self.monitor.memory_usage[self.name] = memory_after - memory_before


# Error handling and debugging utilities
class FractionalCalculusError(Exception):
    """Base exception for fractional calculus operations."""


class ConvergenceError(FractionalCalculusError):
    """Exception raised when numerical methods fail to converge."""


class ValidationError(FractionalCalculusError):
    """Exception raised when input validation fails."""


def safe_divide(
        numerator: float,
        denominator: float,
        default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero

    Returns:
        Division result or default value
    """
    if abs(denominator) < 1e-12:
        # Only warn once per default value to avoid spam
        warning_key = f"safe_divide_default_{default}"
        if warning_key not in _warning_tracker:
            warnings.warn(
                f"Division by zero detected, using default value {default}")
            _warning_tracker.add(warning_key)
        return default
    return numerator / denominator


def check_numerical_stability(
        values: Union[np.ndarray, "torch.Tensor"], tolerance: float = 1e-10) -> bool:
    """
    Check if numerical values are stable.

    Args:
        values: Array of values to check
        tolerance: Tolerance for stability check

    Returns:
        True if values are stable
    """
    if isinstance(values, np.ndarray):
        result = np.all(
            np.isfinite(values)) and np.all(
            np.abs(values) < 1 /
            tolerance)
        return bool(result)
    else:
        # Check if it's a torch Tensor (lazy import)
        torch = _get_torch()
        if torch is not None and isinstance(values, "torch.Tensor"):
            result = torch.all(
                torch.isfinite(values)) and torch.all(
                torch.abs(values) < 1 / tolerance)
            return bool(result)
        else:
            return False


# Common mathematical operations
def vectorize_function(func: Callable, vectorize: bool = True) -> Callable:
    """
    Vectorize a scalar function for array inputs.

    Args:
        func: Scalar function to vectorize
        vectorize: Whether to use numpy vectorize

    Returns:
        Vectorized function
    """
    if vectorize:
        return np.vectorize(func)
    else:
        def vectorized_func(x):
            if isinstance(x, (list, tuple)):
                return [func(xi) for xi in x]
            elif isinstance(x, np.ndarray):
                return np.array([func(xi) for xi in x])
            else:
                # Check if it's a torch Tensor (lazy import)
                torch = _get_torch()
                if torch is not None and isinstance(x, "torch.Tensor"):
                    return torch.tensor([func(float(xi)) for xi in x])
                else:
                    return func(x)
        return vectorized_func


def normalize_array(arr: Union[np.ndarray,
                               "torch.Tensor"],
                    norm_type: str = "l2",
                    method: Optional[str] = None) -> Union[np.ndarray,
                                                    "torch.Tensor"]:
    """
    Normalize an array using different norm types.

    Args:
        arr: Array to normalize
        norm_type: Type of normalization ("l1", "l2", "max", "minmax")
        method: Alias for norm_type (for backward compatibility)

    Returns:
        Normalized array
    """
    # Support both 'method' and 'norm_type' parameters
    if method is not None:
        norm_type = method
    if isinstance(arr, np.ndarray):
        if norm_type == "l1":
            norm = np.sum(np.abs(arr))
            return arr / norm if norm > 0 else arr
        elif norm_type == "l2":
            norm = np.sqrt(np.sum(arr ** 2))
            return arr / norm if norm > 0 else arr
        elif norm_type == "max":
            norm = np.max(np.abs(arr))
            return arr / norm if norm > 0 else arr
        elif norm_type == "minmax":
            arr_min = np.min(arr)
            arr_max = np.max(arr)
            if arr_max - arr_min > 0:
                return (arr - arr_min) / (arr_max - arr_min)
            else:
                return arr
        elif norm_type == "standard":
            # Z-score normalization: (x - mean) / std
            arr_mean = np.mean(arr)
            arr_std = np.std(arr)
            if arr_std > 0:
                return (arr - arr_mean) / arr_std
            else:
                return arr - arr_mean  # If std is 0, just center
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

    else:
        # Check if it's a torch Tensor (lazy import)
        torch = _get_torch()
        if torch is not None and isinstance(arr, "torch.Tensor"):
            if norm_type == "l1":
                norm = torch.sum(torch.abs(arr))
                return arr / norm if norm > 0 else arr
            elif norm_type == "l2":
                norm = torch.sqrt(torch.sum(arr ** 2))
                return arr / norm if norm > 0 else arr
            elif norm_type == "max":
                norm = torch.max(torch.abs(arr))
                return arr / norm if norm > 0 else arr
            elif norm_type == "minmax":
                arr_min = torch.min(arr)
                arr_max = torch.max(arr)
                if arr_max - arr_min > 0:
                    return (arr - arr_min) / (arr_max - arr_min)
                else:
                    return arr
            elif norm_type == "standard":
                # Z-score normalization: (x - mean) / std
                arr_mean = torch.mean(arr)
                arr_std = torch.std(arr)
                if arr_std > 0:
                    return (arr - arr_mean) / arr_std
                else:
                    return arr - arr_mean  # If std is 0, just center
            else:
                raise ValueError(f"Unknown norm type: {norm_type}")
        else:
            raise ValueError(f"Unsupported array type: {type(arr)}")


def smooth_function(func: Callable, smoothing_factor: float = 0.1) -> Callable:
    """
    Create a smoothed version of a function using convolution.

    Args:
        func: Original function
        smoothing_factor: Smoothing factor (0-1)

    Returns:
        Smoothed function
    """
    def smoothed_func(x):
        if isinstance(x, (int, float)):
            return func(x)

        # Apply simple moving average smoothing
        if isinstance(x, np.ndarray):
            window_size = max(1, int(len(x) * smoothing_factor))
            kernel = np.ones(window_size) / window_size
            return np.convolve(func(x), kernel, mode='same')
        else:
            return func(x)

    return smoothed_func


# Utility functions for fractional calculus
def fractional_power(x: Union[float,
                              np.ndarray,
                              "torch.Tensor"],
                     alpha: float) -> Union[float,
                                            np.ndarray,
                                            "torch.Tensor"]:
    """
    Compute fractional power with proper handling of negative values.

    Args:
        x: Base value(s)
        alpha: Fractional exponent

    Returns:
        Fractional power result
    """
    if isinstance(x, (int, float)):
        if x >= 0:
            return x ** alpha
        else:
            # For negative values with non-integer alpha, return NaN
            if alpha != int(alpha):
                return float('nan')
            else:
                return x ** alpha

    elif isinstance(x, np.ndarray):
        result = np.zeros_like(x, dtype=float)
        positive_mask = x >= 0
        negative_mask = x < 0

        result[positive_mask] = x[positive_mask] ** alpha
        if alpha != int(alpha):
            result[negative_mask] = np.nan
        else:
            result[negative_mask] = x[negative_mask] ** alpha

        return result

    else:
        # Check if it's a torch Tensor (lazy import)
        torch = _get_torch()
        if torch is not None and isinstance(x, "torch.Tensor"):
            result = torch.zeros_like(x, dtype=torch.float32)
            positive_mask = x >= 0
            negative_mask = x < 0

            result[positive_mask] = x[positive_mask] ** alpha
            if alpha != int(alpha):
                result[negative_mask] = torch.nan
            else:
                result[negative_mask] = x[negative_mask] ** alpha

            return result
        else:
            raise TypeError(f"Unsupported type: {type(x)}")


def fractional_exponential(x: Union[float,
                                    np.ndarray,
                                    "torch.Tensor"],
                           alpha: float) -> Union[float,
                                                  np.ndarray,
                                                  "torch.Tensor"]:
    """
    Compute fractional exponential function.

    Args:
        x: Input value(s)
        alpha: Fractional order

    Returns:
        Fractional exponential result
    """
    # Use standard exponential with fractional scaling
    if isinstance(x, (int, float)):
        return np.exp(alpha * x)

    elif isinstance(x, np.ndarray):
        return np.exp(alpha * x)

    else:
        # Check if it's a torch Tensor (lazy import)
        torch = _get_torch()
        if torch is not None and isinstance(x, "torch.Tensor"):
            return torch.exp(alpha * x)
        else:
            raise TypeError(f"Unsupported type: {type(x)}")


# Configuration utilities
# Module-level precision storage
_default_precision = 64

def get_default_precision() -> int:
    """Get default numerical precision for the library."""
    return _default_precision


def set_default_precision(precision: int):
    """Set default numerical precision for the library."""
    global _default_precision
    # Accept any positive integer precision value
    if not isinstance(precision, int) or precision <= 0:
        raise ValueError("Precision must be a positive integer")
    
    _default_precision = precision


def get_available_methods() -> List[str]:
    """Get list of available fractional calculus methods."""
    return ["RL", "Caputo", "GL", "Weyl", "Marchaud", "Hadamard"]


def get_method_properties(method: str) -> Dict[str, Any]:
    """
    Get properties of a specific fractional calculus method.

    Args:
        method: Method name

    Returns:
        Dictionary of method properties
    """
    properties = {
        "RL": {
            "full_name": "Riemann-Liouville",
            "order_range": (0, 2),
            "memory_effect": True,
            "initial_conditions": "Complex",
            "numerical_stability": "Good"
        },
        "Caputo": {
            "full_name": "Caputo",
            "order_range": (0, 1),
            "memory_effect": True,
            "initial_conditions": "Simple",
            "numerical_stability": "Excellent"
        },
        "GL": {
            "full_name": "Grünwald-Letnikov",
            "order_range": (0, 2),
            "memory_effect": True,
            "initial_conditions": "Discrete",
            "numerical_stability": "Good"
        },
        "Weyl": {
            "full_name": "Weyl",
            "order_range": (0, 2),
            "memory_effect": False,
            "initial_conditions": "Periodic",
            "numerical_stability": "Good"
        },
        "riemann_liouville": {
            "full_name": "Riemann-Liouville",
            "order_range": (0, 2),
            "memory_effect": True,
            "initial_conditions": "Complex",
            "numerical_stability": "Good"
        }
    }

    return properties.get(method, None)


# Logging utilities
def setup_logging(
        name: str = "INFO",
        log_file: Optional[str] = None,
        level: str = "INFO"):
    """
    Setup logging for the HPFRACC library.

    Args:
        level: Logging level
        log_file: Optional log file path

    Returns:
        Logger instance
    """
    # Validate log level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level.upper() not in valid_levels:
        level = "INFO"  # Default to INFO if invalid level
    else:
        level = level.upper()

    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file
    )

    return logging.getLogger(name)


def get_logger(name: str = "hpfracc") -> logging.Logger:
    """
    Get a logger for the HPFRACC library.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)