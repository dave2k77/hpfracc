"""
Convergence tests for fractional calculus numerical methods.

This module provides tools for analyzing the convergence rates
of numerical methods for fractional derivatives.
"""

import numpy as np
from typing import Callable, Dict, List
import warnings
from enum import Enum
from ..utils.error_analysis import ErrorAnalyzer


class OrderOfAccuracy(Enum):
    """Enumeration for expected orders of accuracy."""

    FIRST_ORDER = 1.0
    SECOND_ORDER = 2.0
    THIRD_ORDER = 3.0
    FOURTH_ORDER = 4.0


class ConvergenceTester:
    """Tester for analyzing convergence of numerical methods."""

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize the convergence tester.

        Args:
            tolerance: Numerical tolerance for convergence calculations
        """
        self.tolerance = tolerance

    def test_convergence(
        self,
        method_func: Callable,
        analytical_func: Callable,
        grid_sizes: List[int],
        test_params: Dict,
        error_norm: str = "l2",
    ) -> Dict:
        """
        Test convergence of a numerical method.

        Args:
            method_func: Function that computes numerical solution
            analytical_func: Function that computes analytical solution
            grid_sizes: List of grid sizes to test
            test_params: Parameters for the test case
            error_norm: Error norm to use ('l1', 'l2', 'linf')

        Returns:
            Convergence test results
        """

        error_analyzer = ErrorAnalyzer(tolerance=self.tolerance)
        errors = []
        successful_runs = 0

        for N in grid_sizes:
            try:
                # Create grid with proper bounds
                x = np.linspace(0, 1, max(N, 2))  # Ensure at least 2 points

                # Compute numerical solution with flexible parameter passing
                try:
                    numerical = method_func(x, **test_params)
                except TypeError as e:
                    # Only try fallback if the error is specifically about arguments
                    if "unexpected keyword argument" in str(e):
                        numerical = method_func(x, test_params)
                    else:
                        raise e

                # Compute analytical solution with flexible parameter passing
                try:
                    analytical = analytical_func(x, **test_params)
                except TypeError as e:
                    # Only try fallback if the error is specifically about arguments
                    if "unexpected keyword argument" in str(e):
                        analytical = analytical_func(x, test_params)
                    else:
                        raise e

                # Ensure both are numpy arrays
                numerical = np.asarray(numerical)
                analytical = np.asarray(analytical)

                # Check for valid results
                if np.any(np.isnan(numerical)) or np.any(np.isnan(analytical)):
                    warnings.warn(f"NaN values detected for N={N}")
                    errors.append(np.nan)
                    continue

                # Compute error with robust handling
                if error_norm == "l1":
                    error = error_analyzer.l1_error(numerical, analytical)
                elif error_norm == "l2":
                    error = error_analyzer.l2_error(numerical, analytical)
                elif error_norm == "linf":
                    error = error_analyzer.linf_error(numerical, analytical)
                elif error_norm == "mape":
                    error = error_analyzer.mean_absolute_percentage_error(
                        numerical, analytical
                    )
                elif error_norm == "smape":
                    error = error_analyzer.symmetric_mean_absolute_percentage_error(
                        numerical, analytical
                    )
                else:
                    raise ValueError(f"Unknown error norm: {error_norm}")

                # Check if error is valid
                if np.isnan(error) or np.isinf(error):
                    warnings.warn(f"Invalid error value for N={N}: {error}")
                    errors.append(np.nan)
                else:
                    errors.append(float(error))
                    successful_runs += 1

            except Exception as e:
                warnings.warn(f"Failed to compute error for N={N}: {e}")
                errors.append(np.nan)

        # Handle cases with insufficient valid data more gracefully
        valid_indices = [i for i, e in enumerate(errors) if not np.isnan(e)]

        if len(valid_indices) == 0:
            return {
                "grid_sizes": grid_sizes,
                "errors": errors,
                "convergence_rate": np.nan,
                "success": False,
                "message": "No valid error measurements obtained"
            }
        elif len(valid_indices) == 1:
            return {
                "grid_sizes": [grid_sizes[i] for i in valid_indices],
                "errors": [errors[i] for i in valid_indices],
                "convergence_rate": np.nan,
                "success": False,
                "message": "Only one valid measurement - cannot compute convergence rate"
            }

        valid_grid_sizes = [grid_sizes[i] for i in valid_indices]
        valid_errors = [errors[i] for i in valid_indices]

        # Compute convergence rate with robust handling
        try:
            convergence_rate = self._compute_convergence_rate(
                valid_grid_sizes, valid_errors
            )
        except Exception as e:
            warnings.warn(f"Failed to compute convergence rate: {e}")
            convergence_rate = np.nan

        return {
            "grid_sizes": valid_grid_sizes,
            "errors": valid_errors,
            "convergence_rate": convergence_rate,
            "error_norm": error_norm,
            "test_params": test_params,
            "all_grid_sizes": grid_sizes,
            "all_errors": errors,
        }

    def _compute_convergence_rate(
        self, grid_sizes: List[int], errors: List[float]
    ) -> float:
        """
        Compute convergence rate using linear regression on log-log plot.

        Args:
            grid_sizes: List of grid sizes
            errors: List of corresponding errors

        Returns:
            Convergence rate (order of accuracy)
        """
        if len(grid_sizes) < 2 or len(errors) < 2:
            raise ValueError(
                "Need at least 2 points to compute convergence rate")

        # Convert to log space
        log_n = np.log(np.array(grid_sizes))
        log_error = np.log(np.array(errors))

        # Linear regression: log(error) = -p * log(N) + C
        # where p is the convergence rate
        coeffs = np.polyfit(log_n, log_error, 1)
        # Negative because error decreases with N
        convergence_rate = -coeffs[0]

        return convergence_rate

    # Compatibility alias used by some tests
    def estimate_convergence_rate(self, grid_sizes: List[int], errors: List[float]) -> float:
        """Alias for computing convergence rate directly.

        Args:
            grid_sizes: List of grid sizes
            errors: List of corresponding errors

        Returns:
            Estimated convergence rate
        """
        return self._compute_convergence_rate(grid_sizes, errors)

    def test_multiple_norms(
        self,
        method_func: Callable,
        analytical_func: Callable,
        grid_sizes: List[int],
        test_params: Dict,
    ) -> Dict:
        """
        Test convergence using multiple error norms.

        Args:
            method_func: Function that computes numerical solution
            analytical_func: Function that computes analytical solution
            grid_sizes: List of grid sizes to test
            test_params: Parameters for the test case

        Returns:
            Convergence test results for multiple norms
        """
        norms = ["l1", "l2", "linf", "mape", "smape"]
        results = {}

        for norm in norms:
            try:
                results[norm] = self.test_convergence(
                    method_func, analytical_func, grid_sizes, test_params, norm
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to test convergence for norm {norm}: {e}")
                results[norm] = None

        return results


class ConvergenceAnalyzer:
    """Analyzer for comprehensive convergence studies."""

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize the convergence analyzer.

        Args:
            tolerance: Numerical tolerance for convergence calculations
        """
        self.tolerance = tolerance
        self.tester = ConvergenceTester(tolerance)

    def analyze(self, *args, **kwargs):
        """Alias for analyze_method_convergence for backward compatibility."""
        return self.analyze_method_convergence(*args, **kwargs)

    def analyze_method_convergence(
        self,
        method_func: Callable,
        analytical_func: Callable,
        test_cases: List[Dict],
        grid_sizes: List[int] = None,
    ) -> Dict:
        """
        Analyze convergence for multiple test cases.

        Args:
            method_func: Function that computes numerical solution
            analytical_func: Function that computes analytical solution
            test_cases: List of test case dictionaries
            grid_sizes: List of grid sizes to test (default: [50, 100, 200, 400])

        Returns:
            Comprehensive convergence analysis
        """
        if grid_sizes is None:
            grid_sizes = [50, 100, 200, 400]

        results = {
            "test_cases": [],
            "summary": {},
            "grid_sizes": grid_sizes,
        }

        convergence_rates = []

        for i, test_case in enumerate(test_cases):
            try:
                # Test convergence for this case
                case_result = self.tester.test_multiple_norms(
                    method_func, analytical_func, grid_sizes, test_case
                )

                # Extract convergence rates
                rates = {}
                for norm, result in case_result.items():
                    if result is not None:
                        rates[norm] = result["convergence_rate"]
                    else:
                        rates[norm] = np.nan

                case_summary = {
                    "case_index": i,
                    "test_params": test_case,
                    "convergence_rates": rates,
                    "success": True,
                }

                # Add to overall rates
                for norm, rate in rates.items():
                    if not np.isnan(rate):
                        convergence_rates.append(rate)

            except Exception as e:
                case_summary = {
                    "case_index": i,
                    "test_params": test_case,
                    "convergence_rates": {},
                    "success": False,
                    "error": str(e),
                }

            results["test_cases"].append(case_summary)

        # Compute summary statistics
        if convergence_rates:
            results["summary"] = {
                "mean_convergence_rate": np.mean(convergence_rates),
                "std_convergence_rate": np.std(convergence_rates),
                "min_convergence_rate": np.min(convergence_rates),
                "max_convergence_rate": np.max(convergence_rates),
                "total_test_cases": len(test_cases),
                "successful_test_cases": len(
                    [c for c in results["test_cases"] if c["success"]]
                ),
            }
        else:
            results["summary"] = {
                "mean_convergence_rate": np.nan,
                "std_convergence_rate": np.nan,
                "min_convergence_rate": np.nan,
                "max_convergence_rate": np.nan,
                "total_test_cases": len(test_cases),
                "successful_test_cases": 0,
            }

        return results

    def estimate_optimal_grid_size(
        self,
        errors: List[float],
        grid_sizes: List[int],
        target_accuracy: float,
    ) -> int:
        """
        Estimate optimal grid size for a target accuracy.

        Args:
            errors: List of errors
            grid_sizes: List of grid sizes
            target_accuracy: Target accuracy

        Returns:
            Estimated optimal grid size
        """
        if len(errors) < 2 or len(grid_sizes) < 2:
            raise ValueError(
                "Need at least 2 points to estimate optimal grid size")

        # Compute convergence rate
        try:
            convergence_rate = self.tester.estimate_convergence_rate(
                grid_sizes, errors)
        except (ValueError, np.linalg.LinAlgError):
            # If we can't compute convergence rate, use the largest grid size
            return max(grid_sizes)

        if convergence_rate <= 0:
            return max(grid_sizes)

        # Estimate using convergence rate: error ~ C * h^p
        # h_optimal = h_ref * (error_ref / error_target)^(1/p)
        reference_error = errors[-1]
        reference_grid_size = grid_sizes[-1]
        ratio = reference_error / target_accuracy
        optimal_size = int(reference_grid_size *
                           (ratio ** (1.0 / convergence_rate)))

        return max(optimal_size, 1)

    def validate_convergence_order(
        self,
        errors: List[float],
        grid_sizes: List[int],
        expected_order: float,
        tolerance: float = 0.5,
    ) -> Dict:
        """
        Validate if observed convergence rate matches expected order.

        Args:
            errors: List of errors
            grid_sizes: List of grid sizes
            expected_order: Expected order of accuracy
            tolerance: Tolerance for validation

        Returns:
            Validation result
        """
        # Compute observed convergence rate
        try:
            observed_rate = self.tester.estimate_convergence_rate(
                grid_sizes, errors)
        except (ValueError, np.linalg.LinAlgError):
            return {
                "convergence_rate": np.nan,
                "expected_rate": expected_order,
                "difference": np.nan,
                "tolerance": tolerance,
                "is_valid": False,
                "actual_order": np.nan,
                "order_achieved": None,
            }

        difference = abs(observed_rate - expected_order)
        is_valid = difference <= tolerance

        return {
            "convergence_rate": observed_rate,
            "expected_rate": expected_order,
            "difference": difference,
            "tolerance": tolerance,
            "is_valid": is_valid,
            "actual_order": observed_rate,
            "order_achieved": OrderOfAccuracy(expected_order) if is_valid else None,
        }

    def compare_methods_convergence(
        self,
        methods: List[str],
        grid_sizes: List[int],
        errors: Dict[str, List[float]],
    ) -> Dict:
        """
        Compare convergence for multiple methods.

        Args:
            methods: List of method names
            grid_sizes: List of grid sizes
            errors: Dictionary of {method: error_list}

        Returns:
            Convergence analysis results
        """
        convergence_rates = {}
        best_method = None
        best_rate = -np.inf

        for method in methods:
            if method in errors:
                try:
                    rate = self.tester.estimate_convergence_rate(
                        grid_sizes, errors[method])
                    convergence_rates[method] = rate
                    if rate > best_rate:
                        best_rate = rate
                        best_method = method
                except (ValueError, np.linalg.LinAlgError):
                    convergence_rates[method] = np.nan

        return {
            "convergence_rates": convergence_rates,
            "best_method": best_method,
            "grid_sizes": grid_sizes,
            "methods": methods,
        }


def run_convergence_study(
    method_func: Callable,
    analytical_func: Callable,
    grid_sizes: List[int],
    test_params: Dict = None,
) -> Dict:
    """
    Run a comprehensive convergence study.

    Args:
        method_func: Function that computes numerical solution
        analytical_func: Function that computes analytical solution
        grid_sizes: List of grid sizes to test
        test_params: Parameters for the test case

    Returns:
        Convergence study results
    """
    tester = ConvergenceTester()
    if test_params is None:
        test_params = {'x': np.linspace(0, 1, 10)}
    results = tester.test_multiple_norms(
        method_func, analytical_func, grid_sizes, test_params
    )

    # Add expected keys for compatibility
    results['grid_sizes'] = grid_sizes
    results['method_func'] = method_func.__name__ if hasattr(
        method_func, '__name__') else 'unknown'

    # Add summary
    results['summary'] = {
        'total_norms': len(results),
        'convergence_rates': {norm: data.get('convergence_rate', np.nan) for norm, data in results.items() if isinstance(data, dict)},
        'best_norm': max(results.keys(), key=lambda k: results[k].get('convergence_rate', -np.inf) if isinstance(results[k], dict) else -np.inf)
    }

    # Add top-level convergence_rate for compatibility
    convergence_rates = [data.get('convergence_rate', np.nan)
                         for data in results.values() if isinstance(data, dict)]
    results['convergence_rate'] = np.nanmean(
        convergence_rates) if convergence_rates else np.nan

    return results


def run_method_convergence_test(
    method_func: Callable,
    analytical_func: Callable,
    grid_sizes: List[int],
    test_params: Dict = None,
) -> Dict:
    """
    Test convergence of a numerical method.

    Args:
        method_func: Function that computes numerical solution
        analytical_func: Function that computes analytical solution
        grid_sizes: List of grid sizes to test
        test_params: Parameters for the test case

    Returns:
        Convergence test results
    """
    tester = ConvergenceTester()
    if test_params is None:
        test_params = {'x': np.linspace(0, 1, 10)}
    results = tester.test_multiple_norms(
        method_func, analytical_func, grid_sizes, test_params
    )

    # Extract convergence rate from l2 norm (most common)
    if 'l2' in results and results['l2'] is not None:
        convergence_rate = results['l2'].get('convergence_rate', np.nan)
    else:
        convergence_rate = np.nan

    # Add convergence_rate at top level for compatibility
    results['convergence_rate'] = convergence_rate

    return results


def estimate_convergence_rate(
        grid_sizes: List[int],
        errors: List[float]) -> float:
    """
    Estimate convergence rate from grid sizes and errors.

    Args:
        grid_sizes: List of grid sizes
        errors: List of corresponding errors

    Returns:
        Estimated convergence rate
    """
    if len(grid_sizes) < 2 or len(errors) < 2:
        raise ValueError("Need at least 2 points to estimate convergence rate")

    # Convert to log space
    log_n = np.log(np.array(grid_sizes))
    log_error = np.log(np.array(errors))

    # Linear regression
    coeffs = np.polyfit(log_n, log_error, 1)
    convergence_rate = -coeffs[0]

    return convergence_rate
