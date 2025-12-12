"""
Error analysis and validation tools for fractional calculus computations.

This module provides tools for:
- Computing error metrics between numerical and analytical solutions
- Analyzing convergence rates
- Validating numerical methods against known solutions
- Error estimation and bounds
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional
import warnings


class ErrorAnalyzer:
    """Analyzer for computing various error metrics."""

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize the error analyzer.

        Args:
            tolerance: Numerical tolerance for avoiding division by zero
        """
        self.tolerance = tolerance

    def absolute_error(
        self, numerical: np.ndarray, analytical: np.ndarray
    ) -> np.ndarray:
        """
        Compute absolute error between numerical and analytical solutions.

        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array

        Returns:
            Absolute error array
        """
        return np.abs(numerical - analytical)

    def relative_error(
        self, numerical: np.ndarray, analytical: np.ndarray
    ) -> np.ndarray:
        """
        Compute relative error between numerical and analytical solutions.

        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array

        Returns:
            Relative error array
        """
        # Avoid division by zero
        denominator = np.abs(analytical)
        denominator = np.where(
            denominator < self.tolerance, self.tolerance, denominator
        )
        return np.abs(numerical - analytical) / denominator

    def l1_error(self, numerical: np.ndarray, analytical: np.ndarray) -> float:
        """
        Compute L1 error norm.

        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array

        Returns:
            L1 error norm
        """
        return np.mean(np.abs(numerical - analytical))

    def l2_error(self, numerical: np.ndarray, analytical: np.ndarray) -> float:
        """
        Compute L2 error norm.

        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array

        Returns:
            L2 error norm
        """
        return np.sqrt(np.mean((numerical - analytical) ** 2))

    def linf_error(
            self,
            numerical: np.ndarray,
            analytical: np.ndarray) -> float:
        """
        Compute L-infinity error norm.

        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array

        Returns:
            L-infinity error norm
        """
        return np.max(np.abs(numerical - analytical))

    def mean_squared_error(
        self, numerical: np.ndarray, analytical: np.ndarray
    ) -> float:
        """
        Compute mean squared error.

        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array

        Returns:
            Mean squared error
        """
        return np.mean((numerical - analytical) ** 2)

    def root_mean_squared_error(
        self, numerical: np.ndarray, analytical: np.ndarray
    ) -> float:
        """
        Compute root mean squared error.

        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array

        Returns:
            Root mean squared error
        """
        return np.sqrt(self.mean_squared_error(numerical, analytical))

    def maximum_error(
        self, numerical: np.ndarray, analytical: np.ndarray
    ) -> float:
        """
        Compute maximum error.

        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array

        Returns:
            Maximum error
        """
        return self.linf_error(numerical, analytical)

    def compute_all_errors(
        self, numerical: np.ndarray, analytical: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute all error metrics.

        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array

        Returns:
            Dictionary containing all error metrics
        """
        abs_error = self.absolute_error(numerical, analytical)
        rel_error = self.relative_error(numerical, analytical)

        return {
            "l1": self.l1_error(numerical, analytical),
            "l2": self.l2_error(numerical, analytical),
            "linf": self.linf_error(numerical, analytical),
            "absolute_error": abs_error,  # Return array
            "relative_error": rel_error,  # Return array
            "mean_absolute": np.mean(abs_error),  # Return scalar
            "mean_relative": np.mean(rel_error),  # Return scalar
            "mse": self.mean_squared_error(numerical, analytical),
            "rmse": self.root_mean_squared_error(numerical, analytical),
            "max_error": self.maximum_error(numerical, analytical),
        }

    def compute_error_metrics(
        self, numerical: np.ndarray, analytical: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute all error metrics (alias for compute_all_errors).

        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array

        Returns:
            Dictionary containing all error metrics
        """
        return self.compute_all_errors(numerical, analytical)


# Top-level convenience functions expected by tests
def _to_numpy(x):
    try:
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def absolute_error(analytical, numerical):
    a = _to_numpy(analytical)
    n = _to_numpy(numerical)
    return np.abs(n - a)


def relative_error(analytical, numerical, tolerance: float = 1e-10):
    a = _to_numpy(analytical)
    n = _to_numpy(numerical)
    denom = np.abs(a)
    denom = np.where(denom < tolerance, tolerance, denom)
    return np.abs(n - a) / denom


def l2_error(analytical, numerical) -> float:
    a = _to_numpy(analytical)
    n = _to_numpy(numerical)
    return float(np.sqrt(np.mean((n - a) ** 2)))


def max_error(analytical, numerical) -> float:
    a = _to_numpy(analytical)
    n = _to_numpy(numerical)
    return float(np.max(np.abs(n - a)))


def compute_rmse(analytical, numerical) -> float:
    return l2_error(analytical, numerical)


def error_statistics(analytical, numerical) -> Dict[str, float]:
    abs_err = absolute_error(analytical, numerical)
    return {
        "mean": float(np.mean(abs_err)),
        "std": float(np.std(abs_err)),
        "min": float(np.min(abs_err)),
        "max": float(np.max(abs_err)),
        "rmse": float(np.sqrt(np.mean(abs_err ** 2))),
    }


class ConvergenceAnalyzer:
    """Analyzer for studying convergence rates of numerical methods."""

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize the convergence analyzer.

        Args:
            tolerance: Numerical tolerance for convergence analysis
        """
        self.tolerance = tolerance

    def compute_convergence_rate(
        self, errors: np.ndarray, h_values: np.ndarray
    ) -> float:
        """
        Compute convergence rate using linear regression on log-log plot.

        Args:
            errors: Array of errors
            h_values: Array of step sizes or grid sizes

        Returns:
            Convergence rate (order of accuracy)
        """
        if len(errors) < 2 or len(h_values) < 2:
            raise ValueError(
                "Need at least 2 points to compute convergence rate")

        # Convert to log space
        log_h = np.log(np.array(h_values))
        log_error = np.log(np.array(errors))

        # Linear regression: log(error) = p * log(h) + C
        # where p is the convergence rate
        coeffs = np.polyfit(log_h, log_error, 1)
        convergence_rate = coeffs[0]

        return convergence_rate

    def analyze_convergence(
        self, methods: List[str], h_values: np.ndarray, errors: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyze convergence for multiple methods.

        Args:
            methods: List of method names
            h_values: Array of step sizes or grid sizes
            errors: Dictionary of error arrays for different methods

        Returns:
            Dictionary containing convergence analysis results
        """
        convergence_rates = {}
        convergence_orders = {}

        for method in methods:
            if method in errors:
                try:
                    rate = self.compute_convergence_rate(
                        errors[method], h_values)
                    convergence_rates[method] = rate
                    convergence_orders[method] = rate
                except (ValueError, np.linalg.LinAlgError) as e:
                    warnings.warn(
                        f"Could not compute convergence rate for {method}: {e}")
                    convergence_rates[method] = np.nan
                    convergence_orders[method] = np.nan
            else:
                convergence_rates[method] = np.nan
                convergence_orders[method] = np.nan

        # Find best method (highest convergence rate)
        valid_rates = {k: v for k, v in convergence_rates.items()
                       if not np.isnan(v)}
        best_method = max(valid_rates.items(), key=lambda x: x[1])[
            0] if valid_rates else None

        return {
            "convergence_rates": convergence_rates,
            "convergence_orders": convergence_orders,
            "best_method": best_method,
        }

    def estimate_optimal_grid_size(
        self, target_error: float, grid_sizes: List[int], errors: List[float]
    ) -> int:
        """
        Estimate optimal grid size for a target error.

        Args:
            target_error: Target error tolerance
            grid_sizes: List of grid sizes used in convergence study
            errors: List of corresponding errors

        Returns:
            Estimated optimal grid size
        """
        if len(grid_sizes) < 2:
            raise ValueError(
                "Need at least 2 points to estimate optimal grid size")

        convergence_rate = self.compute_convergence_rate(grid_sizes, errors)

        # Use the last point as reference
        N_ref = grid_sizes[-1]
        error_ref = errors[-1]

        # Estimate: N_opt = N_ref * (error_ref / target_error)^(1/p)
        N_opt = int(N_ref * (error_ref / target_error)
                    ** (1 / convergence_rate))

        return max(N_opt, 1)  # Ensure positive grid size

    def estimate_convergence_rate(
        self, h_values: np.ndarray, errors: np.ndarray
    ) -> float:
        """
        Estimate convergence rate (alias for compute_convergence_rate with swapped parameter order).

        Args:
            h_values: Array of step sizes or grid sizes
            errors: Array of errors

        Returns:
            Convergence rate (order of accuracy)
        """
        return self.compute_convergence_rate(errors, h_values)

    def analyze_convergence_behavior(
        self, h_values: np.ndarray, errors: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze convergence behavior from h_values and errors.

        Args:
            h_values: Array of step sizes or grid sizes
            errors: Array of errors

        Returns:
            Dictionary containing convergence analysis results
        """
        convergence_rate = self.estimate_convergence_rate(h_values, errors)
        
        # Check if converging (errors should decrease as h decreases)
        is_converging = convergence_rate > 0 and np.all(np.diff(errors) < 0)
        
        return {
            "convergence_rate": convergence_rate,
            "is_converging": bool(is_converging),
        }


class ValidationFramework:
    """Framework for validating numerical methods against analytical solutions."""

    def __init__(self, error_analyzer: Optional[ErrorAnalyzer] = None, tolerance: float = 1e-10):
        """
        Initialize the validation framework.

        Args:
            error_analyzer: Error analyzer instance (optional)
            tolerance: Numerical tolerance for validation
        """
        self.error_analyzer = error_analyzer or ErrorAnalyzer()
        self.convergence_analyzer = ConvergenceAnalyzer()
        self.tolerance = tolerance

    def validate_method(
        self,
        method_func: Callable,
        analytical_func: Callable,
        test_cases: List[Dict],
        grid_sizes: List[int] = None,
    ) -> Dict:
        """
        Validate a numerical method against analytical solutions.

        Args:
            method_func: Function that computes numerical solution
            analytical_func: Function that computes analytical solution
            test_cases: List of test case dictionaries
            grid_sizes: List of grid sizes for convergence study

        Returns:
            Validation results dictionary
        """
        if grid_sizes is None:
            grid_sizes = [50, 100, 200, 400]

        results = {"test_cases": [],
                   "convergence_study": {}, "overall_summary": {}}

        # Test individual cases
        for i, test_case in enumerate(test_cases):
            case_result = self._validate_single_case(
                method_func, analytical_func, test_case
            )
            results["test_cases"].append(case_result)

        # Convergence study
        if len(test_cases) > 0:
            convergence_result = self._convergence_study(
                method_func, analytical_func, test_cases[0], grid_sizes
            )
            results["convergence_study"] = convergence_result

        # Overall summary
        results["overall_summary"] = self._compute_summary(
            results["test_cases"])

        return results

    def _validate_single_case(
        self, method_func: Callable, analytical_func: Callable, test_case: Dict
    ) -> Dict:
        """Validate a single test case."""
        try:
            # Compute numerical solution
            numerical = method_func(**test_case["params"])

            # Compute analytical solution
            analytical = analytical_func(**test_case["params"])

            # Compute errors
            errors = self.error_analyzer.compute_all_errors(
                numerical, analytical)

            return {
                "case_name": test_case.get("name", f"Case_{len(test_case)}"),
                "success": True,
                "errors": errors,
                "numerical_shape": numerical.shape,
                "analytical_shape": analytical.shape,
            }

        except Exception as e:
            return {
                "case_name": test_case.get("name", f"Case_{len(test_case)}"),
                "success": False,
                "error": str(e),
                "errors": None,
            }

    def _convergence_study(
        self,
        method_func: Callable,
        analytical_func: Callable,
        test_case: Dict,
        grid_sizes: List[int],
    ) -> Dict:
        """Perform convergence study."""
        errors_by_metric = {"l2": [], "linf": []}

        for N in grid_sizes:
            try:
                # Update test case with new grid size
                params = test_case["params"].copy()
                params["N"] = N

                numerical = method_func(**params)
                analytical = analytical_func(**params)

                errors = self.error_analyzer.compute_all_errors(
                    numerical, analytical)
                errors_by_metric["l2"].append(errors["l2"])
                errors_by_metric["linf"].append(errors["linf"])

            except Exception:
                errors_by_metric["l2"].append(np.nan)
                errors_by_metric["linf"].append(np.nan)

        # Compute convergence rates
        methods = list(errors_by_metric.keys())
        h_values = np.array(grid_sizes)
        convergence_rates = self.convergence_analyzer.analyze_convergence(
            methods, h_values, errors_by_metric
        )

        return {
            "grid_sizes": grid_sizes,
            "errors": errors_by_metric,
            "convergence_rates": convergence_rates,
        }

    def validate_against_analytical(
        self,
        numerical: np.ndarray,
        analytical: np.ndarray,
        tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Validate numerical solution against analytical solution.

        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array
            tolerance: Error tolerance (uses self.tolerance if not provided)

        Returns:
            Dictionary containing validation results
        """
        if tolerance is None:
            tolerance = self.tolerance

        # Compute errors
        errors = self.error_analyzer.compute_all_errors(numerical, analytical)
        
        # Check if solution is valid (all errors within tolerance)
        is_valid = (
            errors.get("l2", np.inf) < tolerance
            and errors.get("linf", np.inf) < tolerance
            and not np.any(np.isnan(numerical))
            and not np.any(np.isinf(numerical))
        )

        return {
            "is_valid": bool(is_valid),
            "errors": errors,
        }

    def check_convergence(
        self,
        h_values: np.ndarray,
        errors: np.ndarray,
        min_rate: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Check if solution is converging based on h_values and errors.

        Args:
            h_values: Array of step sizes or grid sizes
            errors: Array of errors
            min_rate: Minimum convergence rate required

        Returns:
            Dictionary containing convergence check results
        """
        convergence_rate = self.convergence_analyzer.estimate_convergence_rate(
            h_values, errors
        )
        
        is_converging = convergence_rate >= min_rate and np.all(np.diff(errors) < 0)

        return {
            "is_converging": bool(is_converging),
            "convergence_rate": float(convergence_rate),
            "min_rate_required": float(min_rate),
        }

    def _compute_summary(self, test_cases: List[Dict]) -> Dict:
        """Compute overall summary of validation results."""
        successful_cases = [case for case in test_cases if case["success"]]

        if not successful_cases:
            return {"success_rate": 0.0, "average_errors": None}

        # Compute average errors across all successful cases
        error_metrics = ["l1", "l2", "linf", "mean_absolute", "mean_relative"]
        average_errors = {}

        for metric in error_metrics:
            values = [
                case["errors"][metric]
                for case in successful_cases
                if case["errors"] and metric in case["errors"]
            ]
            if values:
                average_errors[metric] = np.mean(values)

        return {
            "success_rate": len(successful_cases) / len(test_cases),
            "total_cases": len(test_cases),
            "successful_cases": len(successful_cases),
            "average_errors": average_errors,
        }

    def validate_solution(self, solution: np.ndarray) -> bool:
        """
        Validate a solution array.

        Args:
            solution: Solution array to validate

        Returns:
            True if solution is valid, False otherwise
        """
        if not isinstance(solution, np.ndarray):
            return False

        # Check for NaN values
        if np.any(np.isnan(solution)):
            return False

        # Check for infinite values
        if np.any(np.isinf(solution)):
            return False

        # Check for reasonable values
        if np.any(np.abs(solution) > 1e10):
            return False

        return True


# Convenience functions
def compute_error_metrics(
    numerical: np.ndarray, analytical: np.ndarray
) -> Dict[str, Any]:
    """Compute all error metrics for given solutions."""
    analyzer = ErrorAnalyzer()
    return analyzer.compute_all_errors(numerical, analytical)


def analyze_convergence(
    methods_or_grid_sizes, h_values_or_errors=None, errors=None
) -> Dict[str, Any]:
    """Analyze convergence rates for given data.

    Args:
        methods_or_grid_sizes: Either list of method names or grid sizes (h_values)
        h_values_or_errors: Either h_values array or errors dict (depending on first arg)
        errors: Errors dict (only if first arg is methods) or errors array if first arg is h_values

    Returns:
        Dictionary containing convergence analysis results
    """
    analyzer = ConvergenceAnalyzer()

    # Handle different calling patterns
    if h_values_or_errors is None:
        # Called with just grid_sizes, errors
        if isinstance(methods_or_grid_sizes, (list, tuple, np.ndarray)) and len(methods_or_grid_sizes) > 0:
            if isinstance(methods_or_grid_sizes[0], (int, float, np.integer, np.floating)):
                # First arg is grid sizes, second should be errors dict
                grid_sizes = np.array(methods_or_grid_sizes)
                if errors is not None:
                    # Convert to expected format
                    methods = list(errors.keys())
                    h_values = grid_sizes
                    return analyzer.analyze_convergence(methods, h_values, errors)
                else:
                    raise ValueError(
                        "When grid_sizes is provided, errors dict must be provided as second argument")
            else:
                # First arg is methods list
                methods = methods_or_grid_sizes
                raise ValueError(
                    "Methods list provided but h_values and errors not provided")
        else:
            raise ValueError("Invalid arguments provided")
    else:
        if errors is None:
            # Check if called with (h_values, errors) pattern where both are arrays
            if isinstance(methods_or_grid_sizes, (list, tuple, np.ndarray)) and isinstance(h_values_or_errors, (list, tuple, np.ndarray)):
                # Both are arrays - this is the (h_values, errors) pattern
                h_values = np.array(methods_or_grid_sizes)
                errors_array = np.array(h_values_or_errors)
                
                # Compute convergence rate directly
                convergence_rate = analyzer.estimate_convergence_rate(h_values, errors_array)
                
                return {
                    "convergence_rate": float(convergence_rate),
                    "rate": float(convergence_rate),
                }
            # Called with methods, h_values, errors
            elif isinstance(methods_or_grid_sizes, (list, tuple)) and all(isinstance(x, str) for x in methods_or_grid_sizes):
                methods = methods_or_grid_sizes
                h_values = np.array(h_values_or_errors)
                # errors should be the third argument but it's None, this is an error
                raise ValueError(
                    "When methods and h_values provided, errors dict must be provided as third argument")
            else:
                # Called with grid_sizes, errors
                grid_sizes = np.array(methods_or_grid_sizes)
                errors_dict = h_values_or_errors
                if isinstance(errors_dict, dict):
                    methods = list(errors_dict.keys())
                    h_values = grid_sizes
                    return analyzer.analyze_convergence(methods, h_values, errors_dict)
                else:
                    raise ValueError("When grid_sizes provided, errors must be a dict")
        else:
            # Called with methods, h_values, errors
            methods = methods_or_grid_sizes
            h_values = np.array(h_values_or_errors)
            return analyzer.analyze_convergence(methods, h_values, errors)


def validate_solution(*args) -> bool:
    """Validate a solution array or method validation.

    Args:
        If one argument: solution array to validate
        If two arguments: analytical, numerical arrays to compare
        If three arguments: method_func, analytical_func, test_cases for method validation

    Returns:
        Boolean validation result or method validation results
    """
    framework = ValidationFramework()

    if len(args) == 1:
        # Single argument: validate solution array
        solution = args[0]
        return framework.validate_solution(solution)
    elif len(args) == 2:
        # Two arguments: analytical, numerical arrays
        analytical, numerical = args
        result = framework.validate_against_analytical(numerical, analytical)
        return result.get("is_valid", False)
    elif len(args) == 3:
        # Three arguments: validate method
        method_func, analytical_func, test_cases = args
        return framework.validate_method(method_func, analytical_func, test_cases)
    else:
        raise ValueError(
            "validate_solution expects either 1 argument (solution array), 2 arguments (analytical, numerical), or 3 arguments (method_func, analytical_func, test_cases)")
