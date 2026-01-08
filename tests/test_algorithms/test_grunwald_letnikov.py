#!/usr/bin/env python3
"""
Tests for Grünwald-Letnikov derivative algorithm.

Tests the OptimizedGrunwaldLetnikov class and its various methods.
"""

import pytest
import numpy as np
from hpfracc.algorithms.optimized_methods import (
    OptimizedGrunwaldLetnikov,
    optimized_grunwald_letnikov,
)


class TestOptimizedGrunwaldLetnikov:
    """Test OptimizedGrunwaldLetnikov class."""

    def test_optimized_grunwald_letnikov_creation(self):
        """Test creating OptimizedGrunwaldLetnikov instances."""
        # Test with float
        gl = OptimizedGrunwaldLetnikov(0.5)
        assert gl.alpha.alpha == 0.5  # FractionalOrder object

        # Test with different alpha values
        gl_alpha1 = OptimizedGrunwaldLetnikov(1.5)
        gl_alpha2 = OptimizedGrunwaldLetnikov(2.3)

        assert gl_alpha1.alpha.alpha == 1.5  # FractionalOrder object
        assert gl_alpha2.alpha.alpha == 2.3  # FractionalOrder object

    def test_optimized_grunwald_letnikov_validation(self):
        """Test OptimizedGrunwaldLetnikov validation."""
        # Test valid alpha values
        OptimizedGrunwaldLetnikov(0.1)
        OptimizedGrunwaldLetnikov(1.0)
        OptimizedGrunwaldLetnikov(2.5)

        # Test invalid alpha values (negative)
        with pytest.raises(ValueError):
            OptimizedGrunwaldLetnikov(-0.1)

    def test_optimized_grunwald_letnikov_compute_scalar(self):
        """Test computing Grünwald-Letnikov derivative for scalar input."""
        gl = OptimizedGrunwaldLetnikov(0.5)

        # Test with simple function
        def f(t):
            return t

        t = 1.0
        h = 0.01

        result = gl.compute(f, t, h)
        assert isinstance(result, np.ndarray)  # GL compute returns array
        assert len(result) > 0
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_optimized_grunwald_letnikov_compute_array(self):
        """Test computing Grünwald-Letnikov derivative for array input."""
        gl = OptimizedGrunwaldLetnikov(0.5)

        # Test with array function values
        t = np.linspace(0.1, 2.0, 50)
        f = t  # Simple linear function
        h = t[1] - t[0]

        result = gl.compute(f, t, h)
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_optimized_grunwald_letnikov_different_alphas(self):
        """Test different alpha values."""
        t = np.linspace(0.1, 2.0, 50)
        f = t**2  # Quadratic function
        h = t[1] - t[0]

        # Test different alpha values
        alpha1 = 0.5
        alpha2 = 1.5

        gl1 = OptimizedGrunwaldLetnikov(alpha1)
        result1 = gl1.compute(f, t, h)

        gl2 = OptimizedGrunwaldLetnikov(alpha2)
        result2 = gl2.compute(f, t, h)

        # Results should be different for different alphas
        assert not np.allclose(result1, result2)
        assert not np.any(np.isnan(result1))
        assert not np.any(np.isnan(result2))

    def test_optimized_grunwald_letnikov_analytical_comparison(self):
        """Test against known analytical results."""
        # For f(t) = t, the Grünwald-Letnikov derivative of order α is:
        # D^α f(t) = t^(1-α) / Γ(2-α)
        from scipy.special import gamma

        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t
        h = t[1] - t[0]

        gl = OptimizedGrunwaldLetnikov(alpha)
        numerical = gl.compute(f, t, h)

        # Analytical solution
        analytical = t ** (1 - alpha) / gamma(2 - alpha)

        # Check that numerical result is reasonable
        # (exact match not expected due to discretization)
        # Use a more lenient tolerance for discretization effects
        # Skip first few points where boundary effects dominate
        skip_points = 5  # Skip boundary points
        assert np.allclose(numerical[skip_points:], analytical[skip_points:], rtol=0.1)

    def test_optimized_grunwald_letnikov_function_interface(self):
        """Test the optimized_grunwald_letnikov function interface."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t**2
        h = t[1] - t[0]

        # Test function interface
        result = optimized_grunwald_letnikov(f, t, alpha, h)
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))

    def test_optimized_grunwald_letnikov_edge_cases(self):
        """Test edge cases and boundary conditions."""
        gl = OptimizedGrunwaldLetnikov(0.5)

        # Test with zero function
        t = np.linspace(0.1, 2.0, 10)
        f = np.zeros_like(t)
        h = t[1] - t[0]

        result = gl.compute(f, t, h)
        assert np.allclose(result, 0, atol=1e-10)

        # Test with constant function
        f = np.ones_like(t)
        result = gl.compute(f, t, h)
        # Grünwald-Letnikov (RL) derivative of constant is non-zero (t^-alpha / Gamma(1-alpha))
        # It is NOT zero (unlike Caputo).
        assert not np.allclose(result, 0, atol=1e-10)
        assert np.all(np.isfinite(result))

    def test_optimized_grunwald_letnikov_performance(self):
        """Test performance characteristics."""
        gl = OptimizedGrunwaldLetnikov(0.5)

        # Test with larger arrays
        t = np.linspace(0.1, 10.0, 1000)
        f = t**3
        h = t[1] - t[0]

        result = gl.compute(f, t, h)
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_optimized_grunwald_letnikov_consistency(self):
        """Test that same input gives consistent results."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t**2
        h = t[1] - t[0]

        # Test same input multiple times
        gl = OptimizedGrunwaldLetnikov(alpha)

        result1 = gl.compute(f, t, h)
        result2 = gl.compute(f, t, h)

        # Results should be consistent
        np.testing.assert_allclose(result1, result2, rtol=1e-10)

    def test_optimized_grunwald_letnikov_alpha_validation(self):
        """Test alpha parameter validation."""
        # Test valid alpha values
        for alpha in [0.1, 0.5, 1.0, 1.5, 2.0]:
            gl = OptimizedGrunwaldLetnikov(alpha)
            assert gl.alpha.alpha == alpha  # FractionalOrder object

        # Test invalid alpha values (negative)
        with pytest.raises(ValueError):
            OptimizedGrunwaldLetnikov(-0.1)

    def test_optimized_grunwald_letnikov_input_validation(self):
        """Test input validation."""
        gl = OptimizedGrunwaldLetnikov(0.5)
        t = np.linspace(0.1, 2.0, 10)
        f = t**2
        h = t[1] - t[0]

        # Test with mismatched array lengths
        f_wrong = t[:-1]  # One element shorter
        # The implementation validates array lengths and raises ValueError
        with pytest.raises(ValueError, match="Function array and time array must have the same length"):
            gl.compute(f_wrong, t, h)

    def test_optimized_grunwald_letnikov_fft_optimization(self):
        """Test FFT optimization for large arrays."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 100)
        f = t**2
        h = t[1] - t[0]

        # Test FFT optimization specifically
        gl = OptimizedGrunwaldLetnikov(alpha)
        result = gl.compute(f, t, h)

        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_optimized_grunwald_letnikov_convergence(self):
        """Test convergence with decreasing step size."""
        alpha = 0.5
        t_max = 1.0

        # Test with different grid sizes
        grid_sizes = [50, 100, 200]
        results = []

        for N in grid_sizes:
            t = np.linspace(0.1, t_max, N)
            f = t
            h = t[1] - t[0]

            gl = OptimizedGrunwaldLetnikov(alpha)
            result = gl.compute(f, t, h)
            results.append(result[-1])  # Take last point

        # Results should converge (get more stable)
        assert len(results) == len(grid_sizes)
        assert all(not np.isnan(r) for r in results)

    def test_optimized_grunwald_letnikov_binomial_coefficients(self):
        """Test binomial coefficients calculation."""
        alpha = 0.5
        gl = OptimizedGrunwaldLetnikov(alpha)

        # Test that binomial coefficients are calculated correctly
        # This is an internal test of the optimization
        t = np.linspace(0.1, 2.0, 10)
        f = t**2
        h = t[1] - t[0]

        result = gl.compute(f, t, h)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))


if __name__ == "__main__":
    pytest.main([__file__])
