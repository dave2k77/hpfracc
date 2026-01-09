#!/usr/bin/env python3
"""
HPFRACC Fractional Operators Demo

This script demonstrates the usage of all available fractional operators
in the HPFRACC library with practical examples and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add library to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from hpfracc.core.derivatives import create_fractional_derivative
from hpfracc.core.integrals import create_fractional_integral
from hpfracc.core.fractional_implementations import create_riesz_fisher_operator


def demo_classical_derivatives():
    """Demonstrate classical fractional derivatives."""
    print("=== Classical Fractional Derivatives ===")

    # Test function: f(x) = x^2
    def f(x): return x**2

    # Test points
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 0.5

    print(f"Test function: f(x) = x^2")
    print(f"Fractional order: α = {alpha}")
    print(f"Test points: x = {x}")
    print()

    # Riemann-Liouville derivative
    rl_derivative = create_fractional_derivative('riemann_liouville', alpha)
    rl_result = rl_derivative.compute(f, x)
    print(f"Riemann-Liouville D^{alpha}f(x): {rl_result}")

    # Caputo derivative
    caputo_derivative = create_fractional_derivative('caputo', alpha)
    caputo_result = caputo_derivative.compute(f, x)
    print(f"Caputo D^{alpha}f(x): {caputo_result}")

    # Grunwald-Letnikov derivative
    gl_derivative = create_fractional_derivative('grunwald_letnikov', alpha)
    gl_result = gl_derivative.compute(f, x)
    print(f"Grunwald-Letnikov D^{alpha}f(x): {gl_result}")

    print()


def demo_novel_derivatives():
    """Demonstrate novel fractional derivatives."""
    print("=== Novel Fractional Derivatives ===")

    # Test function: f(x) = exp(-x^2)
    def f(x): return np.exp(-x**2)

    # Test points
    x = np.array([0.0, 1.0, 2.0])
    alpha = 0.3

    print(f"Test function: f(x) = exp(-x^2)")
    print(f"Fractional order: α = {alpha}")
    print(f"Test points: x = {x}")
    print()

    try:
        # Caputo-Fabrizio derivative
        cf_derivative = create_fractional_derivative('caputo_fabrizio', alpha)
        cf_result = cf_derivative.compute(f, x)
        print(f"Caputo-Fabrizio D^{alpha}f(x): {cf_result}")
    except Exception as e:
        print(f"Caputo-Fabrizio: {e}")

    try:
        # Atangana-Baleanu derivative
        ab_derivative = create_fractional_derivative('atangana_baleanu', alpha)
        ab_result = ab_derivative.compute(f, x)
        print(f"Atangana-Baleanu D^{alpha}f(x): {ab_result}")
    except Exception as e:
        print(f"Atangana-Baleanu: {e}")

    print()


def demo_advanced_methods():
    """Demonstrate advanced fractional methods."""
    print("=== Advanced Fractional Methods ===")

    # Test function: f(x) = sin(x)
    def f(x): return np.sin(x)

    # Test points
    x = np.array([0.0, np.pi/4, np.pi/2])
    alpha = 0.5

    print(f"Test function: f(x) = sin(x)")
    print(f"Fractional order: α = {alpha}")
    print(f"Test points: x = {x}")
    print()

    try:
        # Weyl derivative
        weyl_derivative = create_fractional_derivative('weyl', alpha)
        weyl_result = weyl_derivative.compute(f, x)
        print(f"Weyl D^{alpha}f(x): {weyl_result}")
    except Exception as e:
        print(f"Weyl: {e}")

    try:
        # Marchaud derivative
        marchaud_derivative = create_fractional_derivative('marchaud', alpha)
        marchaud_result = marchaud_derivative.compute(f, x)
        print(f"Marchaud D^{alpha}f(x): {marchaud_result}")
    except Exception as e:
        print(f"Marchaud: {e}")

    try:
        # Hadamard derivative
        hadamard_derivative = create_fractional_derivative('hadamard', alpha)
        hadamard_result = hadamard_derivative.compute(f, x)
        print(f"Hadamard D^{alpha}f(x): {hadamard_result}")
    except Exception as e:
        print(f"Hadamard: {e}")

    print()


def demo_parallel_methods():
    """Demonstrate parallel-optimized methods."""
    print("=== Parallel-Optimized Methods ===")

    # Test function: f(x) = x^3
    def f(x): return x**3

    # Larger array for parallel processing
    x = np.linspace(0, 10, 1000)
    alpha = 0.7

    print(f"Test function: f(x) = x^3")
    print(f"Fractional order: α = {alpha}")
    print(f"Array size: {len(x)} points")
    print()

    try:
        # Parallel Riemann-Liouville
        parallel_rl = create_fractional_derivative(
            'parallel_riemann_liouville', alpha)
        parallel_rl_result = parallel_rl.compute(f, x)
        print(
            f"Parallel Riemann-Liouville D^{alpha}f(x): shape {parallel_rl_result.shape}")
        print(f"First 5 values: {parallel_rl_result[:5]}")
    except Exception as e:
        print(f"Parallel Riemann-Liouville: {e}")

    try:
        # Parallel Caputo
        parallel_caputo = create_fractional_derivative(
            'parallel_caputo', alpha)
        parallel_caputo_result = parallel_caputo.compute(f, x)
        print(
            f"Parallel Caputo D^{alpha}f(x): shape {parallel_caputo_result.shape}")
        print(f"First 5 values: {parallel_caputo_result[:5]}")
    except Exception as e:
        print(f"Parallel Caputo: {e}")

    print()


def demo_special_operators():
    """Demonstrate special fractional operators."""
    print("=== Special Fractional Operators ===")

    # Test function: f(x) = exp(-x^2)
    def f(x): return np.exp(-x**2)

    # Test points
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    print(f"Test function: f(x) = exp(-x^2)")
    print(f"Test points: x = {x}")
    print()

    try:
        # Fractional Laplacian
        laplacian = create_fractional_derivative('fractional_laplacian', 0.5)
        laplacian_result = laplacian.compute(f, x)
        print(f"Fractional Laplacian (-Δ)^(0.5)f(x): {laplacian_result}")
    except Exception as e:
        print(f"Fractional Laplacian: {e}")

    try:
        # Fractional Fourier Transform
        fft = create_fractional_derivative('fractional_fourier_transform', 0.5)
        fft_result = fft.compute(f, x)
        print(f"Fractional Fourier Transform F^(0.5)f(x): {fft_result}")
    except Exception as e:
        print(f"Fractional Fourier Transform: {e}")

    print()


def demo_riesz_fisher_operator():
    """Demonstrate the Riesz-Fisher operator."""
    print("=== Riesz-Fisher Operator ===")

    # Test function: f(x) = exp(-x^2)
    def f(x): return np.exp(-x**2)

    # Test points
    x = np.array([-1.0, 0.0, 1.0])

    print(f"Test function: f(x) = exp(-x^2)")
    print(f"Test points: x = {x}")
    print()

    # Derivative behavior (α > 0)
    rf_derivative = create_riesz_fisher_operator(0.5)
    derivative_result = rf_derivative.compute(f, x)
    print(f"Riesz-Fisher (α=0.5, derivative): {derivative_result}")

    # Integral behavior (α < 0)
    rf_integral = create_riesz_fisher_operator(-0.5)
    integral_result = rf_integral.compute(f, x)
    print(f"Riesz-Fisher (α=-0.5, integral): {integral_result}")

    # Identity behavior (α = 0)
    rf_identity = create_riesz_fisher_operator(0.0)
    identity_result = rf_identity.compute(f, x)
    print(f"Riesz-Fisher (α=0.0, identity): {identity_result}")

    print()


def demo_fractional_integrals():
    """Demonstrate fractional integrals."""
    print("=== Fractional Integrals ===")

    # Test function: f(x) = x
    def f(x): return x

    # Test points
    x = np.array([1.0, 2.0, 3.0])
    alpha = 0.5

    print(f"Test function: f(x) = x")
    print(f"Fractional order: α = {alpha}")
    print(f"Test points: x = {x}")
    print()

    try:
        # Riemann-Liouville integral
        rl_integral = create_fractional_integral("RL", alpha)
        rl_result = rl_integral(f, x)
        print(f"Riemann-Liouville I^{alpha}f(x): {rl_result}")
    except Exception as e:
        print(f"Riemann-Liouville integral: {e}")

    try:
        # Caputo integral
        caputo_integral = create_fractional_integral("Caputo", alpha)
        caputo_result = caputo_integral(f, x)
        print(f"Caputo I^{alpha}f(x): {caputo_result}")
    except Exception as e:
        print(f"Caputo integral: {e}")

    try:
        # Weyl integral
        weyl_integral = create_fractional_integral("Weyl", alpha)
        weyl_result = weyl_integral(f, x)
        print(f"Weyl I^{alpha}f(x): {weyl_result}")
    except Exception as e:
        print(f"Weyl integral: {e}")

    print()


def demo_performance_comparison():
    """Compare performance of different methods."""
    print("=== Performance Comparison ===")

    # Test function: f(x) = x^2
    def f(x): return x**2

    # Large array for performance testing
    x = np.linspace(0, 10, 5000)
    alpha = 0.5

    print(f"Test function: f(x) = x^2")
    print(f"Fractional order: α = {alpha}")
    print(f"Array size: {len(x)} points")
    print()

    import time

    # Test classical method
    start_time = time.time()
    classical_derivative = create_fractional_derivative(
        'riemann_liouville', alpha)
    classical_derivative.compute(f, x)
    classical_time = time.time() - start_time
    print(f"Classical Riemann-Liouville: {classical_time:.4f} seconds")

    # Test parallel method
    try:
        start_time = time.time()
        parallel_derivative = create_fractional_derivative(
            'parallel_riemann_liouville', alpha)
        parallel_derivative.compute(f, x)
        parallel_time = time.time() - start_time
        print(f"Parallel Riemann-Liouville: {parallel_time:.4f} seconds")
        print(f"Speedup: {classical_time/parallel_time:.2f}x")
    except Exception as e:
        print(f"Parallel method: {e}")

    print()


def create_visualization():
    """Create a visualization of different fractional derivatives."""
    print("=== Creating Visualization ===")

    # Test function: f(x) = exp(-x^2)
    def f(x): return np.exp(-x**2)

    # Create x array
    x = np.linspace(-3, 3, 200)
    alpha = 0.5

    # Compute different derivatives
    results = {}

    # Classical methods
    try:
        rl_derivative = create_fractional_derivative(
            'riemann_liouville', alpha)
        rl_result = rl_derivative.compute(f, x)
        # Ensure result is 1D array with same length as x
        if hasattr(rl_result, 'shape') and len(rl_result.shape) > 1:
            rl_result = rl_result.flatten()
        if len(rl_result) != len(x):
            rl_result = rl_result[:len(x)] if len(rl_result) > len(
                x) else np.pad(rl_result, (0, len(x) - len(rl_result)))
        results['Riemann-Liouville'] = rl_result
    except Exception as e:
        print(f"Riemann-Liouville failed: {e}")

    try:
        caputo_derivative = create_fractional_derivative('caputo', alpha)
        caputo_result = caputo_derivative.compute(f, x)
        # Ensure result is 1D array with same length as x
        if hasattr(caputo_result, 'shape') and len(caputo_result.shape) > 1:
            caputo_result = caputo_result.flatten()
        if len(caputo_result) != len(x):
            caputo_result = caputo_result[:len(x)] if len(caputo_result) > len(
                x) else np.pad(caputo_result, (0, len(x) - len(caputo_result)))
        results['Caputo'] = caputo_result
    except Exception as e:
        print(f"Caputo failed: {e}")

    try:
        gl_derivative = create_fractional_derivative(
            'grunwald_letnikov', alpha)
        gl_result = gl_derivative.compute(f, x)
        # Ensure result is 1D array with same length as x
        if hasattr(gl_result, 'shape') and len(gl_result.shape) > 1:
            gl_result = gl_result.flatten()
        if len(gl_result) != len(x):
            gl_result = gl_result[:len(x)] if len(gl_result) > len(
                x) else np.pad(gl_result, (0, len(x) - len(gl_result)))
        results['Grunwald-Letnikov'] = gl_result
    except Exception as e:
        print(f"Grunwald-Letnikov failed: {e}")

    # Riesz-Fisher
    try:
        rf_derivative = create_riesz_fisher_operator(alpha)
        rf_result = rf_derivative.compute(f, x)
        # Ensure result is 1D array with same length as x
        if hasattr(rf_result, 'shape') and len(rf_result.shape) > 1:
            rf_result = rf_result.flatten()
        if len(rf_result) != len(x):
            rf_result = rf_result[:len(x)] if len(rf_result) > len(
                x) else np.pad(rf_result, (0, len(x) - len(rf_result)))
        results['Riesz-Fisher'] = rf_result
    except Exception as e:
        print(f"Riesz-Fisher failed: {e}")

    # Create plot
    plt.figure(figsize=(12, 8))

    # Original function
    plt.plot(x, f(x), 'k-', linewidth=3, label='Original: f(x) = exp(-x²)')

    # Derivatives
    for name, result in results.items():
        try:
            plt.plot(x, result, '--', linewidth=2,
                     label=f'{name} D^{alpha}f(x)')
        except Exception as e:
            print(f"Failed to plot {name}: {e}")

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Fractional Derivatives of f(x) = exp(-x²) with α = {alpha}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig('fractional_derivatives_comparison.png',
                dpi=300, bbox_inches='tight')
    print("Visualization saved as 'fractional_derivatives_comparison.png'")

    plt.show()


def main():
    """Run all demonstrations."""
    print("HPFRACC Fractional Operators Demo")
    print("=" * 50)
    print()

    # Run all demos
    demo_classical_derivatives()
    demo_novel_derivatives()
    demo_advanced_methods()
    demo_parallel_methods()
    demo_special_operators()
    demo_riesz_fisher_operator()
    demo_fractional_integrals()
    demo_performance_comparison()

    # Create visualization
    try:
        create_visualization()
    except Exception as e:
        print(f"Visualization failed: {e}")

    print("\n" + "=" * 50)
    print("Demo completed! Check the generated plot for visual comparison.")
    print("For more details, see the documentation in docs/FRACTIONAL_OPERATORS_GUIDE.md")


if __name__ == "__main__":
    main()
