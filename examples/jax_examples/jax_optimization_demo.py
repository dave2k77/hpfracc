#!/usr/bin/env python3
"""
JAX Optimization Demo for Fractional Calculus

This example demonstrates the use of JAX for GPU acceleration, automatic
differentiation, and vectorization in fractional calculus computations.
"""

from hpfracc.algorithms.impls.jax_backend import (
    _caputo_jax,
    _riemann_liouville_jax,
    _grunwald_letnikov_jax,
)
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jacrev, hessian, vmap
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Updated imports for consolidated structure


def gpu_acceleration_demo():
    """Demonstrate GPU acceleration with JAX."""
    print("🚀 JAX GPU Acceleration Demo")
    print("=" * 50)

    # Create test data
    t = jnp.linspace(0, 5, 1000)
    h = t[1] - t[0]
    f = jnp.sin(t) * jnp.exp(-t / 2)

    # Test different derivative methods
    methods = {
        "Caputo": _caputo_jax,
        "Riemann-Liouville": _riemann_liouville_jax,
        "Grünwald-Letnikov": _grunwald_letnikov_jax,
    }

    alpha = 0.5
    results = {}
    timings = {}

    print(
        f"Computing fractional derivatives (α = {alpha}) for f(t) = sin(t) * exp(-t/2)"
    )
    print(f"Grid size: {len(t)} points")

    for method_name, method_func in methods.items():
        print(f"\n🧪 Testing {method_name}...")

        try:
            # Warm-up run
            if "riemann" in method_name.lower():
                _ = method_func(f, alpha, int(np.ceil(alpha)), h)
            else:
                _ = method_func(f, alpha, h)

            # Time the computation
            start_time = time.time()
            if "riemann" in method_name.lower():
                result = method_func(f, alpha, int(np.ceil(alpha)), h)
            else:
                result = method_func(f, alpha, h)
            end_time = time.time()

            results[method_name] = result
            timings[method_name] = end_time - start_time

            print(f"  ⏱️  Execution time: {timings[method_name]:.4f}s")
            print(f"  📊 Result shape: {result.shape}")
        except Exception as e:
            print(f"  ❌ {method_name} failed: {e}")
            print(f"  ⚠️  This is expected if JAX GPU support is not available")
            continue

    # Plot results
    plt.figure(figsize=(15, 10))

    # Original function
    plt.subplot(2, 2, 1)
    plt.plot(t, f, "k-", linewidth=2,
             label="Original: f(t) = sin(t) * exp(-t/2)")
    plt.xlabel("Time t")
    plt.ylabel("Function Value")
    plt.title("Original Function")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Derivatives
    colors = ["r", "b", "g"]
    for i, (method_name, result) in enumerate(results.items()):
        plt.subplot(2, 2, i + 2)
        plt.plot(t, f, "k-", linewidth=1, alpha=0.3, label="Original")
        plt.plot(
            t, result, color=colors[i], linewidth=2, label=f"{method_name} (α={alpha})"
        )
        plt.xlabel("Time t")
        plt.ylabel("Derivative Value")
        plt.title(
            f"{method_name} Derivative\nTime: {timings[method_name]:.4f}s")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    import os

    output_dir = os.path.join("examples", "jax_examples")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "gpu_acceleration_demo.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("✅ GPU acceleration demo completed!")


def automatic_differentiation_demo():
    """Demonstrate automatic differentiation capabilities."""
    print("\n🔬 Automatic Differentiation Demo")
    print("=" * 50)

    # Create test data
    t = jnp.linspace(0, 3, 100)
    h = t[1] - t[0]
    f = jnp.sin(t)
    alpha = 0.5

    print(f"Computing gradients and Jacobians for fractional derivative")
    print(f"Function: f(t) = sin(t), α = {alpha}")

    # Test automatic differentiation
    try:
        def caputo_for_grad(a):
            return _caputo_jax(f, a, h).sum()

        # Gradient with respect to alpha
        grad_alpha = grad(caputo_for_grad)(alpha)
        print(f"✅ Gradient w.r.t. α computed: {grad_alpha}")
        
        def caputo_for_jac(f_vals):
            return _caputo_jax(f_vals, alpha, h)

        # Jacobian with respect to function values
        jacobian = jacrev(caputo_for_jac)(f)
        print(f"✅ Jacobian w.r.t. f computed: shape {jacobian.shape}")

        # Hessian with respect to alpha
        hessian_val = hessian(caputo_for_grad)(alpha)
        print(f"✅ Hessian w.r.t. α computed: {hessian_val}")

    except Exception as e:
        print(f"⚠️  Automatic differentiation failed: {e}")
        print("This is expected if JAX is not properly configured for GPU")

    print("✅ Automatic differentiation demo completed!")


def vectorization_demo():
    """Demonstrate vectorization capabilities."""
    print("\n📊 Vectorization Demo")
    print("=" * 50)

    # Create test data
    t = jnp.linspace(0, 2, 100)
    h = t[1] - t[0]
    f = jnp.sin(t)

    # Test vectorization over different alpha values
    alphas = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])

    print(f"Vectorizing over {len(alphas)} different α values")

    try:
        # Vectorize over alpha values
        vectorized_results = vmap(_caputo_jax, in_axes=(None, 0, None))(f, alphas, h)

        print(
            f"✅ Vectorized computation completed: shape {vectorized_results.shape}")

        # Plot results
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, f, "k-", linewidth=2, label="Original: f(t) = sin(t)")
        plt.xlabel("Time t")
        plt.ylabel("Function Value")
        plt.title("Original Function")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        for i, alpha in enumerate(alphas):
            plt.plot(
                t,
                vectorized_results[i],
                linewidth=2,
                label=f"Caputo Derivative (α={alpha:.1f})",
            )

        plt.xlabel("Time t")
        plt.ylabel("Derivative Value")
        plt.title("Vectorized Caputo Derivatives")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "examples/jax_examples/vectorization_demo.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    except Exception as e:
        print(f"⚠️  Vectorization failed: {e}")
        print("This is expected if JAX is not properly configured")

    print("✅ Vectorization demo completed!")


def performance_benchmark():
    """Benchmark JAX performance against different methods."""
    print("\n⚡ Performance Benchmark")
    print("=" * 50)

    # Create test data of different sizes
    grid_sizes = [100, 500, 1000, 2000]
    alpha = 0.5

    results = {}

    for N in grid_sizes:
        print(f"\n📊 Testing grid size: {N}")

        t = jnp.linspace(0, 2, N)
        h = t[1] - t[0]
        f = jnp.sin(t)

        # Test different methods
        methods = {
            "Caputo GPU": _caputo_jax,
            "Riemann-Liouville GPU": _riemann_liouville_jax,
            "Grünwald-Letnikov GPU": _grunwald_letnikov_jax,
        }

        timings = {}

        for method_name, method_func in methods.items():
            try:
                # Time the computation
                start_time = time.time()
                if "riemann" in method_name.lower():
                    _ = method_func(f, alpha, int(np.ceil(alpha)), h).block_until_ready()
                else:
                    _ = method_func(f, alpha, h).block_until_ready()
                end_time = time.time()

                timings[method_name] = end_time - start_time
            except Exception as e:
                print(f"  ❌ {method_name} failed: {e}")
                timings[method_name] = float("inf")  # Mark as failed
                continue

        results[N] = timings

    # Plot performance comparison
    plt.figure(figsize=(12, 8))

    methods = list(results[grid_sizes[0]].keys())
    colors = ["r", "b", "g"]

    for i, method in enumerate(methods):
        times = [results[N][method] for N in grid_sizes]
        plt.loglog(
            grid_sizes,
            times,
            "o-",
            color=colors[i],
            label=method,
            linewidth=2,
            markersize=8,
        )

    plt.xlabel("Grid Size N")
    plt.ylabel("Execution Time (s)")
    plt.title("JAX Performance Benchmark")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    import os

    output_dir = os.path.join("examples", "jax_examples")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "performance_benchmark.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("✅ Performance benchmark completed!")


def fft_methods_demo():
    """Demonstrate FFT-based fractional derivative methods."""
    print("\n🌊 FFT Methods Demo")
    print("=" * 50)

    # Create test data
    t = jnp.linspace(0, 4, 200)
    h = t[1] - t[0]
    f = jnp.sin(2 * jnp.pi * t) * jnp.exp(-t / 2)
    alpha = 0.5

    print(f"Computing FFT-based fractional derivatives")
    print(f"Function: f(t) = sin(2πt) * exp(-t/2), α = {alpha}")

    try:
        # Test different FFT methods
        methods = {"Riemann-Liouville": _riemann_liouville_jax}
        results = {}

        for method_name, method_func in methods.items():
            result = method_func(f, alpha, int(np.ceil(alpha)), h)
            results[method_name] = result
            print(f"✅ {method_name} method completed")

        # Plot results
        plt.figure(figsize=(10, 5))

        # Original function
        plt.subplot(1, 2, 1)
        plt.plot(t, f, "k-", linewidth=2, label="Original")
        plt.xlabel("Time t")
        plt.ylabel("Function Value")
        plt.title("Original Function")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Spectral method
        plt.subplot(1, 2, 2)
        plt.plot(t, f, "k-", linewidth=1, alpha=0.3, label="Original")
        plt.plot(t, results["Riemann-Liouville"], "r-",
                 linewidth=2, label="Riemann-Liouville (FFT)")
        plt.xlabel("Time t")
        plt.ylabel("Derivative Value")
        plt.title("FFT-based Method")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "examples/jax_examples/fft_methods_demo.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    except Exception as e:
        print(f"⚠️  FFT methods failed: {e}")
        print("This is expected if JAX is not properly configured")

    print("✅ FFT methods demo completed!")


def main():
    """Run all JAX optimization examples."""
    print("🚀 JAX Optimization Demo for Fractional Calculus")
    print("=" * 60)

    # Run examples
    gpu_acceleration_demo()
    automatic_differentiation_demo()
    vectorization_demo()
    performance_benchmark()
    fft_methods_demo()

    print("\n🎉 All JAX optimization examples completed!")
    print("\n📁 Generated plots saved in 'examples/jax_examples/' directory")
    print("\n💡 Note: Some features may require proper JAX/GPU setup")


if __name__ == "__main__":
    main()
