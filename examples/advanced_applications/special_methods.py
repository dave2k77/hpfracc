"""
Examples for Special Fractional Calculus Methods

This module demonstrates the use of special fractional calculus methods:
- Fractional Laplacian
- Fractional Fourier Transform  
- Fractional Z-Transform

These examples show practical applications and usage patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add library to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import our special methods
from hpfracc.algorithms.special_methods import (
    FractionalLaplacian,
    FractionalFourierTransform,
    FractionalZTransform,
)


def example_fractional_laplacian():
    """Example: Fractional Laplacian of Gaussian function."""
    print("=== Fractional Laplacian Example ===")

    # Create domain
    x = np.linspace(-5, 5, 200)

    # Define Gaussian function
    def gaussian(x):
        return np.exp(-(x**2))

    # Compute function values
    f = gaussian(x)

    # Test different alpha values
    alpha_values = [0.5, 1.0, 1.5]
    methods = ["spectral", "finite_difference", "integral"]

    plt.figure(figsize=(15, 10))

    # Plot original function
    plt.subplot(2, 3, 1)
    plt.plot(x, f, 'b-', linewidth=2, label='Original Gaussian')
    plt.title('Original Function: Gaussian')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)

    # Plot results for different alpha values using spectral method
    plt.subplot(2, 3, 2)
    for alpha in alpha_values:
        laplacian = FractionalLaplacian(alpha)
        result = laplacian.compute(f, x, method="spectral")
        plt.plot(x, result, linewidth=2, label=f'α = {alpha}')

    plt.title('Fractional Laplacian (Spectral Method)')
    plt.xlabel('x')
    plt.ylabel('(-Δ)^(α/2) f(x)')
    plt.legend()
    plt.grid(True)

    # Plot results for different methods with alpha = 1.0
    plt.subplot(2, 3, 3)
    alpha = 1.0
    laplacian = FractionalLaplacian(alpha)

    for method in methods:
        result = laplacian.compute(f, x, method=method)
        plt.plot(x, result, linewidth=2, label=f'{method}')

    plt.title(f'Fractional Laplacian Methods (α = {alpha})')
    plt.xlabel('x')
    plt.ylabel('(-Δ)^(α/2) f(x)')
    plt.legend()
    plt.grid(True)

    # Performance comparison
    plt.subplot(2, 3, 4)
    method_times = {}
    for method in methods:
        start_time = time.time()
        result = laplacian.compute(f, x, method=method)
        method_times[method] = time.time() - start_time

    methods_list = list(method_times.keys())
    times_list = list(method_times.values())

    plt.bar(methods_list, times_list, color=['blue', 'green', 'red'])
    plt.title('Performance Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)

    # Error analysis
    plt.subplot(2, 3, 5)
    # Use spectral method as reference
    reference = laplacian.compute(f, x, method="spectral")

    for method in methods[1:]:  # Skip spectral (reference)
        result = laplacian.compute(f, x, method=method)
        error = np.abs(result - reference)
        plt.plot(x, error, linewidth=2, label=f'Error vs {method}')

    plt.title('Error Analysis')
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)

    # Alpha dependence
    plt.subplot(2, 3, 6)
    alpha_range = np.linspace(0.1, 1.9, 20)
    max_values = []

    for alpha in alpha_range:
        laplacian.alpha_val = alpha
        result = laplacian.compute(f, x, method="spectral")
        max_values.append(np.max(np.abs(result)))

    plt.plot(alpha_range, max_values, 'b-', linewidth=2)
    plt.title('Dependence on α')
    plt.xlabel('α')
    plt.ylabel('Max |(-Δ)^(α/2) f(x)|')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('examples/fractional_laplacian_example.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Performance times: {method_times}")
    print("Fractional Laplacian example completed!\n")


def example_fractional_fourier_transform():
    """Example: Fractional Fourier Transform of various functions."""
    print("=== Fractional Fourier Transform Example ===")

    # Create domain
    x = np.linspace(-3, 3, 100)

    # Define test functions
    def gaussian(x):
        return np.exp(-(x**2))

    def sinc_function(x):
        return np.sinc(x)

    def cosine_function(x):
        return np.cos(2 * np.pi * x)

    functions = {
        'Gaussian': gaussian,
        'Sinc': sinc_function,
        'Cosine': cosine_function
    }

    plt.figure(figsize=(15, 12))

    # Test different alpha values
    alpha_values = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    alpha_names = ['0', 'π/4', 'π/2', 'π', '3π/2']

    for i, (func_name, func) in enumerate(functions.items()):
        f = func(x)

        # Plot original function
        plt.subplot(3, 3, 3*i + 1)
        plt.plot(x, f, 'b-', linewidth=2, label=f'Original {func_name}')
        plt.title(f'{func_name} Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)

        # Plot FrFT for different alpha values
        plt.subplot(3, 3, 3*i + 2)
        frft = FractionalFourierTransform(0)

        for alpha, alpha_name in zip(alpha_values, alpha_names):
            frft.alpha_val = alpha
            u, result = frft.transform(f, x, method="discrete")
            plt.plot(u, np.real(result), linewidth=2,
                     label=f'α = {alpha_name}')

        plt.title(f'Fractional Fourier Transform of {func_name}')
        plt.xlabel('u')
        plt.ylabel('Re[FrFT(f)](u)')
        plt.legend()
        plt.grid(True)

        # Plot magnitude spectrum
        plt.subplot(3, 3, 3*i + 3)
        for alpha, alpha_name in zip(alpha_values, alpha_names):
            frft.alpha_val = alpha
            u, result = frft.transform(f, x, method="discrete")
            plt.plot(u, np.abs(result), linewidth=2, label=f'α = {alpha_name}')

        plt.title(f'Magnitude Spectrum of {func_name}')
        plt.xlabel('u')
        plt.ylabel('|FrFT(f)(u)|')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('examples/fractional_fourier_transform_example.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    # Performance test
    print("Performance test for FrFT:")
    frft = FractionalFourierTransform(np.pi/2)
    f = gaussian(x)

    start_time = time.time()
    u, result = frft.transform(f, x, method="discrete")
    discrete_time = time.time() - start_time

    start_time = time.time()
    u, result = frft.transform(f, x, method="spectral")
    spectral_time = time.time() - start_time

    print(f"Discrete method: {discrete_time:.4f}s")
    print(f"Spectral method: {spectral_time:.4f}s")
    print("Fractional Fourier Transform example completed!\n")


def example_fractional_z_transform():
    """Example: Fractional Z-Transform of discrete signals."""
    print("=== Fractional Z-Transform Example ===")

    # Create discrete signals
    N = 50
    n = np.arange(N)

    # Define different signals
    signals = {
        'Unit Step': np.ones(N),
        'Exponential': np.exp(-0.1 * n),
        'Sinusoidal': np.sin(0.2 * n),
        'Random': np.random.random(N)
    }

    plt.figure(figsize=(15, 10))

    # Test different alpha values
    alpha_values = [0.1, 0.5, 1.0, 1.5]

    for i, (signal_name, signal) in enumerate(signals.items()):
        # Plot original signal
        plt.subplot(2, 4, i + 1)
        plt.stem(n, signal)
        plt.title(f'{signal_name} Signal')
        plt.xlabel('n')
        plt.ylabel('x[n]')
        plt.grid(True)

        # Plot Z-transform magnitude on unit circle
        plt.subplot(2, 4, i + 5)
        z_transform = FractionalZTransform(0.5)

        # Points on unit circle
        theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
        z_unit = np.exp(1j * theta)

        for alpha in alpha_values:
            z_transform.alpha_val = alpha
            result = z_transform.transform(signal, z_unit, method="fft")
            plt.plot(theta, np.abs(result), linewidth=2, label=f'α = {alpha}')

        plt.title(f'Z-Transform Magnitude of {signal_name}')
        plt.xlabel('θ (radians)')
        plt.ylabel('|X(z)|')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('examples/fractional_z_transform_example.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    # Performance comparison
    print("Performance comparison for Z-transform:")
    signal = np.random.random(100)
    z_values = np.exp(1j * np.linspace(0, 2*np.pi, 50))
    z_transform = FractionalZTransform(0.5)

    start_time = time.time()
    result = z_transform.transform(signal, z_values, method="direct")
    direct_time = time.time() - start_time

    start_time = time.time()
    result = z_transform.transform(signal, z_values, method="fft")
    fft_time = time.time() - start_time

    print(f"Direct method: {direct_time:.4f}s")
    print(f"FFT method: {fft_time:.4f}s")
    print(f"Speedup: {direct_time/fft_time:.2f}x")
    print("Fractional Z-Transform example completed!\n")


def example_integration():
    """Example: Integration of special methods for advanced applications."""
    print("=== Integration Example ===")

    # Create domain
    x = np.linspace(-3, 3, 100)

    # Define a function
    def f(x):
        return np.exp(-(x**2)) * np.cos(2 * np.pi * x)

    # Compute function values
    y = f(x)

    plt.figure(figsize=(15, 10))

    # 1. Fractional Laplacian
    plt.subplot(2, 3, 1)
    laplacian = FractionalLaplacian(1.0)
    result_laplacian = laplacian.compute(y, x, method="spectral")
    plt.plot(x, y, 'b-', linewidth=2, label='Original')
    plt.plot(x, result_laplacian, 'r-', linewidth=2,
             label='Fractional Laplacian')
    plt.title('Fractional Laplacian')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)

    # 2. Fractional Fourier Transform
    plt.subplot(2, 3, 2)
    frft = FractionalFourierTransform(np.pi/2)
    u, result_frft = frft.transform(y, x, method="discrete")
    plt.plot(u, np.real(result_frft), 'g-', linewidth=2, label='Real part')
    plt.plot(u, np.imag(result_frft), 'm-',
             linewidth=2, label='Imaginary part')
    plt.title('Fractional Fourier Transform')
    plt.xlabel('u')
    plt.ylabel('FrFT(f)(u)')
    plt.legend()
    plt.grid(True)

    # 3. Z-Transform (discrete version)
    plt.subplot(2, 3, 3)
    # Sample the function at discrete points
    np.arange(20)
    x_discrete = np.linspace(-2, 2, 20)
    y_discrete = f(x_discrete)

    z_transform = FractionalZTransform(0.5)
    theta = np.linspace(0, 2*np.pi, 50, endpoint=False)
    z_unit = np.exp(1j * theta)
    result_z = z_transform.transform(y_discrete, z_unit, method="fft")

    plt.plot(theta, np.abs(result_z), 'c-', linewidth=2)
    plt.title('Fractional Z-Transform')
    plt.xlabel('θ (radians)')
    plt.ylabel('|X(z)|')
    plt.grid(True)

    # 4. Combined analysis
    plt.subplot(2, 3, 4)
    # Show how different alpha values affect the Laplacian
    alpha_values = [0.5, 1.0, 1.5]
    for alpha in alpha_values:
        laplacian.alpha_val = alpha
        result = laplacian.compute(y, x, method="spectral")
        plt.plot(x, result, linewidth=2, label=f'α = {alpha}')

    plt.title('Laplacian: Dependence on α')
    plt.xlabel('x')
    plt.ylabel('(-Δ)^(α/2) f(x)')
    plt.legend()
    plt.grid(True)

    # 5. FrFT phase analysis
    plt.subplot(2, 3, 5)
    frft = FractionalFourierTransform(np.pi/4)
    u, result = frft.transform(y, x, method="discrete")
    plt.plot(u, np.angle(result), 'b-', linewidth=2)
    plt.title('FrFT Phase Analysis')
    plt.xlabel('u')
    plt.ylabel('Phase [radians]')
    plt.grid(True)

    # 6. Z-transform phase
    plt.subplot(2, 3, 6)
    plt.plot(theta, np.angle(result_z), 'r-', linewidth=2)
    plt.title('Z-Transform Phase')
    plt.xlabel('θ (radians)')
    plt.ylabel('Phase [radians]')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('examples/integration_example.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print("Integration example completed!\n")


def main():
    """Run all examples."""
    print("Special Fractional Calculus Methods Examples")
    print("=" * 50)

    # Create examples directory if it doesn't exist
    import os
    os.makedirs('examples', exist_ok=True)

    # Run examples
    example_fractional_laplacian()
    example_fractional_fourier_transform()
    example_fractional_z_transform()
    example_integration()

    print("All examples completed successfully!")
    print("Generated plots saved in 'examples/' directory.")


if __name__ == "__main__":
    main()
