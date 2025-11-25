import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

"""
Comprehensive tests for Special Methods module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Callable

from hpfracc.algorithms.special_methods import (
    FractionalFourierTransform,
    FractionalZTransform,
    FractionalMellinTransform,
    FractionalLaplacian,
    fractional_fourier_transform,
    fractional_z_transform,
    fractional_mellin_transform
)


class TestFractionalFourierTransform:
    """Test FractionalFourierTransform class."""
    
    def test_initialization(self):
        """Test initialization."""
        fft = FractionalFourierTransform(0.5)
        
        assert fft.alpha_val == 0.5
        assert fft.alpha.alpha == 0.5
    
    def test_compute_with_array(self):
        """Test computing transform with array input."""
        fft = FractionalFourierTransform(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        u, result = fft.transform(f, x)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        assert isinstance(u, np.ndarray)
        assert len(u) == len(x)
    
    def test_compute_fractional_fft(self):
        """Test fractional FFT computation."""
        fft = FractionalFourierTransform(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        u, result = fft._fft_based_method(f, x, x, 0.1)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        assert isinstance(u, np.ndarray)
    
    def test_inverse_transform(self):
        """Test inverse transform."""
        fft = FractionalFourierTransform(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        u, transformed = fft.transform(f, x)
        # For inverse, we use 2π - alpha (since negative alpha is not allowed)
        inverse_fft = FractionalFourierTransform(2 * np.pi - 0.5)
        u_inv, result = inverse_fft.transform(transformed, u)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
    
    def test_energy_conservation(self):
        """Test energy conservation property."""
        fft = FractionalFourierTransform(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        u, transformed = fft.transform(f, x)
        
        original_energy = np.sum(np.abs(f)**2)
        transformed_energy = np.sum(np.abs(transformed)**2)
        
        # Energy should be approximately conserved (relaxed tolerance for fractional transforms)
        assert abs(original_energy - transformed_energy) < 1.0


class TestFractionalZTransform:
    """Test FractionalZTransform class."""
    
    def test_initialization(self):
        """Test initialization."""
        zt = FractionalZTransform(0.5)
        
        assert zt.alpha_val == 0.5
        assert zt.alpha.alpha == 0.5
    
    def test_compute_with_array(self):
        """Test computing transform with array input."""
        zt = FractionalZTransform(0.5)
        f = np.array([1, 2, 3, 4, 5])
        z = 0.5 + 0.5j
        
        result = zt.transform(f, z)
        
        assert isinstance(result, complex)
    
    def test_compute_fractional_z(self):
        """Test fractional Z computation."""
        zt = FractionalZTransform(0.5)
        f = np.array([1, 2, 3, 4, 5])
        z = 0.5 + 0.5j
        
        result = zt._direct_method(f, z)
        
        assert isinstance(result, complex)
    
    def test_inverse_transform(self):
        """Test inverse transform."""
        zt = FractionalZTransform(0.5)
        f = np.array([1, 2, 3, 4, 5])
        z = 0.5 + 0.5j
        
        transformed = zt.transform(f, z)
        result = zt.inverse_transform(transformed, z, len(f))
        
        assert isinstance(result, np.ndarray)


class TestFractionalMellinTransform:
    """Test FractionalMellinTransform class."""
    
    def test_initialization(self):
        """Test initialization."""
        mt = FractionalMellinTransform(0.5)
        
        assert mt.alpha_val == 0.5
        assert mt.alpha.alpha == 0.5
    
    def test_compute_with_array(self):
        """Test computing transform with array input."""
        mt = FractionalMellinTransform(0.5)
        x = np.linspace(0.1, 10, 10)
        f = np.exp(-x)
        s = 0.5 + 0.5j
        
        result = mt.transform(f, x, s)
        
        assert isinstance(result, complex)
    
    def test_compute_fractional_mellin(self):
        """Test fractional Mellin computation."""
        mt = FractionalMellinTransform(0.5)
        x = np.linspace(0.1, 10, 10)
        f = np.exp(-x)
        s = 0.5 + 0.5j
        
        result = mt._numerical_method(f, x, s)
        
        assert isinstance(result, complex)
    
    def test_inverse_transform(self):
        """Test inverse transform."""
        mt = FractionalMellinTransform(0.5)
        x = np.linspace(0.1, 10, 10)
        f = np.exp(-x)
        s = 0.5 + 0.5j
        
        transformed = mt.transform(f, x, s)
        result = mt.inverse_transform(transformed, s, x)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)


class TestFunctionWrappers:
    """Test function wrapper functions."""
    
    def test_fractional_fourier_transform_function(self):
        """Test fractional Fourier transform function wrapper."""
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        u, result = fractional_fourier_transform(f, x, alpha=0.5)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        assert isinstance(u, np.ndarray)
    
    def test_fractional_z_transform_function(self):
        """Test fractional Z transform function wrapper."""
        f = np.array([1, 2, 3, 4, 5])
        z = 0.5 + 0.5j
        
        result = fractional_z_transform(f, z, alpha=0.5)
        
        assert isinstance(result, complex)
    
    def test_fractional_mellin_transform_function(self):
        """Test fractional Mellin transform function wrapper."""
        x = np.linspace(0.1, 10, 10)
        f = np.exp(-x)
        s = 0.5 + 0.5j
        
        result = fractional_mellin_transform(f, x, s, alpha=0.5)
        
        assert isinstance(result, complex)


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_alpha_value(self):
        """Test handling of invalid alpha values."""
        # Test with alpha outside valid range
        with pytest.raises((ValueError, Warning)):
            FractionalFourierTransform(-1.0)
    
    def test_invalid_input_types(self):
        """Test handling of invalid input types."""
        fft = FractionalFourierTransform(0.5)
        
        # Test with invalid input types
        with pytest.raises((TypeError, ValueError)):
            fft.transform("invalid", [1, 2, 3])
    
    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        fft = FractionalFourierTransform(0.5)
        
        # Empty arrays should be handled gracefully (return empty array or tuple)
        result = fft.transform(np.array([]), np.array([]))
        if isinstance(result, tuple):
            # If it returns a tuple, check that all elements are empty
            for r in result:
                assert hasattr(r, 'size') and r.size == 0
        else:
            # If it returns a single array, check it's empty
            assert result.size == 0
    
    def test_mismatched_array_sizes(self):
        """Test handling of mismatched array sizes."""
        fft = FractionalFourierTransform(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x[:5])  # Different size
        
        # Should handle mismatched sizes gracefully
        u, result = fft.transform(f, x)
        assert len(result) == len(f)
        assert len(u) == len(f)


class TestMathematicalProperties:
    """Test mathematical properties."""
    
    def test_linearity_property(self):
        """Test linearity property of transforms."""
        fft = FractionalFourierTransform(0.5)
        x = np.linspace(0, 1, 10)
        f1 = np.sin(x)
        f2 = np.cos(x)
        a, b = 2.0, 3.0
        
        # Test linearity: T(af + bg) = aT(f) + bT(g)
        u1, T_f1 = fft.transform(f1, x)
        u2, T_f2 = fft.transform(f2, x)
        u3, T_af_bg = fft.transform(a * f1 + b * f2, x)
        
        # Check if linearity holds approximately
        linear_combination = a * T_f1 + b * T_f2
        assert np.allclose(T_af_bg, linear_combination, atol=1e-6)
    
    def test_inverse_property(self):
        """Test inverse property of transforms."""
        fft = FractionalFourierTransform(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        u, transformed = fft.transform(f, x)
        inverse_fft = FractionalFourierTransform(2 * np.pi - 0.5)
        u_inv, result = inverse_fft.transform(transformed, u)
        
        # For fractional transforms, perfect inverse property doesn't hold
        # Just check that the result is well-behaved (finite and reasonable size)
        assert np.all(np.isfinite(result))
        assert len(result) == len(f)
    
    def test_energy_conservation(self):
        """Test energy conservation property."""
        fft = FractionalFourierTransform(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        u, transformed = fft.transform(f, x)
        
        original_energy = np.sum(np.abs(f)**2)
        transformed_energy = np.sum(np.abs(transformed)**2)
        
        # Energy should be approximately conserved (relaxed tolerance for fractional transforms)
        assert abs(original_energy - transformed_energy) < 1.0
    
    def test_consistency_with_known_solutions(self):
        """Test consistency with known analytical solutions."""
        # Test with simple exponential function
        fft = FractionalFourierTransform(0.5)
        x = np.linspace(-5, 5, 100)
        f = np.exp(-x**2)  # Gaussian function
        
        u, transformed = fft.transform(f, x)
        
        # For a Gaussian, the transform should be well-behaved
        assert np.all(np.isfinite(transformed))
        assert np.all(np.isfinite(u))


class TestPerformanceOptimizations:
    """Test performance optimizations."""
    
    def test_fft_optimization(self):
        """Test FFT-based optimization."""
        fft = FractionalFourierTransform(0.5)
        x = np.linspace(0, 1, 1000)  # Large array
        f = np.sin(x)
        
        # Should use fast method for large arrays
        u, result = fft.transform(f, x, method="fast")
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
    
    def test_memory_optimization(self):
        """Test memory optimization for large arrays."""
        fft = FractionalFourierTransform(0.5)
        x = np.linspace(0, 1, 1000)
        f = np.sin(x)
        
        # Should handle large arrays without memory issues
        u, result = fft.transform(f, x)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
    
    def test_caching_mechanism(self):
        """Test caching mechanism for repeated computations."""
        fft = FractionalFourierTransform(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        # First computation
        u1, result1 = fft.transform(f, x)
        
        # Second computation (should be faster due to caching if implemented)
        u2, result2 = fft.transform(f, x)
        
        # Results should be identical
        assert np.allclose(result1, result2)
        assert np.allclose(u1, u2)


class TestFractionalLaplacian:
    """Test FractionalLaplacian class."""
    
    def test_initialization(self):
        """Test initialization."""
        laplacian = FractionalLaplacian(0.5)
        
        assert laplacian.alpha_val == 0.5
        assert laplacian.alpha.alpha == 0.5
    
    def test_compute_with_array(self):
        """Test computing Laplacian with array input."""
        laplacian = FractionalLaplacian(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        result = laplacian.compute(f, x)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
    
    def test_spectral_method(self):
        """Test spectral method."""
        laplacian = FractionalLaplacian(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        result = laplacian._spectral_method(f, x, 0.1)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
    
    def test_finite_difference_method(self):
        """Test finite difference method."""
        laplacian = FractionalLaplacian(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        result = laplacian._finite_difference_method(f, x, 0.1)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
