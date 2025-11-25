import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

#!/usr/bin/env python3
"""Coverage tests for special/gamma_beta.py - currently 70% coverage."""

import pytest
import numpy as np
from hpfracc.special.gamma_beta import *


class TestGammaBetaCoverage:
    """Tests to improve gamma_beta.py coverage."""
    
    def test_gamma_function_basic(self):
        """Test basic gamma function computation."""
        # Test for various inputs
        inputs = [0.5, 1.0, 1.5, 2.0, 2.5]
        for x in inputs:
            result = gamma_function(x)
            assert isinstance(result, (float, np.ndarray))
            assert np.isfinite(result)
            
    def test_beta_function_basic(self):
        """Test basic beta function computation."""
        # Test for various inputs
        pairs = [(0.5, 0.5), (1.0, 1.0), (1.5, 2.0), (2.0, 3.0)]
        for a, b in pairs:
            result = beta_function(a, b)
            assert isinstance(result, (float, np.ndarray))
            assert np.isfinite(result)
            
    def test_log_gamma_function(self):
        """Test log gamma function."""
        inputs = [0.5, 1.0, 2.0, 5.0, 10.0]
        for x in inputs:
            result = log_gamma_function(x)
            assert isinstance(result, (float, np.ndarray))
            assert np.isfinite(result)
            
    def test_digamma_function(self):
        """Test digamma function."""
        inputs = [0.5, 1.0, 2.0, 5.0]
        for x in inputs:
            result = digamma_function(x)
            assert isinstance(result, (float, np.ndarray))
            assert np.isfinite(result)
            
    def test_array_inputs(self):
        """Test with array inputs."""
        x_array = np.array([0.5, 1.0, 1.5, 2.0])
        
        gamma_result = gamma_function(x_array)
        assert isinstance(gamma_result, np.ndarray)
        assert gamma_result.shape == x_array.shape
        assert np.all(np.isfinite(gamma_result))
        
    def test_edge_cases(self):
        """Test edge cases."""
        # Small positive values
        small_vals = [0.01, 0.1, 0.001]
        for x in small_vals:
            result = gamma_function(x)
            assert np.isfinite(result)
            
        # Large values
        large_vals = [10.0, 50.0, 100.0]
        for x in large_vals:
            result = log_gamma_function(x)  # Use log for large values
            assert np.isfinite(result)
            
    def test_mathematical_properties(self):
        """Test mathematical properties."""
        # Gamma(n+1) = n! for integers
        for n in [1, 2, 3, 4]:
            gamma_result = gamma_function(n + 1)
            import math
            factorial = math.factorial(n)
            assert np.isclose(gamma_result, factorial, rtol=1e-10)
            
    def test_beta_symmetry(self):
        """Test beta function symmetry."""
        # Beta(a,b) = Beta(b,a)
        a, b = 2.5, 3.5
        beta1 = beta_function(a, b)
        beta2 = beta_function(b, a)
        assert np.isclose(beta1, beta2, rtol=1e-12)
        
    def test_error_handling(self):
        """Test error handling."""
        # Test with invalid inputs
        invalid_inputs = [-1.0, 0.0, -0.5]
        for x in invalid_inputs:
            try:
                result = gamma_function(x)
                # If no error, result should handle edge case gracefully
                assert isinstance(result, (float, np.ndarray))
            except (ValueError, OverflowError, RuntimeError):
                # Expected for invalid inputs
                pass






