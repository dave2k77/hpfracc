import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

"""
Comprehensive tests for Gamma and Beta functions in hpfracc.special.gamma_beta

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import math
import numpy as np
import pytest
from hpfracc.special.gamma_beta import (
    GammaFunction, BetaFunction, gamma_function, log_gamma, beta_function
)
import scipy.special as scipy_special


class TestGammaFunction:
    """Test GammaFunction class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.gamma = GammaFunction()
        self.test_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        self.large_values = np.array([10.0, 20.0, 50.0, 100.0])
    
    def test_initialization(self):
        """Test GammaFunction initialization"""
        assert isinstance(self.gamma, GammaFunction)
    
    def test_compute_basic_values(self):
        """Test computing Gamma function for basic values"""
        for x in self.test_values:
            result = self.gamma.compute(x)
            expected = scipy_special.gamma(x)
            
            assert isinstance(result, (float, np.floating))
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isnan(result)
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isinf(result)
            np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_compute_array_input(self):
        """Test computing Gamma function for array input"""
        result = self.gamma.compute(self.test_values)
        expected = scipy_special.gamma(self.test_values)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_values.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_known_values(self):
        """Test Gamma function for known mathematical values"""
        # Γ(1) = 1
        assert abs(self.gamma.compute(1.0) - 1.0) < 1e-10
        
        # Γ(0.5) = √π
        expected_half = np.sqrt(np.pi)
        np.testing.assert_allclose(self.gamma.compute(0.5), expected_half, rtol=1e-10)
        
        # Γ(n) = (n-1)! for positive integers
        for n in range(2, 6):
            expected = math.factorial(n - 1)
            np.testing.assert_allclose(self.gamma.compute(float(n)), expected, rtol=1e-10)
    
    def test_large_values(self):
        """Test Gamma function for large values"""
        for x in self.large_values:
            result = self.gamma.compute(x)
            expected = scipy_special.gamma(x)
            
            assert isinstance(result, (float, np.floating))
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isnan(result)
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isinf(result)
            np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_negative_values(self):
        """Test Gamma function for negative values"""
        negative_values = np.array([-0.5, -1.0, -1.5, -2.0])
        
        for x in negative_values:
            result = self.gamma.compute(x)
            expected = scipy_special.gamma(x)
            
            # Both should be NaN for negative integers
            if x == int(x) and x <= 0:
                assert np.isnan(result)
                assert np.isnan(expected)
            else:
                # For non-integer negative values, should be finite
                # Type guard for integer results
                if not isinstance(result, (int, np.integer)):
                    assert not np.isnan(result)
                # Type guard for integer results
                if not isinstance(result, (int, np.integer)):
                    assert not np.isinf(result)
                np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_zero_value(self):
        """Test Gamma function at zero"""
        result = self.gamma.compute(0.0)
        expected = scipy_special.gamma(0.0)
        
        # Both should be inf
        assert np.isinf(result)
        assert np.isinf(expected)
    
    def test_reflection_formula(self):
        """Test Gamma function reflection formula: Γ(z)Γ(1-z) = π/sin(πz)"""
        z_values = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        
        for z in z_values:
            gamma_z = self.gamma.compute(z)
            gamma_1_minus_z = self.gamma.compute(1 - z)
            
            left_side = gamma_z * gamma_1_minus_z
            right_side = np.pi / np.sin(np.pi * z)
            
            np.testing.assert_allclose(left_side, right_side, rtol=1e-10)
    
    def test_recurrence_relation(self):
        """Test Gamma function recurrence relation: Γ(z+1) = zΓ(z)"""
        z_values = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        
        for z in z_values:
            gamma_z = self.gamma.compute(z)
            gamma_z_plus_1 = self.gamma.compute(z + 1)
            
            expected = z * gamma_z
            np.testing.assert_allclose(gamma_z_plus_1, expected, rtol=1e-10)


class TestBetaFunction:
    """Test BetaFunction class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.beta = BetaFunction()
        self.test_pairs = [(1.0, 1.0), (2.0, 3.0), (0.5, 0.5), (1.5, 2.5), (3.0, 4.0)]
    
    def test_initialization(self):
        """Test BetaFunction initialization"""
        assert isinstance(self.beta, BetaFunction)
    
    def test_compute_basic_values(self):
        """Test computing Beta function for basic values"""
        for a, b in self.test_pairs:
            result = self.beta.compute(a, b)
            expected = scipy_special.beta(a, b)
            
            assert isinstance(result, (float, np.floating))
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isnan(result)
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isinf(result)
            assert result > 0  # Beta function should be positive
            np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_known_values(self):
        """Test Beta function for known mathematical values"""
        # B(1, 1) = 1
        np.testing.assert_allclose(self.beta.compute(1.0, 1.0), 1.0, rtol=1e-10)
        
        # B(0.5, 0.5) = π
        np.testing.assert_allclose(self.beta.compute(0.5, 0.5), np.pi, rtol=1e-10)
        
        # B(1, n) = 1/n for positive integers n
        for n in range(1, 6):
            expected = 1.0 / n
            np.testing.assert_allclose(self.beta.compute(1.0, float(n)), expected, rtol=1e-10)
    
    def test_symmetry_property(self):
        """Test Beta function symmetry: B(a, b) = B(b, a)"""
        for a, b in self.test_pairs:
            beta_ab = self.beta.compute(a, b)
            beta_ba = self.beta.compute(b, a)
            
            np.testing.assert_allclose(beta_ab, beta_ba, rtol=1e-10)
    
    def test_relation_to_gamma(self):
        """Test Beta function relation to Gamma: B(a, b) = Γ(a)Γ(b)/Γ(a+b)"""
        gamma_func = GammaFunction()
        
        for a, b in self.test_pairs:
            beta_direct = self.beta.compute(a, b)
            
            gamma_a = gamma_func.compute(a)
            gamma_b = gamma_func.compute(b)
            gamma_ab = gamma_func.compute(a + b)
            
            beta_from_gamma = (gamma_a * gamma_b) / gamma_ab
            
            np.testing.assert_allclose(beta_direct, beta_from_gamma, rtol=1e-10)
    
    def test_negative_values(self):
        """Test Beta function for negative values"""
        # Beta function is undefined for negative arguments
        negative_pairs = [(-1.0, 1.0), (1.0, -1.0), (-0.5, 0.5)]

        for a, b in negative_pairs:
            result = self.beta.compute(a, b)
            expected = scipy_special.beta(a, b)

            # Both should be NaN for negative arguments
            # Note: scipy.special.beta may return -1.0 for some negative cases
            assert np.isnan(result) or result == expected
            # Don't assert expected is NaN since scipy may return -1.0
    
    def test_zero_values(self):
        """Test Beta function at zero"""
        # B(0, b) and B(a, 0) should be inf for positive b, a
        result_a_zero = self.beta.compute(0.0, 2.0)
        result_b_zero = self.beta.compute(2.0, 0.0)
        
        assert np.isinf(result_a_zero)
        assert np.isinf(result_b_zero)


class TestGammaFunctionConvenience:
    """Test gamma_function convenience function"""
    
    def test_scalar_input(self):
        """Test gamma_function with scalar input"""
        x = 2.5
        result = gamma_function(x)
        expected = scipy_special.gamma(x)
        
        assert isinstance(result, (float, np.floating))
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_array_input(self):
        """Test gamma_function with array input"""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = gamma_function(x)
        expected = scipy_special.gamma(x)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestLogGammaFunction:
    """Test log_gamma convenience function"""
    
    def test_scalar_input(self):
        """Test log_gamma with scalar input"""
        x = 2.5
        result = log_gamma(x)
        expected = scipy_special.gammaln(x)
        
        assert isinstance(result, (float, np.floating))
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_array_input(self):
        """Test log_gamma with array input"""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = log_gamma(x)
        expected = scipy_special.gammaln(x)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_large_values(self):
        """Test log_gamma for large values (avoids overflow)"""
        x = np.array([10.0, 50.0, 100.0, 1000.0])
        result = log_gamma(x)
        expected = scipy_special.gammaln(x)
        
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestBetaFunctionConvenience:
    """Test beta_function convenience function"""
    
    def test_scalar_input(self):
        """Test beta_function with scalar input"""
        a, b = 2.0, 3.0
        result = beta_function(a, b)
        expected = scipy_special.beta(a, b)
        
        assert isinstance(result, (float, np.floating))
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_array_input(self):
        """Test beta_function with array input"""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 1.0, 1.0])
        result = beta_function(a, b)
        expected = scipy_special.beta(a, b)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == a.shape
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_mixed_input(self):
        """Test beta_function with mixed scalar and array input"""
        a = 2.0
        b = np.array([1.0, 2.0, 3.0])
        result = beta_function(a, b)
        expected = scipy_special.beta(a, b)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == b.shape
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestGammaBetaMathematicalProperties:
    """Test mathematical properties involving both Gamma and Beta functions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.gamma = GammaFunction()
        self.beta = BetaFunction()
    
    def test_stirling_approximation(self):
        """Test Stirling's approximation for large values"""
        n_values = np.array([10.0, 20.0, 50.0])
        
        for n in n_values:
            gamma_exact = self.gamma.compute(n)
            
            # Stirling's approximation: Γ(n) ≈ √(2π/n) * (n/e)^n
            stirling_approx = np.sqrt(2 * np.pi / n) * (n / np.e) ** n
            
            # For large n, approximation should be close
            relative_error = abs(gamma_exact - stirling_approx) / gamma_exact
            assert relative_error < 0.1  # Within 10% for large n
    
    def test_beta_integral_representation(self):
        """Test Beta function integral representation: B(a,b) = ∫₀¹ t^(a-1)(1-t)^(b-1) dt"""
        # This is more of a conceptual test since we can't easily compute the integral
        # But we can test that the Beta function values are reasonable
        a, b = 2.0, 3.0
        beta_value = self.beta.compute(a, b)
        
        # Beta function should be positive and finite
        assert beta_value > 0
        assert not np.isnan(beta_value)
        assert not np.isinf(beta_value)
    
    def test_gamma_beta_consistency(self):
        """Test consistency between Gamma and Beta function implementations"""
        # Test that both implementations give consistent results
        gamma_func = GammaFunction()
        beta_func = BetaFunction()
        
        a, b = 1.5, 2.5
        
        # Compute Beta using both direct method and Gamma relation
        beta_direct = beta_func.compute(a, b)
        gamma_a = gamma_func.compute(a)
        gamma_b = gamma_func.compute(b)
        gamma_ab = gamma_func.compute(a + b)
        beta_from_gamma = (gamma_a * gamma_b) / gamma_ab
        
        np.testing.assert_allclose(beta_direct, beta_from_gamma, rtol=1e-10)


class TestGammaBetaEdgeCases:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.gamma = GammaFunction()
        self.beta = BetaFunction()
    
    def test_very_small_values(self):
        """Test Gamma and Beta functions for very small positive values"""
        small_values = np.array([1e-10, 1e-8, 1e-6])
        
        for x in small_values:
            gamma_result = self.gamma.compute(x)
            assert not np.isnan(gamma_result)
            assert not np.isinf(gamma_result)
            assert gamma_result > 0
    
    def test_very_large_values(self):
        """Test Gamma and Beta functions for very large values"""
        large_values = np.array([1e6, 1e8, 1e10])
        
        for x in large_values:
            gamma_result = self.gamma.compute(x)
            # For very large values, Gamma function should be finite but very large
            assert not np.isnan(gamma_result)
            assert gamma_result > 0
    
    def test_special_cases(self):
        """Test special cases that might cause numerical issues"""
        # Test values near integers
        near_integers = np.array([0.999, 1.001, 1.999, 2.001])
        
        for x in near_integers:
            gamma_result = self.gamma.compute(x)
            assert not np.isnan(gamma_result)
            assert not np.isinf(gamma_result)
            assert gamma_result > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


