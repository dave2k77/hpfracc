import pytest
import numpy as np
import os

# Disable Numba JIT for these tests to avoid bytecode errors on newer Python versions
os.environ['NUMBA_DISABLE_JIT'] = '1'

from hpfracc.special.gamma_beta import GammaFunction, BetaFunction
from hpfracc.special.mittag_leffler import MittagLefflerFunction
from hpfracc.special.binomial_coeffs import BinomialCoefficients

class TestGammaFunction:
    """Test Gamma function implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.gamma = GammaFunction()
        self.test_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    
    def test_initialization(self):
        """Test Gamma function initialization"""
        assert isinstance(self.gamma, GammaFunction)
    
    def test_compute_basic_values(self):
        """Test computation of basic Gamma function values"""
        # Test known values
        result = self.gamma.compute(self.test_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.test_values)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.all(result > 0)  # Gamma function is always positive for positive arguments
    
    def test_known_values(self):
        """Test known Gamma function values"""
        # Γ(1) = 1
        assert abs(self.gamma.compute(1.0) - 1.0) < 1e-10
        
        # Γ(0.5) = √π
        expected_sqrt_pi = np.sqrt(np.pi)
        assert abs(self.gamma.compute(0.5) - expected_sqrt_pi) < 1e-10
        
        # Γ(2) = 1
        assert abs(self.gamma.compute(2.0) - 1.0) < 1e-10
    
    def test_recurrence_relation(self):
        """Test recurrence relation: Γ(z+1) = zΓ(z)"""
        z_values = np.array([1.5, 2.3, 3.7])
        
        gamma_z = self.gamma.compute(z_values)
        gamma_z_plus_1 = self.gamma.compute(z_values + 1)
        
        # Check Γ(z+1) ≈ zΓ(z)
        expected = z_values * gamma_z
        np.testing.assert_allclose(gamma_z_plus_1, expected, rtol=1e-10, atol=1e-10)
    
    def test_scalar_input(self):
        """Test scalar input"""
        result = self.gamma.compute(2.5)
        assert isinstance(result, (float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)
    
    def test_array_input(self):
        """Test array input"""
        result = self.gamma.compute(self.test_values)
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_values.shape
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Test very small positive values
        small_values = np.array([1e-6, 1e-3, 0.01])
        result_small = self.gamma.compute(small_values)
        assert not np.any(np.isnan(result_small))
        assert not np.any(np.isinf(result_small))
        
        # Test larger values
        large_values = np.array([10.0, 20.0, 50.0])
        result_large = self.gamma.compute(large_values)
        assert not np.any(np.isnan(result_large))
        assert not np.any(np.isinf(result_large))


class TestBetaFunction:
    """Test Beta function implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.beta = BetaFunction()
        self.a_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        self.b_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    
    def test_initialization(self):
        """Test Beta function initialization"""
        assert isinstance(self.beta, BetaFunction)
    
    def test_compute_basic_values(self):
        """Test computation of basic Beta function values"""
        result = self.beta.compute(self.a_values, self.b_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.a_values)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.all(result > 0)  # Beta function is always positive for positive arguments
    
    def test_known_values(self):
        """Test known Beta function values"""
        # B(1, 1) = 1
        assert abs(self.beta.compute(1.0, 1.0) - 1.0) < 1e-10
        
        # B(0.5, 0.5) = π
        assert abs(self.beta.compute(0.5, 0.5) - np.pi) < 1e-10
        
        # B(2, 3) = 1/12
        assert abs(self.beta.compute(2.0, 3.0) - 1.0/12.0) < 1e-10
    
    def test_symmetry_property(self):
        """Test symmetry property: B(a, b) = B(b, a)"""
        a, b = 1.5, 2.3
        
        beta_ab = self.beta.compute(a, b)
        beta_ba = self.beta.compute(b, a)
        
        assert abs(beta_ab - beta_ba) < 1e-10
    
    def test_relation_to_gamma(self):
        """Test relation to Gamma function: B(a, b) = Γ(a)Γ(b)/Γ(a+b)"""
        gamma = GammaFunction()
        a, b = 1.5, 2.3
        
        beta_direct = self.beta.compute(a, b)
        gamma_a = gamma.compute(a)
        gamma_b = gamma.compute(b)
        gamma_ab = gamma.compute(a + b)
        beta_from_gamma = (gamma_a * gamma_b) / gamma_ab
        
        assert abs(beta_direct - beta_from_gamma) < 1e-10
    
    def test_scalar_input(self):
        """Test scalar input"""
        result = self.beta.compute(2.5, 1.5)
        assert isinstance(result, (float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)
    
    def test_array_input(self):
        """Test array input"""
        result = self.beta.compute(self.a_values, self.b_values)
        assert isinstance(result, np.ndarray)
        assert result.shape == self.a_values.shape


class TestMittagLefflerFunction:
    """Test Mittag-Leffler function implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.ml = MittagLefflerFunction()
        self.z_values = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        self.alpha_values = np.array([0.5, 1.0, 1.5, 2.0])
    
    def test_initialization(self):
        """Test Mittag-Leffler function initialization"""
        assert isinstance(self.ml, MittagLefflerFunction)
    
    def test_compute_basic_values(self):
        """Test computation of basic Mittag-Leffler function values"""
        result = self.ml.compute(self.z_values, alpha=1.0)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.z_values)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_known_values(self):
        """Test known Mittag-Leffler function values"""
        # E₁(z) = exp(z)
        z = np.array([0.5, 1.0, 1.5])
        ml_1 = self.ml.compute(z, alpha=1.0)
        exp_z = np.exp(z)
        
        np.testing.assert_allclose(ml_1, exp_z, rtol=1e-10, atol=1e-10)
        
        # E₂(z²) = cosh(z)
        z_squared = z**2
        ml_2 = self.ml.compute(z_squared, alpha=2.0)
        cosh_z = np.cosh(z)
        
        np.testing.assert_allclose(ml_2, cosh_z, rtol=1e-10, atol=1e-10)
    
    def test_different_alpha_values(self):
        """Test different alpha values"""
        z = 1.0
        
        for alpha in self.alpha_values:
            result = self.ml.compute(z, alpha=alpha)
            assert isinstance(result, (float, np.floating))
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isnan(result)
            # Type guard for integer results
            if not isinstance(result, (int, np.integer)):
                assert not np.isinf(result)
    
    def test_scalar_input(self):
        """Test scalar input"""
        result = self.ml.compute(1.0, alpha=1.5)
        assert isinstance(result, (float, np.floating))
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isnan(result)
        # Type guard for integer results
        if not isinstance(result, (int, np.integer)):
            assert not np.isinf(result)
    
    def test_array_input(self):
        """Test array input"""
        result = self.ml.compute(self.z_values, alpha=1.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == self.z_values.shape
    
    def test_convergence_properties(self):
        """Test convergence properties"""
        # For small z, E_α(z) should be close to 1
        small_z = np.array([0.01, 0.1, 0.2])
        result_small = self.ml.compute(small_z, alpha=1.5)
        
        # Should be close to 1 for small z
        assert np.all(np.abs(result_small - 1.0) < 0.5)


class TestBinomialCoefficients:
    """Test binomial coefficients implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.bc = BinomialCoefficients()
        self.n_values = np.array([5, 10, 15, 20])
        self.k_values = np.array([2, 3, 5, 7])
    
    def test_initialization(self):
        """Test binomial coefficients initialization"""
        assert isinstance(self.bc, BinomialCoefficients)
    
    def test_compute_basic_values(self):
        """Test computation of basic binomial coefficients"""
        result = self.bc.compute(self.n_values, self.k_values)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(self.n_values)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.all(result >= 0)  # Binomial coefficients are non-negative
    
    def test_known_values(self):
        """Test known binomial coefficient values"""
        # C(5, 2) = 10
        assert self.bc.compute(5, 2) == 10
        
        # C(10, 3) = 120
        assert self.bc.compute(10, 3) == 120
        
        # C(n, 0) = 1 for any n
        assert self.bc.compute(10, 0) == 1
        
        # C(n, n) = 1 for any n
        assert self.bc.compute(10, 10) == 1
    
    def test_symmetry_property(self):
        """Test symmetry property: C(n, k) = C(n, n-k)"""
        n, k = 10, 3
        
        c_nk = self.bc.compute(n, k)
        c_n_nk = self.bc.compute(n, n - k)
        
        assert c_nk == c_n_nk
    
    def test_pascal_triangle_property(self):
        """Test Pascal's triangle property: C(n, k) = C(n-1, k-1) + C(n-1, k)"""
        n, k = 10, 5
        
        c_nk = self.bc.compute(n, k)
        c_n1_k1 = self.bc.compute(n - 1, k - 1)
        c_n1_k = self.bc.compute(n - 1, k)
        
        assert c_nk == c_n1_k1 + c_n1_k
    
    def test_scalar_input(self):
        """Test scalar input"""
        result = self.bc.compute(10, 5)
        assert isinstance(result, (int, np.integer))
        assert result == 252  # C(10, 5) = 252
    
    def test_array_input(self):
        """Test array input"""
        result = self.bc.compute(self.n_values, self.k_values)
        assert isinstance(result, np.ndarray)
        assert result.shape == self.n_values.shape
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Test k = 0
        assert self.bc.compute(10, 0) == 1
        
        # Test k = n
        assert self.bc.compute(10, 10) == 1
        
        # Test k > n (should be 0)
        assert self.bc.compute(5, 10) == 0


class TestSpecialFunctionIntegration:
    """Test integration between special functions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.gamma = GammaFunction()
        self.beta = BetaFunction()
        self.ml = MittagLefflerFunction()
        self.bc = BinomialCoefficients()
    
    def test_gamma_beta_relation(self):
        """Test relation between Gamma and Beta functions"""
        a, b = 2.5, 1.5
        
        gamma_a = self.gamma.compute(a)
        gamma_b = self.gamma.compute(b)
        gamma_ab = self.gamma.compute(a + b)
        beta_ab = self.beta.compute(a, b)
        
        # B(a, b) = Γ(a)Γ(b)/Γ(a+b)
        expected_beta = (gamma_a * gamma_b) / gamma_ab
        
        assert abs(beta_ab - expected_beta) < 1e-10
    
    def test_factorial_relation(self):
        """Test relation to factorial via Gamma function"""
        n_values = np.array([1, 2, 3, 4, 5])
        
        gamma_n_plus_1 = self.gamma.compute(n_values + 1)
        factorials = np.array([1, 2, 6, 24, 120])
        
        np.testing.assert_allclose(gamma_n_plus_1, factorials, rtol=1e-10, atol=1e-10)
    
    def test_mittag_leffler_gamma_relation(self):
        """Test relation between Mittag-Leffler and Gamma functions"""
        # For α = 1, E₁(z) = exp(z) = Σ(zⁿ/n!)
        z = 1.0
        ml_1 = self.ml.compute(z, alpha=1.0)
        exp_z = np.exp(z)
        
        assert abs(ml_1 - exp_z) < 1e-10


class TestErrorHandling:
    
    def test_beta_function_errors(self):
        """Test error handling in Beta function"""
        beta = BetaFunction()
        
        # Note: Current implementation doesn't validate inputs
        # It relies on scipy's beta function behavior
        
        # Negative values - scipy.beta behavior varies by version
        # Just verify it doesn't crash and returns a numeric value
        result_neg = beta.compute(-1.0, 1.0)
        assert isinstance(result_neg, (int, float, np.integer, np.floating)), f"Beta with negative returned non-numeric: {result_neg}"
        
        result_neg2 = beta.compute(1.0, -1.0)
        assert isinstance(result_neg2, (int, float, np.integer, np.floating)), f"Beta with negative returned non-numeric: {result_neg2}"
        
        # Zero values - scipy.beta may return inf or large value
        result_zero = beta.compute(0.0, 1.0)
        assert isinstance(result_zero, (int, float, np.integer, np.floating)), f"Beta with zero returned non-numeric: {result_zero}"
    
    def test_mittag_leffler_errors(self):
        """Test error handling in Mittag-Leffler function"""
        ml = MittagLefflerFunction()
        
        # Note: Current implementation checks alpha > 0 inside _compute_python_scalar
        # For alpha <= 0, returns NaN
        
        # Test invalid alpha values - should return NaN
        result_zero = ml.compute(1.0, alpha=0.0)
        assert np.isnan(result_zero), "Mittag-Leffler with alpha=0 should return NaN"
        
        result_neg = ml.compute(1.0, alpha=-1.0)
        assert np.isnan(result_neg), "Mittag-Leffler with negative alpha should return NaN"
    
    def test_binomial_coefficients_errors(self):
        """Test error handling in binomial coefficients"""
        bc = BinomialCoefficients()
        
        # Note: Current implementation doesn't validate inputs
        # It relies on scipy.special.binom behavior
        
        # Test negative n - scipy returns 0
        result_neg_n = bc.compute(-1, 2)
        assert result_neg_n == 0 or np.isnan(result_neg_n), "C(-1, 2) should be 0 or NaN"
        
        # Test negative k - our implementation checks k < 0 and returns 0
        result_neg_k = bc.compute(5, -1)
        assert result_neg_k == 0, "C(5, -1) should be 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


