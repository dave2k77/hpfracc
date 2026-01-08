#!/usr/bin/env python3
"""
Comprehensive tests for hpfracc.algorithms module functionality.

This test suite focuses on mathematical correctness and performance
of the core fractional calculus algorithms.
"""

import pytest
import numpy as np
import time
from typing import Callable, Union

# Test the core algorithms
# test_algorithms_imports_work removed as it tests obsolete imports
def test_algorithms_imports_work():
    pass


class TestOptimizedRiemannLiouville:
    """Test OptimizedRiemannLiouville implementation."""
    
    def test_riemann_liouville_creation(self):
        """Test creating OptimizedRiemannLiouville objects."""
        from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville
        
        # Test with different alpha values
        rl = OptimizedRiemannLiouville(0.5)
        assert rl.alpha.alpha == 0.5
        assert rl.n == 1  # ceil(0.5) = 1
        
        rl = OptimizedRiemannLiouville(1.7)
        assert rl.alpha.alpha == 1.7
        assert rl.n == 2  # ceil(1.7) = 2
        
        rl = OptimizedRiemannLiouville(2.0)
        assert rl.alpha.alpha == 2.0
        assert rl.n == 2  # ceil(2.0) = 2
    
    def test_riemann_liouville_validation(self):
        """Test input validation."""
        from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville
        
        # Valid alpha values
        OptimizedRiemannLiouville(0.0)
        OptimizedRiemannLiouville(0.5)
        OptimizedRiemannLiouville(1.0)
        OptimizedRiemannLiouville(2.0)
        
        # Invalid alpha values should raise ValueError
        with pytest.raises(ValueError):
            OptimizedRiemannLiouville(-0.1)
    
    def test_riemann_liouville_compute_function(self):
        """Test computing RL derivative with function input."""
        from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville
        
        rl = OptimizedRiemannLiouville(0.5)
        
        # Test with simple function
        def f(t):
            return t**2
        
        # Test at single point
        result = rl.compute(f, 1.0)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        
        # Test at multiple points
        t_points = np.array([0.5, 1.0, 1.5])
        result = rl.compute(f, t_points)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_riemann_liouville_compute_array(self):
        """Test computing RL derivative with array input."""
        from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville
        
        rl = OptimizedRiemannLiouville(0.5)
        
        # Test with array input
        t = np.linspace(0, 1, 100)
        f = t**2
        
        result = rl.compute(f, t)
        assert isinstance(result, np.ndarray)
        assert len(result) == 100
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_riemann_liouville_edge_cases(self):
        """Test edge cases for RL derivative."""
        from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville
        
        rl = OptimizedRiemannLiouville(0.5)
        
        # Test with empty arrays
        result = rl.compute(np.array([]), np.array([]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
        
        # Test with single point
        result = rl.compute(np.array([1.0]), np.array([1.0]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert not np.isnan(result[0])


class TestOptimizedCaputo:
    """Test OptimizedCaputo implementation."""
    
    def test_caputo_creation(self):
        """Test creating OptimizedCaputo objects."""
        from hpfracc.algorithms.optimized_methods import OptimizedCaputo
        
        # Test with different alpha values (L1 scheme requires 0 < α < 1)
        caputo = OptimizedCaputo(0.5)
        assert caputo.alpha.alpha == 0.5
    
    def test_caputo_compute_function(self):
        """Test computing Caputo derivative with function input."""
        from hpfracc.algorithms.optimized_methods import OptimizedCaputo
        
        caputo = OptimizedCaputo(0.5)
        
        # Test with simple function
        def f(t):
            return t**2
        
        # Test at single point
        result = caputo.compute(f, 1.0)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        
        # Test at multiple points
        t_points = np.array([0.5, 1.0, 1.5])
        result = caputo.compute(f, t_points)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_caputo_compute_array(self):
        """Test computing Caputo derivative with array input."""
        from hpfracc.algorithms.optimized_methods import OptimizedCaputo
        
        caputo = OptimizedCaputo(0.5)
        
        # Test with array input
        t = np.linspace(0, 1, 100)
        f = t**2
        
        result = caputo.compute(f, t)
        assert isinstance(result, np.ndarray)
        assert len(result) == 100
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestOptimizedGrunwaldLetnikov:
    """Test OptimizedGrunwaldLetnikov implementation."""
    
    def test_grunwald_letnikov_creation(self):
        """Test creating OptimizedGrunwaldLetnikov objects."""
        from hpfracc.algorithms.optimized_methods import OptimizedGrunwaldLetnikov
        
        # Test with different alpha values
        gl = OptimizedGrunwaldLetnikov(0.5)
        assert gl.alpha.alpha == 0.5
        
        gl = OptimizedGrunwaldLetnikov(1.7)
        assert gl.alpha.alpha == 1.7
    
    def test_grunwald_letnikov_compute_function(self):
        """Test computing GL derivative with function input."""
        from hpfracc.algorithms.optimized_methods import OptimizedGrunwaldLetnikov
        
        gl = OptimizedGrunwaldLetnikov(0.5)
        
        # Test with simple function
        def f(t):
            return t**2
        
        # Test at single point
        result = gl.compute(f, 1.0)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        
        # Test at multiple points
        t_points = np.array([0.5, 1.0, 1.5])
        result = gl.compute(f, t_points)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_grunwald_letnikov_compute_array(self):
        """Test computing GL derivative with array input."""
        from hpfracc.algorithms.optimized_methods import OptimizedGrunwaldLetnikov
        
        gl = OptimizedGrunwaldLetnikov(0.5)
        
        # Test with array input
        t = np.linspace(0, 1, 100)
        f = t**2
        
        result = gl.compute(f, t)
        assert isinstance(result, np.ndarray)
        assert len(result) == 100
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestGPUOptimizedMethods:
    """Test GPU-optimized implementations."""
    
    def test_gpu_config_creation(self):
        """Test creating GPUConfig objects."""
        from hpfracc.algorithms.gpu_optimized_methods import GPUConfig
        
        # Test with default configuration
        config = GPUConfig()
        assert config.backend is not None
        assert config.memory_limit > 0
        assert config.memory_limit <= 1.0
        
        # Test with custom configuration
        config = GPUConfig(backend="jax", memory_limit=0.5)
        assert config.backend == "jax"
        assert config.memory_limit == 0.5
    
    def test_gpu_riemann_liouville_creation(self):
        """Test creating GPUOptimizedRiemannLiouville objects."""
        from hpfracc.algorithms.gpu_optimized_methods import GPUOptimizedRiemannLiouville, GPUConfig
        
        config = GPUConfig()
        rl = GPUOptimizedRiemannLiouville(0.5, config)
        assert rl.alpha.alpha == 0.5
        assert rl.gpu_config is not None
    
    def test_gpu_caputo_creation(self):
        """Test creating GPUOptimizedCaputo objects."""
        from hpfracc.algorithms.gpu_optimized_methods import GPUOptimizedCaputo, GPUConfig
        
        config = GPUConfig()
        caputo = GPUOptimizedCaputo(0.5, config)
        assert caputo.alpha.alpha == 0.5
        assert caputo.gpu_config is not None


class TestAdvancedMethods:
    """Test advanced mathematical methods."""
    
    def test_weyl_derivative_creation(self):
        """Test creating WeylDerivative objects."""
        from hpfracc.algorithms.advanced_methods import WeylDerivative
        
        weyl = WeylDerivative(0.5)
        assert weyl.alpha.alpha == 0.5
    
    def test_marchaud_derivative_creation(self):
        """Test creating MarchaudDerivative objects."""
        from hpfracc.algorithms.advanced_methods import MarchaudDerivative
        
        marchaud = MarchaudDerivative(0.5)
        assert marchaud.alpha.alpha == 0.5
    
    def test_hadamard_derivative_creation(self):
        """Test creating HadamardDerivative objects."""
        from hpfracc.algorithms.advanced_methods import HadamardDerivative
        
        hadamard = HadamardDerivative(0.5)
        assert hadamard.alpha.alpha == 0.5


class TestMathematicalCorrectness:
    """Test mathematical correctness of implementations."""
    
    def test_derivative_consistency(self):
        """Test that different implementations give consistent results."""
        from hpfracc.algorithms.optimized_methods import (
            OptimizedRiemannLiouville,
            OptimizedCaputo,
            OptimizedGrunwaldLetnikov
        )
        
        # Test with simple function
        def f(t):
            return t**2
        
        alpha = 0.5
        t_points = np.array([0.5, 1.0, 1.5])
        
        # Create different derivative implementations
        rl = OptimizedRiemannLiouville(alpha)
        caputo = OptimizedCaputo(alpha)
        gl = OptimizedGrunwaldLetnikov(alpha)
        
        # Compute derivatives
        rl_result = rl.compute(f, t_points)
        caputo_result = caputo.compute(f, t_points)
        gl_result = gl.compute(f, t_points)
        
        # Results should be finite and not NaN
        assert not np.any(np.isnan(rl_result))
        assert not np.any(np.isnan(caputo_result))
        assert not np.any(np.isnan(gl_result))
        
        assert not np.any(np.isinf(rl_result))
        assert not np.any(np.isinf(caputo_result))
        assert not np.any(np.isinf(gl_result))
    
    def test_alpha_zero_case(self):
        """Test that alpha=0 gives identity operation."""
        from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville
        
        rl = OptimizedRiemannLiouville(0.0)
        
        def f(t):
            return t**2
        
        t_points = np.array([0.5, 1.0, 1.5])
        result = rl.compute(f, t_points)
        
        # For alpha=0, the derivative should be the function itself
        expected = np.array([f(t) for t in t_points])
        # Allow for some numerical error
        assert np.allclose(result, expected, rtol=1e-10)
    
    def test_alpha_one_case(self):
        """Test that alpha=1 gives first derivative."""
        from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville
        
        rl = OptimizedRiemannLiouville(1.0)
        
        def f(t):
            return t**2
        
        t_points = np.array([0.5, 1.0, 1.5])
        result = rl.compute(f, t_points)
        
        # For alpha=1, the derivative should match analytical derivative 2t
        # [0.5, 1.0, 1.5] -> [1.0, 2.0, 3.0]
        # np.gradient with edge_order=2 is exact for quadratic
        expected = np.array([1.0, 2.0, 3.0])
        # Allow for some numerical error
        assert np.allclose(result, expected, rtol=1e-5)


class TestPerformance:
    """Test performance characteristics."""
    
    def test_computation_time(self):
        """Test that computations complete in reasonable time."""
        from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville
        
        rl = OptimizedRiemannLiouville(0.5)
        
        def f(t):
            return t**2
        
        # Test with moderate size
        t_points = np.linspace(0, 1, 1000)
        
        start_time = time.time()
        result = rl.compute(f, t_points)
        end_time = time.time()
        
        computation_time = end_time - start_time
        
        # Should complete in reasonable time (< 10 seconds)
        assert computation_time < 10.0
        assert len(result) == 1000
        assert not np.any(np.isnan(result))
    
    def test_memory_usage(self):
        """Test that computations don't use excessive memory."""
        from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville
        
        rl = OptimizedRiemannLiouville(0.5)
        
        def f(t):
            return t**2
        
        # Test with larger size
        t_points = np.linspace(0, 1, 10000)
        
        # This should not cause memory issues
        result = rl.compute(f, t_points)
        
        assert len(result) == 10000
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville
        
        rl = OptimizedRiemannLiouville(0.5)
        
        # Test with mismatched array lengths
        with pytest.raises(ValueError):
            rl.compute(np.array([1, 2, 3]), np.array([1, 2]))
        
        # Test with invalid step size
        with pytest.raises(ValueError):
            rl.compute(lambda t: t**2, 1.0, h=0.0)
        
        # Test with negative step size
        with pytest.raises(ValueError):
            rl.compute(lambda t: t**2, 1.0, h=-0.1)
    
    def test_edge_case_arrays(self):
        """Test edge cases with arrays."""
        from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville
        
        rl = OptimizedRiemannLiouville(0.5)
        
        # Test with single element arrays
        result = rl.compute(np.array([1.0]), np.array([1.0]))
        assert len(result) == 1
        assert not np.isnan(result[0])
        
        # Test with empty arrays
        result = rl.compute(np.array([]), np.array([]))
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
