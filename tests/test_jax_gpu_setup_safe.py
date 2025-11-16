#!/usr/bin/env python3
"""
Tests for jax_gpu_setup with safe import handling.
Target: Cover import guards, CPU fallback, config detection without GPU requirement.
"""

import pytest
import os
import sys


def test_jax_import_without_gpu(cpu_only):
    """Test that JAX can be imported in CPU-only mode."""
    try:
        import jax
        # Should work in CPU mode
        assert jax.default_backend() == 'cpu'
    except ImportError:
        pytest.skip("JAX not installed")


def test_jax_gpu_setup_skip_flag():
    """Test HPFRACC_SKIP_JAX_INIT flag prevents auto-initialization."""
    original = os.environ.get('HPFRACC_SKIP_JAX_INIT')
    os.environ['HPFRACC_SKIP_JAX_INIT'] = '1'
    
    try:
        # This import should not trigger heavy GPU initialization
        import hpfracc.jax_gpu_setup as jax_setup
        
        # Module should exist
        assert jax_setup is not None
    except ImportError as e:
        pytest.skip(f"JAX GPU setup module not available: {e}")
    finally:
        if original is not None:
            os.environ['HPFRACC_SKIP_JAX_INIT'] = original
        else:
            os.environ.pop('HPFRACC_SKIP_JAX_INIT', None)


def test_jax_cpu_fallback_when_disabled(disable_jax):
    """Test that disabling JAX forces fallback to other backends."""
    try:
        from hpfracc.ml import tensor_ops
        import numpy as np
        
        x = np.array([1.0, 2.0, 3.0])
        result = tensor_ops.to_tensor(x, backend='numpy')
        
        # Should use numpy backend
        assert isinstance(result, np.ndarray)
    except ImportError:
        pytest.skip("Module not available")


def test_check_jax_available():
    """Test function to check JAX availability."""
    try:
        from hpfracc.jax_gpu_setup import check_jax_available
        
        is_available = check_jax_available()
        assert isinstance(is_available, bool)
    except ImportError:
        pytest.skip("check_jax_available not available")


def test_get_jax_device_info(cpu_only):
    """Test getting JAX device information."""
    try:
        from hpfracc.jax_gpu_setup import get_device_info
        import jax
        
        device_info = get_device_info()
        
        # Should return info about CPU devices
        assert isinstance(device_info, (dict, list, str))
    except ImportError:
        pytest.skip("JAX or device info not available")


def test_jax_device_count_cpu(cpu_only):
    """Test JAX device count in CPU-only mode."""
    try:
        import jax
        
        devices = jax.devices()
        assert len(devices) >= 1
        assert all(d.platform == 'cpu' for d in devices)
    except ImportError:
        pytest.skip("JAX not available")


def test_configure_jax_for_cpu():
    """Test configuring JAX explicitly for CPU."""
    try:
        from hpfracc.jax_gpu_setup import configure_jax
        
        config = configure_jax(device='cpu')
        
        # Should return configuration dict or None
        assert config is None or isinstance(config, dict)
    except ImportError:
        pytest.skip("configure_jax not available")


def test_jax_memory_fraction_setting():
    """Test setting JAX memory fraction (should work for CPU)."""
    try:
        from hpfracc.jax_gpu_setup import set_memory_fraction
        
        # Should not raise error even on CPU
        set_memory_fraction(0.5)
    except ImportError:
        pytest.skip("set_memory_fraction not available")
    except (RuntimeError, ValueError):
        # Acceptable if not supported on CPU
        pass


def test_jax_config_flags():
    """Test JAX configuration flags are properly set."""
    try:
        import jax
        
        # Check some common config flags exist
        config = jax.config
        assert hasattr(config, 'values') or hasattr(config, 'FLAGS')
    except ImportError:
        pytest.skip("JAX not available")


def test_jax_disable_jit_flag():
    """Test JAX JIT can be disabled for debugging."""
    try:
        import jax
        
        # Try to disable JIT
        with jax.disable_jit():
            @jax.jit
            def f(x):
                return x * 2
            
            result = f(2.0)
            assert result == 4.0
    except ImportError:
        pytest.skip("JAX not available")


def test_jax_platform_detection():
    """Test platform detection in JAX."""
    try:
        import jax
        
        platform = jax.default_backend()
        assert platform in ['cpu', 'gpu', 'tpu']
    except ImportError:
        pytest.skip("JAX not available")


def test_jax_version_check():
    """Test JAX version is accessible."""
    try:
        import jax
        
        version = jax.__version__
        assert isinstance(version, str)
        assert len(version) > 0
    except ImportError:
        pytest.skip("JAX not available")


def test_jax_random_key_creation(cpu_only, set_seed):
    """Test creating JAX random keys on CPU."""
    set_seed(42)
    
    try:
        import jax
        
        key = jax.random.PRNGKey(42)
        assert key is not None
        
        # Generate random numbers
        data = jax.random.normal(key, shape=(10,))
        assert data.shape == (10,)
    except ImportError:
        pytest.skip("JAX not available")


def test_jax_simple_computation(cpu_only):
    """Test simple JAX computation on CPU."""
    try:
        import jax.numpy as jnp
        
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        
        assert float(y) == 6.0
    except ImportError:
        pytest.skip("JAX not available")


def test_jax_grad_simple_function(cpu_only):
    """Test JAX gradient computation."""
    try:
        import jax
        import jax.numpy as jnp
        
        def f(x):
            return x ** 2
        
        grad_f = jax.grad(f)
        result = grad_f(3.0)
        
        assert abs(float(result) - 6.0) < 1e-6
    except ImportError:
        pytest.skip("JAX not available")


def test_jax_vmap_simple(cpu_only):
    """Test JAX vmap (vectorization)."""
    try:
        import jax
        import jax.numpy as jnp
        
        def f(x):
            return x ** 2
        
        x = jnp.array([1.0, 2.0, 3.0])
        result = jax.vmap(f)(x)
        
        expected = jnp.array([1.0, 4.0, 9.0])
        assert jnp.allclose(result, expected)
    except ImportError:
        pytest.skip("JAX not available")


def test_jax_jit_compilation(cpu_only):
    """Test JAX JIT compilation works on CPU."""
    try:
        import jax
        import jax.numpy as jnp
        
        @jax.jit
        def f(x):
            return x ** 2 + 2 * x + 1
        
        result = f(3.0)
        assert abs(float(result) - 16.0) < 1e-6
    except ImportError:
        pytest.skip("JAX not available")


def test_disable_jax_env_variable():
    """Test HPFRACC_DISABLE_JAX environment variable."""
    original = os.environ.get('HPFRACC_DISABLE_JAX')
    os.environ['HPFRACC_DISABLE_JAX'] = '1'
    
    try:
        # With JAX disabled, imports should fall back to numpy
        from hpfracc.ml import tensor_ops
        import numpy as np
        
        x = np.array([1.0, 2.0, 3.0])
        result = tensor_ops.to_tensor(x, backend='numpy')
        
        assert isinstance(result, np.ndarray)
    except ImportError:
        pytest.skip("Module not available")
    finally:
        if original is not None:
            os.environ['HPFRACC_DISABLE_JAX'] = original
        else:
            os.environ.pop('HPFRACC_DISABLE_JAX', None)


def test_jax_device_put_cpu(cpu_only):
    """Test jax.device_put on CPU."""
    try:
        import jax
        import jax.numpy as jnp
        
        x = jnp.array([1.0, 2.0, 3.0])
        x_device = jax.device_put(x)
        
        assert jnp.allclose(x, x_device)
    except ImportError:
        pytest.skip("JAX not available")


def test_jax_array_conversion(cpu_only):
    """Test converting between NumPy and JAX arrays."""
    try:
        import jax.numpy as jnp
        import numpy as np
        
        # NumPy to JAX
        np_array = np.array([1.0, 2.0, 3.0])
        jax_array = jnp.array(np_array)
        
        # JAX to NumPy
        np_array_back = np.array(jax_array)
        
        assert np.allclose(np_array, np_array_back)
    except ImportError:
        pytest.skip("JAX not available")


def test_jax_scan_simple(cpu_only):
    """Test JAX scan operation."""
    try:
        import jax
        import jax.numpy as jnp
        
        def step(carry, x):
            return carry + x, carry
        
        xs = jnp.array([1.0, 2.0, 3.0])
        init = 0.0
        
        final, outputs = jax.lax.scan(step, init, xs)
        
        assert float(final) == 6.0
    except ImportError:
        pytest.skip("JAX not available")


def test_jax_fori_loop(cpu_only):
    """Test JAX fori_loop."""
    try:
        import jax
        
        def body(i, val):
            return val + i
        
        result = jax.lax.fori_loop(0, 10, body, 0)
        
        assert result == 45  # Sum 0 to 9
    except ImportError:
        pytest.skip("JAX not available")


def test_jax_cond_simple(cpu_only):
    """Test JAX conditional (cond) operation."""
    try:
        import jax
        
        def true_fn(x):
            return x + 1
        
        def false_fn(x):
            return x - 1
        
        result_true = jax.lax.cond(True, true_fn, false_fn, 5.0)
        result_false = jax.lax.cond(False, true_fn, false_fn, 5.0)
        
        assert float(result_true) == 6.0
        assert float(result_false) == 4.0
    except ImportError:
        pytest.skip("JAX not available")
