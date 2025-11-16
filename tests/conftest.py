#!/usr/bin/env python3
"""
Pytest configuration and common fixtures for fractional calculus library tests.
"""

import pytest
import numpy as np
import sys
import os
import random

# Force JAX to use CPU for tests to avoid GPU/CuDNN version issues
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Force matplotlib to use non-interactive backend for tests
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

# Add hpfracc to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="function")
def reset_backend_state():
    """
    Reset backend manager state before a test to ensure test isolation.
    
    This fixture is available for tests that need it, but is NOT autouse.
    Tests must explicitly request it by including it as a parameter.
    
    Usage:
        def test_something(reset_backend_state):
            # Your test code here
    
    Note: Autouse was tried but caused issues with the full test suite.
    The ordering issues with tensor_ops/losses are likely due to other
    state pollution, not backend manager state.
    """
    # Reset the global backend manager before the test
    try:
        import hpfracc.ml.backends as backends_module
        backends_module._backend_manager = None
    except ImportError:
        pass  # Module not available yet, that's fine
    except Exception:
        pass  # Other import errors are also fine (test may not need ML)
    
    yield
    
    # Clean up after test
    try:
        import hpfracc.ml.backends as backends_module
        backends_module._backend_manager = None
    except ImportError:
        pass
    except Exception:
        pass


@pytest.fixture
def sample_time_array():
    """Provide a sample time array for testing."""
    return np.linspace(0.1, 2.0, 50)


@pytest.fixture
def sample_function_values():
    """Provide sample function values for testing."""
    t = np.linspace(0.1, 2.0, 50)
    return t  # Simple linear function


@pytest.fixture
def sample_quadratic_function():
    """Provide quadratic function values for testing."""
    t = np.linspace(0.1, 2.0, 50)
    return t**2


@pytest.fixture
def sample_exponential_function():
    """Provide exponential function values for testing."""
    t = np.linspace(0.1, 2.0, 50)
    return np.exp(-t)


@pytest.fixture
def sample_trigonometric_function():
    """Provide trigonometric function values for testing."""
    t = np.linspace(0.1, 2.0, 50)
    return np.sin(t)


@pytest.fixture
def fractional_orders():
    """Provide various fractional orders for testing."""
    return [0.25, 0.5, 0.75, 1.0, 1.5]


@pytest.fixture
def step_sizes():
    """Provide various step sizes for testing."""
    return [0.01, 0.05, 0.1]


@pytest.fixture
def grid_sizes():
    """Provide various grid sizes for testing."""
    return [25, 50, 100, 200]


@pytest.fixture
def tolerance():
    """Provide tolerance for numerical comparisons."""
    return 1e-10


@pytest.fixture
def analytical_solutions():
    """Provide analytical solutions for known test cases."""
    from scipy.special import gamma

    def get_caputo_analytical(t, alpha):
        """Analytical solution for Caputo derivative of f(t) = t."""
        return t ** (1 - alpha) / gamma(2 - alpha)

    def get_riemann_liouville_analytical(t, alpha):
        """Analytical solution for Riemann-Liouville derivative of f(t) = t."""
        return t ** (1 - alpha) / gamma(2 - alpha)

    return {
        "caputo_linear": get_caputo_analytical,
        "riemann_liouville_linear": get_riemann_liouville_analytical,
    }


# Week 1 Test Infrastructure Fixtures

@pytest.fixture
def set_seed():
    """Set random seeds for reproducible tests."""
    def _set_seed(seed=1234):
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
    return _set_seed


@pytest.fixture
def force_backend():
    """Force a specific backend for testing."""
    def _force_backend(backend_name):
        """
        Force backend by setting environment variable.
        Args:
            backend_name: 'numpy', 'torch', 'jax'
        """
        os.environ['HPFRACC_BACKEND'] = backend_name.upper()
        # Clear any cached backend manager
        try:
            import hpfracc.ml.backends as backends_module
            backends_module._backend_manager = None
        except ImportError:
            pass
    return _force_backend


@pytest.fixture
def cpu_only():
    """Ensure CPU-only execution for tests."""
    original_jax_platforms = os.environ.get('JAX_PLATFORMS')
    original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    yield
    
    # Restore original values
    if original_jax_platforms is not None:
        os.environ['JAX_PLATFORMS'] = original_jax_platforms
    else:
        os.environ.pop('JAX_PLATFORMS', None)
    
    if original_cuda_visible is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)


@pytest.fixture
def disable_jax():
    """Disable JAX imports for testing fallback paths."""
    original_value = os.environ.get('HPFRACC_DISABLE_JAX')
    os.environ['HPFRACC_DISABLE_JAX'] = '1'
    
    # Clear any cached imports
    try:
        import hpfracc.ml.backends as backends_module
        backends_module._backend_manager = None
    except ImportError:
        pass
    
    yield
    
    # Restore
    if original_value is not None:
        os.environ['HPFRACC_DISABLE_JAX'] = original_value
    else:
        os.environ.pop('HPFRACC_DISABLE_JAX', None)


@pytest.fixture
def tiny_ode_data(set_seed):
    """Provide tiny ODE test data for fast tests."""
    set_seed(1234)
    batch_size = 2
    state_dim = 3
    time_steps = 5
    
    t = np.linspace(0.0, 1.0, time_steps)
    y0 = np.random.randn(batch_size, state_dim).astype(np.float32)
    
    return {
        't': t,
        'y0': y0,
        'batch_size': batch_size,
        'state_dim': state_dim,
        'time_steps': time_steps
    }


@pytest.fixture
def small_grid():
    """Provide small grid for PDE solver tests."""
    return {
        'nx': 8,
        'ny': 8,
        'nt': 10,
        'x_range': (0.0, 1.0),
        'y_range': (0.0, 1.0),
        't_range': (0.0, 0.5)
    }


@pytest.fixture
def tiny_edge_index(set_seed):
    """Provide tiny graph edge index for GNN tests."""
    set_seed(1234)
    # Simple 4-node graph with 6 edges (bidirectional)
    edge_index = np.array([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=np.int64)
    return edge_index


@pytest.fixture
def tiny_features(set_seed):
    """Provide tiny feature matrix for graph tests."""
    set_seed(1234)
    num_nodes = 4
    num_features = 5
    return np.random.randn(num_nodes, num_features).astype(np.float32)


@pytest.fixture
def tiny_graph(tiny_edge_index, tiny_features):
    """Provide complete tiny graph for GNN tests."""
    return {
        'edge_index': tiny_edge_index,
        'features': tiny_features,
        'num_nodes': 4,
        'num_features': 5,
        'num_edges': 6
    }


# Markers for different test types
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")
    config.addinivalue_line("markers", "jax: marks tests that require JAX backend")
    config.addinivalue_line("markers", "torch: marks tests that require PyTorch backend")


# Skip GPU tests if CUDA is not available
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA is not available."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")

    for item in items:
        if "gpu" in item.keywords:
            try:
                import jax

                # Check if GPU is available
                if not jax.devices("gpu"):
                    item.add_marker(skip_gpu)
            except ImportError:
                item.add_marker(skip_gpu)
