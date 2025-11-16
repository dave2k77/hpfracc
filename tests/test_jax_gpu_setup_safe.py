#!/usr/bin/env python3
"""
Week 1: Safe JAX setup tests that avoid direct JAX imports.
Focus on environment flags and TensorOps CPU fallback behavior.
"""

import pytest
import os
import numpy as np

pytestmark = pytest.mark.week1


def test_jax_gpu_setup_skip_flag():
    """HPFRACC_SKIP_JAX_INIT prevents heavy initialization during import."""
    original = os.environ.get("HPFRACC_SKIP_JAX_INIT")
    os.environ["HPFRACC_SKIP_JAX_INIT"] = "1"

    try:
        import hpfracc.jax_gpu_setup as jax_setup
        assert jax_setup is not None
    except ImportError as e:
        pytest.skip(f"jax_gpu_setup not available: {e}")
    finally:
        if original is not None:
            os.environ["HPFRACC_SKIP_JAX_INIT"] = original
        else:
            os.environ.pop("HPFRACC_SKIP_JAX_INIT", None)


def test_check_jax_available_returns_bool():
    """check_jax_available should always return a boolean."""
    try:
        from hpfracc.jax_gpu_setup import check_jax_available
    except ImportError:
        pytest.skip("jax_gpu_setup not available")

    available = check_jax_available()
    assert isinstance(available, bool)


def test_disable_jax_env_variable_falls_back_to_cpu(disable_jax):
    """With JAX disabled, TensorOps should still create/convert tensors."""
    from hpfracc.ml.tensor_ops import get_tensor_ops

    ops = get_tensor_ops()
    arr = np.array([1.0, 2.0, 3.0])
    t = ops.create_tensor(arr)
    out = ops.to_numpy(t)
    assert np.allclose(out, arr)
