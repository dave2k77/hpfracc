import pytest
pytestmark = pytest.mark.week1
import os
import numpy as np
import pytest

from hpfracc.ml.tensor_ops import get_tensor_ops, switch_backend
from hpfracc.ml.backends import BackendType


def test_backend_autodetection_numpy():
    os.environ.pop("HPFRACC_BACKEND", None)
    ops = get_tensor_ops()

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = ops.create_tensor(x)
    x_back = ops.to_numpy(t)

    assert isinstance(x_back, np.ndarray)
    np.testing.assert_allclose(x_back, x)


def test_basic_creation_and_shapes():
    ops = get_tensor_ops()

    z = ops.zeros((2, 3))
    o = ops.ones((2, 3))
    r = ops.reshape(o, (3, 2))

    assert ops.to_numpy(z).shape == (2, 3)
    assert ops.to_numpy(o).shape == (2, 3)
    assert ops.to_numpy(r).shape == (3, 2)


@pytest.mark.skipif(not pytest.importorskip("torch", reason="torch not available"), reason="torch not available")
def test_switch_to_torch_and_back():
    # Switch to torch, then back to numpy
    switch_backend(BackendType.TORCH)
    ops = get_tensor_ops()
    t = ops.ones((2,))
    arr = ops.to_numpy(t)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,)

    # Back to numpy
    switch_backend(BackendType.NUMBA)
    ops2 = get_tensor_ops()
    a2 = ops2.zeros((1,))
    assert ops2.to_numpy(a2).shape == (1,)

