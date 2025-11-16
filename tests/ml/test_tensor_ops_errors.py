import pytest
pytestmark = pytest.mark.week1
import numpy as np
import pytest

from hpfracc.ml.tensor_ops import get_tensor_ops


def test_mismatched_shapes_concatenate():
    ops = get_tensor_ops()

    x = np.random.randn(2, 3).astype(np.float32)
    y = np.random.randn(2, 4).astype(np.float32)  # mismatched second dim

    tx = ops.create_tensor(x)
    ty = ops.create_tensor(y)

    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        ops.concatenate([tx, ty], dim=0)


def test_invalid_einsum_raises():
    ops = get_tensor_ops()

    a = ops.create_tensor(np.random.randn(2, 3).astype(np.float32))
    b = ops.create_tensor(np.random.randn(4, 5).astype(np.float32))

    with pytest.raises(Exception):
        ops.einsum("ij,jk->ik", a, b)  # incompatible inner dims

