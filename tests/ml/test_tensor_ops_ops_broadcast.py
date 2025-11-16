#!/usr/bin/env python3
"""
Week 2: TensorOps elementwise ops, broadcasting, and basic transforms.
"""

import pytest
import numpy as np

from hpfracc.ml.tensor_ops import get_tensor_ops

pytestmark = pytest.mark.week2


def _to_np(x, ops):
    return ops.to_numpy(x) if hasattr(ops, "to_numpy") else x


def test_elementwise_add_broadcast():
    ops = get_tensor_ops()
    a = ops.create_tensor([[1.0], [2.0]])       # (2,1)
    b = ops.create_tensor([[10.0, 20.0, 30.0]]) # (1,3)

    c = ops.add(a, b)  # broadcast to (2,3)
    c_np = _to_np(c, ops)

    expected = np.array([[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]])
    assert c_np.shape == (2, 3)
    assert np.allclose(c_np, expected)


def test_multiply_and_power():
    ops = get_tensor_ops()
    x = ops.create_tensor([1.0, 2.0, 3.0])
    y = ops.create_tensor([2.0, 2.0, 2.0])

    prod = ops.multiply(x, y)
    sq = ops.power(x, 2.0)
    prod_np = _to_np(prod, ops)
    sq_np = _to_np(sq, ops)

    assert np.allclose(prod_np, [2.0, 4.0, 6.0])
    assert np.allclose(sq_np, [1.0, 4.0, 9.0])


def test_reshape_unsqueeze_squeeze_cat():
    ops = get_tensor_ops()
    v = ops.create_tensor([1, 2, 3, 4, 5, 6])
    m = ops.reshape(v, (2, 3))
    m2 = ops.unsqueeze(m, 0)
    m3 = ops.squeeze(m2, 0)
    cat = ops.concatenate([m3, m3], dim=0)

    m_np = _to_np(m, ops)
    cat_np = _to_np(cat, ops)

    assert m_np.shape == (2, 3)
    assert cat_np.shape == (4, 3)


def test_transpose_and_matmul():
    ops = get_tensor_ops()
    a = ops.create_tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])  # (2,3)
    b = ops.create_tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])  # (3,2)

    ab = ops.matmul(a, b)
    ab_np = _to_np(ab, ops)
    assert ab_np.shape == (2, 2)
    assert np.allclose(ab_np, [[14.0, 8.0], [2.0, 1.0]])