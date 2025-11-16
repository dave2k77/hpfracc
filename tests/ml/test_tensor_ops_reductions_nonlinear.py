#!/usr/bin/env python3
"""
Week 2: TensorOps reductions and nonlinearities.
"""

import pytest
import numpy as np

from hpfracc.ml.tensor_ops import get_tensor_ops

pytestmark = pytest.mark.week2


def _np(x, ops):
    return ops.to_numpy(x) if hasattr(ops, "to_numpy") else x


def test_sum_mean_std():
    ops = get_tensor_ops()
    x = ops.create_tensor([[1.0, 2.0], [3.0, 4.0]])

    s0 = ops.sum(x)
    m0 = ops.mean(x)
    s1 = ops.sum(x, dim=0)
    m1 = ops.mean(x, dim=1)
    st = ops.std(x)

    s0, m0, s1, m1, st = map(lambda v: _np(v, ops), (s0, m0, s1, m1, st))
    assert np.isclose(s0, 10.0)
    assert np.isclose(m0, 2.5)
    assert np.allclose(s1, [4.0, 6.0])
    assert np.allclose(m1, [1.5, 3.5])
    assert st >= 0.0


def test_softmax_sigmoid_shapes():
    ops = get_tensor_ops()
    x = ops.create_tensor([[1.0, 2.0, 3.0]])
    sm = ops.softmax(x, dim=-1)
    sg = ops.sigmoid(x)
    sm_np = _np(sm, ops)
    sg_np = _np(sg, ops)

    assert sm_np.shape == (1, 3)
    assert sg_np.shape == (1, 3)
    assert np.allclose(sm_np.sum(axis=-1), 1.0)