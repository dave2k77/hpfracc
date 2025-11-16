#!/usr/bin/env python3
"""
Week 2: Fractional ODE system shape checks.
"""

import pytest
import numpy as np

pytestmark = pytest.mark.week2


def test_solve_fractional_system_shapes():
    from hpfracc.solvers.ode_solvers import solve_fractional_system

    def f(t, y):
        # y is (2,), simple coupled linear decay
        return np.array([-0.5 * y[0], -1.0 * y[1]])

    t, Y = solve_fractional_system(
        f,
        t_span=(0.0, 0.3),
        y0=np.array([1.0, 2.0]),
        alpha=1.0,
        method="euler",
        h=0.01,
    )

    assert t.ndim == 1 and Y.ndim == 2
    assert Y.shape[0] == t.shape[0]
    assert Y.shape[1] == 2
    assert np.isfinite(Y).all()