#!/usr/bin/env python3
"""
Week 2: Fractional ODE predictor-corrector accuracy (alpha=1) against exp(-kt).
"""

import pytest
import numpy as np

pytestmark = pytest.mark.week2


def test_predictor_corrector_accuracy_alpha1():
    from hpfracc.solvers.ode_solvers import solve_fractional_ode

    k = 1.0

    def f(t, y):
        return -k * y

    t_span = (0.0, 1.0)
    y0 = 1.0
    alpha = 1.0
    h = 0.005

    t, y = solve_fractional_ode(f, t_span, y0, alpha, method="predictor_corrector", h=h)

    y_vec = y[:, 0] if y.ndim == 2 else y
    y_true = np.exp(-k * t)
    rmse = np.sqrt(np.mean((y_vec - y_true) ** 2))

    assert rmse < 0.05