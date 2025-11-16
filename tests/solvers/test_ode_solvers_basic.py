#!/usr/bin/env python3
"""
Week 2: Basic fractional ODE solver tests.
Focus: API usage, shapes, simple correctness for alpha=1 (Euler).
"""

import pytest
import numpy as np

pytestmark = pytest.mark.week2


def test_euler_decay_alpha1_small_step():
    from hpfracc.solvers.ode_solvers import solve_fractional_ode

    k = 1.0

    def f(t, y):
        return -k * y

    t_span = (0.0, 1.0)
    y0 = 1.0
    alpha = 1.0
    h = 0.002  # small step for reasonable Euler accuracy

    t, y = solve_fractional_ode(f, t_span, y0, alpha, method="euler", h=h)

    assert t.ndim == 1 and y.ndim == 2 or y.ndim == 1
    y_end = y[-1][0] if y.ndim == 2 else y[-1]

    # Expected solution for dy/dt=-y at t=1 is exp(-1)
    expected = np.exp(-1.0)
    assert abs(float(y_end) - expected) < 0.03


@pytest.mark.parametrize("alpha", [0.5, 0.8])
def test_predictor_corrector_shapes_and_finite(alpha):
    from hpfracc.solvers.ode_solvers import solve_fractional_ode

    k = 0.5

    def f(t, y):
        return -k * y

    t_span = (0.0, 0.5)
    y0 = 1.0

    t, y = solve_fractional_ode(
        f, t_span, y0, alpha, method="predictor_corrector", h=0.01
    )

    assert t.shape[0] == y.shape[0]
    assert np.isfinite(y).all()
    # Starts at initial condition
    y0_observed = y[0][0] if y.ndim == 2 else y[0]
    assert abs(float(y0_observed) - 1.0) < 1e-12


@pytest.mark.parametrize("bad_alpha", [0.0, 2.5])
def test_invalid_alpha_raises_for_ode(bad_alpha):
    from hpfracc.solvers.ode_solvers import solve_fractional_ode

    def f(t, y):
        return -y

    with pytest.raises(ValueError):
        solve_fractional_ode(f, (0.0, 0.1), 1.0, bad_alpha, method="euler", h=0.01)