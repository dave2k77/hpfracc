#!/usr/bin/env python3
"""
Week 2: Fractional SDE solver smoke tests.
Focus: deterministic decay (sigma=0), invalid method and alpha.
"""

import pytest
import numpy as np

pytestmark = pytest.mark.week2


def test_sde_euler_maruyama_deterministic_decay():
    from hpfracc.solvers.sde_solvers import solve_fractional_sde

    k = 0.5

    def drift(t, x):
        return -k * x

    def diffusion(t, x):
        return 0.0  # deterministic

    x0 = np.array([1.0])
    t_span = (0.0, 1.0)

    sol = solve_fractional_sde(
        drift,
        diffusion,
        x0,
        t_span,
        fractional_order=1.0,
        method="euler_maruyama",
        num_steps=50,
        seed=123,
    )

    assert sol.t.shape[0] == sol.y.shape[0]
    assert sol.y.ndim == 2 and sol.y.shape[1] == 1
    # Deterministic decay should be non-increasing
    assert np.all(np.diff(sol.y[:, 0]) <= 1e-8)
    assert np.isfinite(sol.y).all()


def test_sde_invalid_method_raises():
    from hpfracc.solvers.sde_solvers import solve_fractional_sde

    def drift(t, x):
        return -x

    def diffusion(t, x):
        return 0.0

    with pytest.raises(ValueError):
        solve_fractional_sde(drift, diffusion, np.array([1.0]), (0.0, 1.0), method="unknown")


@pytest.mark.parametrize("bad_alpha", [0.0, 2.1])
def test_sde_invalid_fractional_order_raises(bad_alpha):
    from hpfracc.solvers.sde_solvers import solve_fractional_sde

    def drift(t, x):
        return -x

    def diffusion(t, x):
        return 0.0

    with pytest.raises(ValueError):
        solve_fractional_sde(drift, diffusion, np.array([1.0]), (0.0, 1.0), fractional_order=bad_alpha)