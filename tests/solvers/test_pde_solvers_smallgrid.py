#!/usr/bin/env python3
"""
Week 1: Minimal PDE diffusion smoke tests on a tiny grid.
Covers API usage, shape checks, basic properties, and error paths.
"""

import pytest
import numpy as np

pytestmark = pytest.mark.week1


def _dirichlet_zero():
    return (lambda t: 0.0, lambda t: 0.0)


def test_fractional_diffusion_smallgrid(small_grid, set_seed):
    set_seed(42)

    from hpfracc.solvers.pde_solvers import solve_fractional_diffusion

    nx = small_grid["nx"]
    nt = small_grid["nt"]

    x_span = (0.0, 1.0)
    t_span = (0.0, 0.05)
    alpha = 0.5
    beta = 2.0

    def u0(x):
        return np.sin(np.pi * x)

    t, x, u = solve_fractional_diffusion(
        x_span=x_span,
        t_span=t_span,
        initial_condition=u0,
        boundary_conditions=_dirichlet_zero(),
        alpha=alpha,
        beta=beta,
        diffusion_coeff=1.0,
        nx=nx,
        nt=nt,
    )

    assert u.shape == (nt, nx)
    assert np.isfinite(u).all()


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_fractional_diffusion_orders(alpha, small_grid):
    from hpfracc.solvers.pde_solvers import solve_fractional_diffusion

    nx = small_grid["nx"]
    nt = 6

    x_span = (0.0, 1.0)
    t_span = (0.0, 0.03)
    beta = 2.0

    def u0(x):
        return np.sin(np.pi * x)

    t, x, u = solve_fractional_diffusion(
        x_span=x_span,
        t_span=t_span,
        initial_condition=u0,
        boundary_conditions=_dirichlet_zero(),
        alpha=alpha,
        beta=beta,
        diffusion_coeff=1.0,
        nx=nx,
        nt=nt,
    )

    assert u.shape == (nt, nx)


def test_zero_initial_condition_remains_zero(small_grid):
    from hpfracc.solvers.pde_solvers import solve_fractional_diffusion

    nx = small_grid["nx"]
    nt = 5

    x_span = (0.0, 1.0)
    t_span = (0.0, 0.02)

    def u0(x):
        return 0.0

    t, x, u = solve_fractional_diffusion(
        x_span=x_span,
        t_span=t_span,
        initial_condition=u0,
        boundary_conditions=_dirichlet_zero(),
        alpha=1.0,
        beta=2.0,
        diffusion_coeff=1.0,
        nx=nx,
        nt=nt,
    )

    assert np.allclose(u, 0.0, atol=1e-8)


def test_invalid_alpha_raises():
    from hpfracc.solvers.pde_solvers import solve_fractional_diffusion

    def u0(x):
        return 1.0

    with pytest.raises(ValueError):
        solve_fractional_diffusion(
            x_span=(0.0, 1.0),
            t_span=(0.0, 0.01),
            initial_condition=u0,
            boundary_conditions=_dirichlet_zero(),
            alpha=2.5,
            beta=2.0,
            nx=8,
            nt=4,
        )

    with pytest.raises(ValueError):
        solve_fractional_diffusion(
            x_span=(0.0, 1.0),
            t_span=(0.0, 0.01),
            initial_condition=u0,
            boundary_conditions=_dirichlet_zero(),
            alpha=0.0,
            beta=2.0,
            nx=8,
            nt=4,
        )
