#!/usr/bin/env python3
"""
Tests for PDE solvers with small grids for fast execution.
Target: Cover fractional heat/wave equations, initialization, error handling.
"""

import pytest
import numpy as np


def test_fractional_heat_equation_1d(small_grid, set_seed):
    """Test 1D fractional heat equation solver with small grid."""
    set_seed(42)
    
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_1d
        
        nx = small_grid['nx']
        nt = small_grid['nt']
        alpha = 0.5
        
        # Initial condition
        x = np.linspace(0, 1, nx)
        u0 = np.sin(np.pi * x)
        
        # Solve
        u = solve_fractional_heat_1d(u0, alpha=alpha, dx=1.0/nx, dt=0.01, nt=nt)
        
        assert u.shape == (nt + 1, nx)
        assert not np.isnan(u).any()
    except ImportError:
        pytest.skip("solve_fractional_heat_1d not available")


def test_fractional_heat_equation_2d(small_grid, set_seed):
    """Test 2D fractional heat equation solver with small grid."""
    set_seed(42)
    
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_2d
        
        nx = small_grid['nx']
        ny = small_grid['ny']
        nt = small_grid['nt']
        alpha = 0.5
        
        # Initial condition
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
        
        # Solve
        u = solve_fractional_heat_2d(u0, alpha=alpha, dx=1.0/nx, dy=1.0/ny, 
                                     dt=0.01, nt=nt)
        
        assert u.shape == (nt + 1, nx, ny)
        assert not np.isnan(u).any()
    except ImportError:
        pytest.skip("solve_fractional_heat_2d not available")


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7, 1.0])
def test_fractional_heat_various_orders(alpha, small_grid, set_seed):
    """Test fractional heat equation with various fractional orders."""
    set_seed(42)
    
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_1d
        
        nx = small_grid['nx']
        nt = 5  # Very small for speed
        
        x = np.linspace(0, 1, nx)
        u0 = np.sin(np.pi * x)
        
        u = solve_fractional_heat_1d(u0, alpha=alpha, dx=1.0/nx, dt=0.01, nt=nt)
        
        assert u.shape == (nt + 1, nx)
    except ImportError:
        pytest.skip("solve_fractional_heat_1d not available")


def test_fractional_wave_equation_1d(small_grid, set_seed):
    """Test 1D fractional wave equation solver."""
    set_seed(42)
    
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_wave_1d
        
        nx = small_grid['nx']
        nt = small_grid['nt']
        alpha = 0.5
        
        x = np.linspace(0, 1, nx)
        u0 = np.sin(np.pi * x)
        v0 = np.zeros(nx)  # Initial velocity
        
        u = solve_fractional_wave_1d(u0, v0, alpha=alpha, dx=1.0/nx, 
                                     dt=0.01, nt=nt)
        
        assert u.shape == (nt + 1, nx)
    except ImportError:
        pytest.skip("solve_fractional_wave_1d not available")


def test_fractional_diffusion_equation(small_grid, set_seed):
    """Test fractional diffusion equation solver."""
    set_seed(42)
    
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_diffusion
        
        nx = small_grid['nx']
        nt = small_grid['nt']
        alpha = 0.8
        
        x = np.linspace(0, 1, nx)
        u0 = np.exp(-((x - 0.5) ** 2) / 0.01)  # Gaussian
        
        u = solve_fractional_diffusion(u0, alpha=alpha, dx=1.0/nx, 
                                       dt=0.01, nt=nt)
        
        assert u.shape == (nt + 1, nx)
    except ImportError:
        pytest.skip("solve_fractional_diffusion not available")


def test_pde_solver_zero_initial_condition(small_grid):
    """Test PDE solver with zero initial condition."""
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_1d
        
        nx = small_grid['nx']
        nt = 5
        
        u0 = np.zeros(nx)
        u = solve_fractional_heat_1d(u0, alpha=0.5, dx=1.0/nx, dt=0.01, nt=nt)
        
        # With zero IC and zero BC, solution should remain zero
        assert np.allclose(u, 0.0, atol=1e-6)
    except ImportError:
        pytest.skip("solve_fractional_heat_1d not available")


def test_pde_solver_constant_initial_condition(small_grid):
    """Test PDE solver with constant initial condition."""
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_1d
        
        nx = small_grid['nx']
        nt = 5
        
        u0 = np.ones(nx) * 2.0
        u = solve_fractional_heat_1d(u0, alpha=0.5, dx=1.0/nx, dt=0.01, nt=nt)
        
        assert u.shape == (nt + 1, nx)
    except ImportError:
        pytest.skip("solve_fractional_heat_1d not available")


def test_pde_solver_dirichlet_boundary(small_grid, set_seed):
    """Test PDE solver with Dirichlet boundary conditions."""
    set_seed(42)
    
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_1d
        
        nx = small_grid['nx']
        nt = 5
        
        x = np.linspace(0, 1, nx)
        u0 = np.sin(np.pi * x)
        
        # Dirichlet BC: u(0) = u(1) = 0
        u = solve_fractional_heat_1d(u0, alpha=0.5, dx=1.0/nx, dt=0.01, nt=nt,
                                     boundary='dirichlet')
        
        # Check boundaries are zero (or close to zero)
        assert np.abs(u[:, 0]).max() < 1e-3
        assert np.abs(u[:, -1]).max() < 1e-3
    except (ImportError, TypeError):
        pytest.skip("Dirichlet BC not available")


def test_pde_solver_neumann_boundary(small_grid, set_seed):
    """Test PDE solver with Neumann boundary conditions."""
    set_seed(42)
    
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_1d
        
        nx = small_grid['nx']
        nt = 5
        
        x = np.linspace(0, 1, nx)
        u0 = np.ones(nx)
        
        u = solve_fractional_heat_1d(u0, alpha=0.5, dx=1.0/nx, dt=0.01, nt=nt,
                                     boundary='neumann')
        
        assert u.shape == (nt + 1, nx)
    except (ImportError, TypeError):
        pytest.skip("Neumann BC not available")


def test_pde_solver_periodic_boundary(small_grid, set_seed):
    """Test PDE solver with periodic boundary conditions."""
    set_seed(42)
    
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_1d
        
        nx = small_grid['nx']
        nt = 5
        
        x = np.linspace(0, 1, nx, endpoint=False)
        u0 = np.sin(2 * np.pi * x)
        
        u = solve_fractional_heat_1d(u0, alpha=0.5, dx=1.0/nx, dt=0.01, nt=nt,
                                     boundary='periodic')
        
        # Check periodicity
        assert np.allclose(u[:, 0], u[:, -1], rtol=1e-2)
    except (ImportError, TypeError):
        pytest.skip("Periodic BC not available")


@pytest.mark.parametrize("method", ["explicit", "implicit", "crank_nicolson"])
def test_pde_solver_different_methods(method, small_grid, set_seed):
    """Test different numerical methods for PDE solving."""
    set_seed(42)
    
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_1d
        
        nx = small_grid['nx']
        nt = 5
        
        x = np.linspace(0, 1, nx)
        u0 = np.sin(np.pi * x)
        
        u = solve_fractional_heat_1d(u0, alpha=0.5, dx=1.0/nx, dt=0.01, nt=nt,
                                     method=method)
        
        assert u.shape == (nt + 1, nx)
    except (ImportError, TypeError, ValueError):
        pytest.skip(f"Method {method} not available")


def test_pde_solver_energy_decay_heat(small_grid, set_seed):
    """Test that heat equation shows energy decay."""
    set_seed(42)
    
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_1d
        
        nx = small_grid['nx']
        nt = small_grid['nt']
        
        x = np.linspace(0, 1, nx)
        u0 = np.sin(np.pi * x)
        
        u = solve_fractional_heat_1d(u0, alpha=0.5, dx=1.0/nx, dt=0.01, nt=nt)
        
        # Energy (L2 norm) should decay over time
        energy_initial = np.linalg.norm(u[0])
        energy_final = np.linalg.norm(u[-1])
        
        assert energy_final <= energy_initial
    except ImportError:
        pytest.skip("solve_fractional_heat_1d not available")


def test_pde_solver_stability(small_grid, set_seed):
    """Test stability of PDE solver (no blow-up)."""
    set_seed(42)
    
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_1d
        
        nx = small_grid['nx']
        nt = small_grid['nt']
        
        x = np.linspace(0, 1, nx)
        u0 = np.sin(np.pi * x)
        
        u = solve_fractional_heat_1d(u0, alpha=0.5, dx=1.0/nx, dt=0.01, nt=nt)
        
        # Check no NaN or Inf
        assert not np.isnan(u).any()
        assert not np.isinf(u).any()
        
        # Check solution doesn't blow up
        assert np.abs(u).max() < 100.0
    except ImportError:
        pytest.skip("solve_fractional_heat_1d not available")


def test_pde_solver_invalid_alpha():
    """Test PDE solver with invalid fractional order."""
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_1d
        
        nx = 8
        u0 = np.ones(nx)
        
        with pytest.raises((ValueError, AssertionError)):
            solve_fractional_heat_1d(u0, alpha=2.5, dx=0.1, dt=0.01, nt=5)
    except ImportError:
        pytest.skip("solve_fractional_heat_1d not available")


def test_pde_solver_negative_alpha():
    """Test PDE solver with negative fractional order."""
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_1d
        
        nx = 8
        u0 = np.ones(nx)
        
        with pytest.raises((ValueError, AssertionError)):
            solve_fractional_heat_1d(u0, alpha=-0.5, dx=0.1, dt=0.01, nt=5)
    except ImportError:
        pytest.skip("solve_fractional_heat_1d not available")


def test_pde_solver_negative_dt():
    """Test PDE solver with negative time step."""
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_1d
        
        nx = 8
        u0 = np.ones(nx)
        
        with pytest.raises((ValueError, AssertionError)):
            solve_fractional_heat_1d(u0, alpha=0.5, dx=0.1, dt=-0.01, nt=5)
    except ImportError:
        pytest.skip("solve_fractional_heat_1d not available")


def test_pde_solver_zero_spatial_step():
    """Test PDE solver with zero spatial step."""
    try:
        from hpfracc.solvers.pde_solvers import solve_fractional_heat_1d
        
        nx = 8
        u0 = np.ones(nx)
        
        with pytest.raises((ValueError, ZeroDivisionError, AssertionError)):
            solve_fractional_heat_1d(u0, alpha=0.5, dx=0.0, dt=0.01, nt=5)
    except ImportError:
        pytest.skip("solve_fractional_heat_1d not available")
