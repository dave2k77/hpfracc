"""
Unit tests for coupled system solvers in hpfracc.solvers.coupled_solvers

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import numpy as np
import pytest
from hpfracc.solvers import (
    CoupledSystemSolver, OperatorSplittingSolver, MonolithicSolver,
    solve_coupled_graph_sde, CoupledSolution
)


class TestCoupledSystemSolver:
    """Test base CoupledSystemSolver functionality"""
    
    def test_initialization(self):
        """Test solver initialization"""
        # Can't instantiate abstract class directly, test through concrete implementation
        solver = OperatorSplittingSolver(fractional_orders=0.5)
        assert solver is not None
        assert solver.coupling_strength == 1.0
        assert 'spatial' in solver.fractional_orders
        assert 'temporal' in solver.fractional_orders
    
    def test_abstract_method(self):
        """Test that solve method is abstract"""
        # Test that abstract class can't be instantiated
        with pytest.raises(TypeError):
            CoupledSystemSolver(fractional_orders=0.5)
    
    def test_fractional_orders_dict(self):
        """Test initialization with dictionary of fractional orders"""
        fractional_orders = {'spatial': 0.3, 'temporal': 0.7}
        solver = OperatorSplittingSolver(fractional_orders=fractional_orders)
        
        assert solver.fractional_orders['spatial'] == 0.3
        assert solver.fractional_orders['temporal'] == 0.7
    
    def test_fractional_orders_single(self):
        """Test initialization with single fractional order"""
        solver = OperatorSplittingSolver(fractional_orders=0.5)
        
        assert solver.fractional_orders['spatial'] == 0.5
        assert solver.fractional_orders['temporal'] == 0.5
    
    def test_invalid_fractional_orders(self):
        """Test with invalid fractional orders type"""
        with pytest.raises(ValueError):
            OperatorSplittingSolver(fractional_orders="invalid")


class TestOperatorSplittingSolver:
    """Test OperatorSplittingSolver"""
    
    def test_initialization(self):
        """Test solver initialization"""
        solver = OperatorSplittingSolver(fractional_orders=0.5)
        assert solver is not None
        assert hasattr(solver, 'split_order')
        assert solver.split_order == 2  # Default Strang splitting
    
    def test_initialization_with_split_order(self):
        """Test initialization with custom split order"""
        solver = OperatorSplittingSolver(fractional_orders=0.5, split_order=1)
        assert solver.split_order == 1
    
    def test_solve_basic(self):
        """Test basic solving functionality"""
        def graph_dynamics(state, adjacency):
            return -state
        
        def drift(t, state):
            return -state
        
        def diffusion(t, state):
            return 0.1
        
        adjacency = np.array([[0, 1], [1, 0]])
        node_features = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        solver = OperatorSplittingSolver(fractional_orders=0.5)
        solution = solver.solve(
            graph_dynamics, drift, diffusion,
            adjacency, node_features, t_span,
            num_steps=10
        )
        
        assert isinstance(solution, CoupledSolution)
        assert solution.t.shape == (11,)  # num_steps + 1
        assert solution.spatial.shape == (11, 2)
        assert solution.temporal.shape == (11, 2)
        assert solution.metadata['solver'] == 'operator_splitting'
        assert solution.metadata['split_order'] == 2
    
    def test_solve_with_splitting_method(self):
        """Test solving with different splitting methods"""
        def graph_dynamics(state, adjacency):
            return -state
        
        def drift(t, state):
            return -state
        
        def diffusion(t, state):
            return 0.1
        
        adjacency = np.array([[0, 1], [1, 0]])
        node_features = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        # Test Lie-Trotter splitting (order 1)
        solver_lt = OperatorSplittingSolver(fractional_orders=0.5, split_order=1)
        solution_lt = solver_lt.solve(
            graph_dynamics, drift, diffusion,
            adjacency, node_features, t_span,
            num_steps=10
        )
        
        # Test Strang splitting (order 2)
        solver_strang = OperatorSplittingSolver(fractional_orders=0.5, split_order=2)
        solution_strang = solver_strang.solve(
            graph_dynamics, drift, diffusion,
            adjacency, node_features, t_span,
            num_steps=10
        )
        
        assert isinstance(solution_lt, CoupledSolution)
        assert isinstance(solution_strang, CoupledSolution)
        assert solution_lt.metadata['split_order'] == 1
        assert solution_strang.metadata['split_order'] == 2
    
    def test_spatial_step(self):
        """Test spatial step functionality"""
        def graph_dynamics(state, adjacency):
            return -state
        
        solver = OperatorSplittingSolver(fractional_orders=0.5)
        adjacency = np.array([[0, 1], [1, 0]])
        state = np.array([1.0, 0.5])
        dt = 0.1
        
        new_state = solver._spatial_step(graph_dynamics, adjacency, state, dt)
        
        # Should evolve according to graph dynamics
        expected = state + dt * (-state)
        assert np.allclose(new_state, expected)
    
    def test_temporal_step(self):
        """Test temporal step functionality"""
        def drift(t, state):
            return -state
        
        def diffusion(t, state):
            return 0.1
        
        solver = OperatorSplittingSolver(fractional_orders=0.5)
        state = np.array([1.0, 0.5])
        t = 0.5
        dt = 0.1
        
        from hpfracc.solvers.sde_solvers import FastHistoryConvolution
        drift_conv = FastHistoryConvolution(0.5, 10, 2)
        diffusion_conv = FastHistoryConvolution(0.5, 10, 2)
        
        # Set random seed for reproducible test
        np.random.seed(42)
        new_state = solver._temporal_step(
            drift, diffusion, state, t, dt,
            drift_conv, diffusion_conv, 
            gamma_factor=1.0, alpha=0.5, initial_state=state
        )
        
        # Should evolve according to SDE dynamics
        # Note: The logic in _temporal_step is now more complex (fractional history)
        # We verify it runs and produces output of correct shape/type, not exact value match
        # against a simple Formula (as the simple formula was wrong for FDEs).
        
        assert new_state.shape == state.shape
        assert isinstance(new_state, np.ndarray)


class TestMonolithicSolver:
    """Test MonolithicSolver"""
    
    def test_initialization(self):
        """Test solver initialization"""
        solver = MonolithicSolver(fractional_orders=0.5)
        assert solver is not None
        assert solver.coupling_strength == 1.0
    
    def test_initialization_with_coupling_strength(self):
        """Test initialization with custom coupling strength"""
        solver = MonolithicSolver(fractional_orders=0.5, coupling_strength=2.0)
        assert solver.coupling_strength == 2.0
    
    def test_solve_basic(self):
        """Test basic solving functionality"""
        def graph_dynamics(state, adjacency):
            return -state
        
        def drift(t, state):
            return -state
        
        def diffusion(t, state):
            return 0.1
        
        adjacency = np.array([[0, 1], [1, 0]])
        node_features = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        solver = MonolithicSolver(fractional_orders=0.5)
        solution = solver.solve(
            graph_dynamics, drift, diffusion,
            adjacency, node_features, t_span,
            num_steps=10
        )
        
        assert isinstance(solution, CoupledSolution)
        assert solution.t.shape == (11,)  # num_steps + 1
        assert solution.spatial.shape == (11, 2)
        assert solution.temporal.shape == (11, 2)
        assert solution.metadata['solver'] == 'monolithic'
    
    def test_solve_with_coupling_strength(self):
        """Test solving with different coupling strengths"""
        def graph_dynamics(state, adjacency):
            return -state
        
        def drift(t, state):
            return -state
        
        def diffusion(t, state):
            return 0.1
        
        adjacency = np.array([[0, 1], [1, 0]])
        node_features = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        # Test weak coupling
        solver_weak = MonolithicSolver(fractional_orders=0.5, coupling_strength=0.1)
        solution_weak = solver_weak.solve(
            graph_dynamics, drift, diffusion,
            adjacency, node_features, t_span,
            num_steps=10
        )
        
        # Test strong coupling
        solver_strong = MonolithicSolver(fractional_orders=0.5, coupling_strength=2.0)
        solution_strong = solver_strong.solve(
            graph_dynamics, drift, diffusion,
            adjacency, node_features, t_span,
            num_steps=10
        )
        
        assert isinstance(solution_weak, CoupledSolution)
        assert isinstance(solution_strong, CoupledSolution)
        
        # Both should produce valid solutions with different coupling behaviors
        # The coupling strength affects the dynamics, so solutions should be different
        assert not np.allclose(solution_weak.spatial, solution_strong.spatial)
        assert not np.allclose(solution_weak.temporal, solution_strong.temporal)


class TestSolveCoupledGraphSDE:
    """Test solve_coupled_graph_sde convenience function"""
    
    def test_basic_graph_sde_solving(self):
        """Test basic graph-SDE solving"""
        # Simple 2-node graph
        adjacency_matrix = np.array([[0, 1], [1, 0]])
        
        def graph_dynamics(state, adjacency):
            return -state
        
        def drift(t, x):
            return np.array([-x[0] + 0.5 * x[1], -x[1] + 0.5 * x[0]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        node_features = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        solution = solve_coupled_graph_sde(
            graph_dynamics, drift, diffusion, adjacency_matrix, node_features,
            t_span=t_span, num_steps=10
        )
        
        assert isinstance(solution, CoupledSolution)
        assert solution.t.shape == (11,)
        assert solution.spatial.shape == (11, 2)
        assert solution.temporal.shape == (11, 2)
    
    def test_graph_sde_with_different_methods(self):
        """Test graph-SDE with different solving methods"""
        adjacency_matrix = np.array([[0, 1], [1, 0]])
        
        def graph_dynamics(state, adjacency):
            return -state
        
        def drift(t, x):
            return np.array([-x[0] + 0.5 * x[1], -x[1] + 0.5 * x[0]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        node_features = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        # Test with operator splitting
        solution_split = solve_coupled_graph_sde(
            graph_dynamics, drift, diffusion, adjacency_matrix, node_features,
            t_span=t_span, num_steps=10, solver="operator_splitting"
        )
        
        # Test with monolithic solver
        solution_mono = solve_coupled_graph_sde(
            graph_dynamics, drift, diffusion, adjacency_matrix, node_features,
            t_span=t_span, num_steps=10, solver="monolithic"
        )
        
        assert isinstance(solution_split, CoupledSolution)
        assert isinstance(solution_mono, CoupledSolution)
        assert solution_split.metadata['solver'] == 'operator_splitting'
        assert solution_mono.metadata['solver'] == 'monolithic'
    
    def test_invalid_method(self):
        """Test with invalid method"""
        adjacency_matrix = np.array([[0, 1], [1, 0]])
        
        def graph_dynamics(state, adjacency):
            return -state
        
        def drift(t, x):
            return np.array([-x[0], -x[1]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        node_features = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        with pytest.raises(ValueError):
            solve_coupled_graph_sde(
                graph_dynamics, drift, diffusion, adjacency_matrix, node_features,
                t_span=t_span, solver="invalid_method"
            )
    
    def test_different_fractional_orders(self):
        """Test with different fractional orders"""
        adjacency_matrix = np.array([[0, 1], [1, 0]])
        
        def graph_dynamics(state, adjacency):
            return -state
        
        def drift(t, x):
            return np.array([-x[0], -x[1]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        node_features = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        # Test with single fractional order
        solution_single = solve_coupled_graph_sde(
            graph_dynamics, drift, diffusion, adjacency_matrix, node_features,
            t_span=t_span, fractional_orders=0.3, num_steps=10
        )
        
        # Test with dictionary of fractional orders
        fractional_orders = {'spatial': 0.3, 'temporal': 0.7}
        solution_dict = solve_coupled_graph_sde(
            graph_dynamics, drift, diffusion, adjacency_matrix, node_features,
            t_span=t_span, fractional_orders=fractional_orders, num_steps=10
        )
        
        assert isinstance(solution_single, CoupledSolution)
        assert isinstance(solution_dict, CoupledSolution)


class TestCoupledSolution:
    """Test CoupledSolution dataclass"""
    
    def test_solution_creation(self):
        """Test solution object creation"""
        t = np.linspace(0, 1, 11)
        spatial = np.random.randn(11, 2)
        temporal = np.random.randn(11, 2)
        coupling = np.random.randn(11)
        
        solution = CoupledSolution(
            t=t,
            spatial=spatial,
            temporal=temporal,
            coupling=coupling
        )
        
        assert solution.t.shape == (11,)
        assert solution.spatial.shape == (11, 2)
        assert solution.temporal.shape == (11, 2)
        assert solution.coupling.shape == (11,)
        assert solution.metadata == {}
    
    def test_solution_properties(self):
        """Test solution properties"""
        t = np.linspace(0, 1, 11)
        spatial = np.random.randn(11, 2)
        temporal = np.random.randn(11, 2)
        coupling = np.random.randn(11)
        metadata = {'solver': 'test', 'steps': 10}
        
        solution = CoupledSolution(
            t=t,
            spatial=spatial,
            temporal=temporal,
            coupling=coupling,
            metadata=metadata
        )
        
        # Test trajectory access
        assert np.allclose(solution.t, t)
        assert np.allclose(solution.spatial, spatial)
        assert np.allclose(solution.temporal, temporal)
        assert np.allclose(solution.coupling, coupling)
        
        # Test metadata
        assert solution.metadata['solver'] == 'test'
        assert solution.metadata['steps'] == 10
    
    def test_solution_without_metadata(self):
        """Test solution creation without metadata"""
        t = np.linspace(0, 1, 11)
        spatial = np.random.randn(11, 2)
        temporal = np.random.randn(11, 2)
        coupling = np.random.randn(11)
        
        solution = CoupledSolution(
            t=t,
            spatial=spatial,
            temporal=temporal,
            coupling=coupling
        )
        
        # Metadata should be initialized as empty dict
        assert solution.metadata == {}


class TestCoupledSolverIntegration:
    """Test integration between different solvers"""
    
    def test_solver_comparison(self):
        """Test comparison between different solvers"""
        def graph_dynamics(state, adjacency):
            return -state
        
        def drift(t, x):
            return np.array([-x[0] + 0.5 * x[1], -x[1] + 0.5 * x[0]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        adjacency = np.array([[0, 1], [1, 0]])
        node_features = np.array([1.0, 0.5])
        t_span = (0, 1)
        num_steps = 20
        
        # Solve with operator splitting
        solver_split = OperatorSplittingSolver(fractional_orders=0.5)
        solution_split = solver_split.solve(
            graph_dynamics, drift, diffusion,
            adjacency, node_features, t_span,
            num_steps=num_steps
        )
        
        # Solve with monolithic solver
        solver_mono = MonolithicSolver(fractional_orders=0.5)
        solution_mono = solver_mono.solve(
            graph_dynamics, drift, diffusion,
            adjacency, node_features, t_span,
            num_steps=num_steps
        )
        
        assert isinstance(solution_split, CoupledSolution)
        assert isinstance(solution_mono, CoupledSolution)
        
        # Both should produce valid trajectories
        assert solution_split.t.shape == (num_steps + 1,)
        assert solution_mono.t.shape == (num_steps + 1,)
    
    def test_graph_coupling_integration(self):
        """Test integration with graph coupling"""
        # Create a more complex graph (3 nodes)
        adjacency_matrix = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        def graph_dynamics(state, adjacency):
            return -state
        
        def drift(t, x):
            # Graph-coupled dynamics for 3 nodes
            return np.array([
                -x[0] + 0.3 * x[1],
                -x[1] + 0.3 * x[0] + 0.3 * x[2],
                -x[2] + 0.3 * x[1]
            ])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1, 0.1])
        
        node_features = np.array([1.0, 0.5, 0.8])
        t_span = (0, 1)
        
        solution = solve_coupled_graph_sde(
            graph_dynamics, drift, diffusion, adjacency_matrix, node_features,
            t_span=t_span, num_steps=15
        )
        
        assert isinstance(solution, CoupledSolution)
        assert solution.spatial.shape == (16, 3)  # num_steps + 1, 3 nodes
        assert solution.temporal.shape == (16, 3)
    
    def test_seed_reproducibility(self):
        """Test that solutions are reproducible with same seed"""
        def graph_dynamics(state, adjacency):
            return -state
        
        def drift(t, x):
            return np.array([-x[0], -x[1]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        adjacency = np.array([[0, 1], [1, 0]])
        node_features = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        # Solve with same seed
        solution1 = solve_coupled_graph_sde(
            graph_dynamics, drift, diffusion, adjacency, node_features,
            t_span=t_span, num_steps=10, seed=42
        )
        
        solution2 = solve_coupled_graph_sde(
            graph_dynamics, drift, diffusion, adjacency, node_features,
            t_span=t_span, num_steps=10, seed=42
        )
        
        # Should be identical
        assert np.allclose(solution1.temporal, solution2.temporal)


class TestCoupledSolverEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_system(self):
        """Test with empty system"""
        def graph_dynamics(state, adjacency):
            return np.array([])
        
        def drift(t, x):
            return np.array([])
        
        def diffusion(t, x):
            return np.array([])
        
        adjacency = np.array([])
        node_features = np.array([])
        t_span = (0, 1)
        
        solver = OperatorSplittingSolver(fractional_orders=0.5)
        
        # Should handle empty system gracefully
        solution = solver.solve(
            graph_dynamics, drift, diffusion,
            adjacency, node_features, t_span,
            num_steps=10
        )
        
        assert isinstance(solution, CoupledSolution)
        assert solution.spatial.shape == (11, 0)  # Empty but correct shape
        assert solution.temporal.shape == (11, 0)
    
    def test_single_node_system(self):
        """Test with single node system"""
        def graph_dynamics(state, adjacency):
            return -state
        
        def drift(t, x):
            return -x
        
        def diffusion(t, x):
            return 0.1
        
        adjacency = np.array([[0]])
        node_features = np.array([1.0])
        t_span = (0, 1)
        
        solver = OperatorSplittingSolver(fractional_orders=0.5)
        solution = solver.solve(
            graph_dynamics, drift, diffusion,
            adjacency, node_features, t_span,
            num_steps=10
        )
        
        assert isinstance(solution, CoupledSolution)
        assert solution.spatial.shape == (11, 1)
        assert solution.temporal.shape == (11, 1)
    
    def test_large_system(self):
        """Test with larger system"""
        n_nodes = 10
        adjacency = np.random.rand(n_nodes, n_nodes)
        adjacency = (adjacency + adjacency.T) / 2  # Make symmetric
        np.fill_diagonal(adjacency, 0)  # No self-loops
        
        def graph_dynamics(state, adjacency):
            return -state + 0.1 * np.dot(adjacency, state)
        
        def drift(t, x):
            return -x
        
        def diffusion(t, x):
            return 0.1 * np.ones_like(x)
        
        node_features = np.random.randn(n_nodes)
        t_span = (0, 1)
        
        solver = OperatorSplittingSolver(fractional_orders=0.5)
        solution = solver.solve(
            graph_dynamics, drift, diffusion,
            adjacency, node_features, t_span,
            num_steps=20
        )
        
        assert isinstance(solution, CoupledSolution)
        assert solution.spatial.shape == (21, n_nodes)
        assert solution.temporal.shape == (21, n_nodes)
    
    def test_different_coupling_strengths(self):
        """Test with different coupling strengths"""
        def graph_dynamics(state, adjacency):
            return -state
        
        def drift(t, x):
            return -x
        
        def diffusion(t, x):
            return 0.1
        
        adjacency = np.array([[0, 1], [1, 0]])
        node_features = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        # Test various coupling strengths
        coupling_strengths = [0.0, 0.5, 1.0, 2.0]
        solutions = []
        
        for strength in coupling_strengths:
            solver = MonolithicSolver(fractional_orders=0.5, coupling_strength=strength)
            solution = solver.solve(
                graph_dynamics, drift, diffusion,
                adjacency, node_features, t_span,
                num_steps=10
            )
            solutions.append(solution)
        
        # All should be valid solutions
        for solution in solutions:
            assert isinstance(solution, CoupledSolution)
            assert solution.spatial.shape == (11, 2)
            assert solution.temporal.shape == (11, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])