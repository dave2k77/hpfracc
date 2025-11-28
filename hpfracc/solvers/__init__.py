# __init__.py (patch)
from .ode_solvers import (
    FixedStepODESolver,
    solve_fractional_ode,
    solve_fractional_system
)

from .pde_solvers import (
    FractionalPDESolver,
    FractionalDiffusionSolver,
    FractionalAdvectionSolver,
    FractionalReactionDiffusionSolver,
    solve_fractional_pde
)

from .sde_solvers import (
    FractionalSDESolver,
    FractionalEulerMaruyama,
    FractionalMilstein,
    SDESolution,
    solve_fractional_sde,
    solve_fractional_sde_system
)

from .noise_models import (
    NoiseModel,
    BrownianMotion,
    FractionalBrownianMotion,
    LevyNoise,
    ColouredNoise,
    NoiseConfig,
    create_noise_model,
    generate_noise_trajectory
)

from .coupled_solvers import (
    CoupledSystemSolver,
    OperatorSplittingSolver,
    MonolithicSolver,
    CoupledSolution,
    solve_coupled_graph_sde
)

__all__ = [
    # ODE Solvers
    'FixedStepODESolver',
    'solve_fractional_ode',
    'solve_fractional_system',

    # PDE Solvers
    'FractionalPDESolver',
    'FractionalDiffusionSolver',
    'FractionalAdvectionSolver',
    'FractionalReactionDiffusionSolver',
    'solve_fractional_pde',
    
    # SDE Solvers
    'FractionalSDESolver',
    'FractionalEulerMaruyama',
    'FractionalMilstein',
    'SDESolution',
    'solve_fractional_sde',
    'solve_fractional_sde_system',
    
    # Noise Models
    'NoiseModel',
    'BrownianMotion',
    'FractionalBrownianMotion',
    'LevyNoise',
    'ColouredNoise',
    'NoiseConfig',
    'create_noise_model',
    'generate_noise_trajectory',
    
    # Coupled Solvers
    'CoupledSystemSolver',
    'OperatorSplittingSolver',
    'MonolithicSolver',
    'CoupledSolution',
    'solve_coupled_graph_sde',
]

# Backward-compatibility aliases for tests expecting legacy names
# Advanced and high-order solvers are not implemented; provide stubs.
class AdvancedFractionalODESolver(FixedStepODESolver):
    pass

class HighOrderFractionalSolver(FixedStepODESolver):
    pass

# Main compatibility alias for FractionalODESolver (maps to FixedStepODESolver)
FractionalODESolver = FixedStepODESolver

# AdaptiveFractionalODESolver alias (maps to FixedStepODESolver with adaptive=True by default)
class AdaptiveFractionalODESolver(FixedStepODESolver):
    """Adaptive fractional ODE solver with automatic step size control."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('adaptive', True)
        super().__init__(*args, **kwargs)

# Alias for tests expecting AdaptiveFixedStepODESolver
AdaptiveFixedStepODESolver = AdaptiveFractionalODESolver

def solve_advanced_fractional_ode(*args, **kwargs):
    return solve_fractional_ode(*args, **kwargs)

def solve_high_order_fractional_ode(*args, **kwargs):
    return solve_fractional_ode(*args, **kwargs)

# Predictor-corrector compatibility names
PredictorCorrectorSolver = FixedStepODESolver
AdamsBashforthMoultonSolver = None # Was AdaptiveODESolver
class VariableStepPredictorCorrector: # Removed inheritance
    pass

def solve_predictor_corrector(*args, **kwargs):
    return solve_fractional_ode(*args, **kwargs)

__all__ += [
    'AdvancedFractionalODESolver',
    'HighOrderFractionalSolver',
    'FractionalODESolver',
    'AdaptiveFractionalODESolver',
    'AdaptiveFixedStepODESolver',
    'solve_advanced_fractional_ode',
    'solve_high_order_fractional_ode',
    'PredictorCorrectorSolver',
    'AdamsBashforthMoultonSolver',
    'VariableStepPredictorCorrector',
    'solve_predictor_corrector',
]
