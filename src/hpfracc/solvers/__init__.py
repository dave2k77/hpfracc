"""Fractional differential equation solver namespace."""

from hpfracc.solvers.base import SimulationResult, SolverInfo
from hpfracc.solvers.predictor_corrector import (
    ImplicitPredictorCorrector,
    NonUniformPredictorCorrector,
    PredictorCorrector,
    simulate,
)

__all__ = [
    "ImplicitPredictorCorrector",
    "NonUniformPredictorCorrector",
    "PredictorCorrector",
    "SimulationResult",
    "SolverInfo",
    "simulate",
]
