"""Fractional differential equation solver namespace."""

from hpfracc.solvers.base import SimulationResult, SolverInfo
from hpfracc.solvers.predictor_corrector import (
    AdaptivePredictorCorrector,
    ImplicitPredictorCorrector,
    NonUniformPredictorCorrector,
    PredictorCorrector,
    simulate,
)

__all__ = [
    "AdaptivePredictorCorrector",
    "ImplicitPredictorCorrector",
    "NonUniformPredictorCorrector",
    "PredictorCorrector",
    "SimulationResult",
    "SolverInfo",
    "simulate",
]
