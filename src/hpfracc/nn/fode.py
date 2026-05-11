"""Differentiable fractional ODE model wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hpfracc.solvers import PredictorCorrector, SimulationResult, simulate
from hpfracc.typing import DynamicsFn, PRNGKey, PyTree, StateTransform


@dataclass(frozen=True, slots=True)
class NeuralFODE:
    """Thin differentiable wrapper around a trainable FODE dynamics function.

    Parameters are explicit pytrees passed to ``simulate`` at call time. This
    keeps the v0.1 layer JAX-native without taking a dependency on a neural
    module framework before the core APIs stabilize.
    """

    dynamics: DynamicsFn
    solver: PredictorCorrector
    observe: StateTransform | None = None

    def __call__(
        self,
        *,
        ts: PyTree,
        initial_state: PyTree,
        params: PyTree,
        rng_key: PRNGKey | None = None,
        inputs: PyTree | None = None,
    ) -> SimulationResult:
        """Simulate the model and optionally transform latent states."""

        result = simulate(
            model=self.dynamics,
            ts=ts,
            solver=self.solver,
            initial_state=initial_state,
            params=params,
            rng_key=rng_key,
            inputs=inputs,
        )
        if self.observe is None:
            return result
        return SimulationResult(
            ts=result.ts,
            latent_state=result.latent_state,
            observed=self.observe(result.latent_state),
            solver_info=result.solver_info,
            metadata=result.metadata,
        )


def mse_loss(predicted: Any, target: Any) -> Any:
    """Mean-squared error for JAX arrays or array-like values."""

    jnp = _jnp()
    residual = jnp.asarray(predicted) - jnp.asarray(target)
    return jnp.mean(residual**2)


def trajectory_mse(
    model: NeuralFODE,
    *,
    ts: PyTree,
    initial_state: PyTree,
    params: PyTree,
    target: PyTree,
    rng_key: PRNGKey | None = None,
    inputs: PyTree | None = None,
) -> Any:
    """Compute MSE between a model trajectory and a target trajectory."""

    result = model(
        ts=ts,
        initial_state=initial_state,
        params=params,
        rng_key=rng_key,
        inputs=inputs,
    )
    predicted = result.observed if result.observed is not None else result.latent_state
    return mse_loss(predicted, target)


def sgd_step(params: PyTree, grads: PyTree, *, learning_rate: float) -> PyTree:
    """Apply one SGD step to a pytree of parameters."""

    jax = _jax()
    return jax.tree_util.tree_map(
        lambda param, grad: param - learning_rate * grad,
        params,
        grads,
    )


def _jax() -> Any:
    try:
        import jax
    except ModuleNotFoundError as exc:
        msg = (
            "JAX is required for hpfracc.nn differentiable models. Install the "
            "package with its runtime dependencies before calling this function."
        )
        raise ModuleNotFoundError(msg) from exc
    return jax


def _jnp() -> Any:
    try:
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        msg = (
            "JAX is required for hpfracc.nn differentiable models. Install the "
            "package with its runtime dependencies before calling this function."
        )
        raise ModuleNotFoundError(msg) from exc
    return jnp
