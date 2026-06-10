"""Lightweight probabilistic calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

from hpfracc.nn import NeuralFODE
from hpfracc.typing import PyTree


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Grid-calibration result for a scalar parameter."""

    parameter_name: str
    parameter_grid: PyTree
    log_likelihoods: PyTree
    posterior_weights: PyTree
    best_index: int
    best_params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PosteriorPredictiveResult:
    """Posterior predictive trajectories and summary statistics."""

    trajectories: PyTree
    mean: PyTree
    lower: PyTree
    upper: PyTree
    weights: PyTree


def gaussian_log_likelihood(
    predicted: PyTree,
    observed: PyTree,
    *,
    noise_scale: float,
) -> Any:
    """Compute an independent Gaussian log likelihood."""

    if not float(noise_scale) > 0.0:
        msg = f"Expected positive noise_scale, got {noise_scale}."
        raise ValueError(msg)

    jnp = _jnp()
    residual = jnp.asarray(observed) - jnp.asarray(predicted)
    variance = float(noise_scale) ** 2
    return -0.5 * jnp.sum((residual**2) / variance + jnp.log(2.0 * pi * variance))


def normalize_log_weights(log_weights: PyTree) -> Any:
    """Normalize log weights with a numerically stable softmax."""

    jax = _jax()
    return jax.nn.softmax(_jnp().asarray(log_weights))


def weighted_quantile(values: PyTree, weights: PyTree, q: float) -> Any:
    """Quantile of a discrete weighted distribution along the leading axis.

    ``values`` has the weighted (grid/sample) axis first, shape ``(n, ...)``;
    ``weights`` is a length-``n`` non-negative vector (it is renormalized here).

    This uses the right-continuous inverse empirical CDF: the ``q``-quantile is
    the smallest value whose cumulative weight, in ascending value order, is at
    least ``q``. It is the *exact* quantile of the discrete grid-weighted
    predictive distribution and makes no smoothness assumption between grid
    points -- so a near-degenerate posterior collapses the quantile onto the
    dominant trajectory rather than interpolating toward improbable neighbours.

    Unlike an unweighted quantile, this respects ``weights``: reweighting the
    grid changes the result.
    """

    if not 0.0 <= float(q) <= 1.0:
        msg = f"Expected quantile level q in [0, 1], got {q}."
        raise ValueError(msg)

    jnp = _jnp()
    values = jnp.asarray(values)
    weights = jnp.asarray(weights)
    total = jnp.sum(weights)
    normalized = weights / total

    order = jnp.argsort(values, axis=0)
    sorted_values = jnp.take_along_axis(values, order, axis=0)
    weight_shape = (values.shape[0],) + (1,) * (values.ndim - 1)
    broadcast_weights = jnp.broadcast_to(normalized.reshape(weight_shape), values.shape)
    sorted_weights = jnp.take_along_axis(broadcast_weights, order, axis=0)

    cumulative = jnp.cumsum(sorted_weights, axis=0)
    # First sorted position whose cumulative weight reaches q. The small
    # tolerance keeps exact cumulative-weight boundaries on the lower side.
    at_or_above = cumulative >= (float(q) - 1e-12)
    index = jnp.argmax(at_or_above, axis=0)
    return jnp.take_along_axis(sorted_values, index[None, ...], axis=0)[0]


def grid_calibrate_scalar(
    model: NeuralFODE,
    *,
    ts: PyTree,
    initial_state: PyTree,
    observations: PyTree,
    parameter_name: str,
    parameter_grid: PyTree,
    fixed_params: dict[str, Any] | None = None,
    noise_scale: float = 0.05,
) -> CalibrationResult:
    """Evaluate a scalar parameter grid under a Gaussian observation model."""

    jnp = _jnp()
    fixed = {} if fixed_params is None else dict(fixed_params)
    grid = jnp.asarray(parameter_grid)
    log_likelihoods = []
    for value in grid:
        params = {**fixed, parameter_name: value}
        result = model(ts=ts, initial_state=initial_state, params=params)
        predicted = result.observed if result.observed is not None else result.latent_state
        log_likelihoods.append(
            gaussian_log_likelihood(
                predicted,
                observations,
                noise_scale=noise_scale,
            )
        )

    log_likelihood_array = jnp.asarray(log_likelihoods)
    weights = normalize_log_weights(log_likelihood_array)
    best_index = int(jnp.argmax(log_likelihood_array))
    best_params = {**fixed, parameter_name: grid[best_index]}
    return CalibrationResult(
        parameter_name=parameter_name,
        parameter_grid=grid,
        log_likelihoods=log_likelihood_array,
        posterior_weights=weights,
        best_index=best_index,
        best_params=best_params,
    )


def posterior_predictive(
    model: NeuralFODE,
    *,
    ts: PyTree,
    initial_state: PyTree,
    parameter_name: str,
    parameter_grid: PyTree,
    weights: PyTree,
    fixed_params: dict[str, Any] | None = None,
    interval_mass: float = 0.9,
) -> PosteriorPredictiveResult:
    """Generate weighted posterior predictive trajectories from a scalar grid."""

    if not 0.0 < float(interval_mass) < 1.0:
        msg = f"Expected interval_mass in (0, 1), got {interval_mass}."
        raise ValueError(msg)

    jnp = _jnp()
    fixed = {} if fixed_params is None else dict(fixed_params)
    grid = jnp.asarray(parameter_grid)
    normalized = normalize_log_weights(jnp.log(jnp.asarray(weights)))
    trajectories = []
    for value in grid:
        params = {**fixed, parameter_name: value}
        result = model(ts=ts, initial_state=initial_state, params=params)
        predicted = result.observed if result.observed is not None else result.latent_state
        trajectories.append(predicted)

    stacked = jnp.stack(trajectories, axis=0)
    reshape = (normalized.shape[0],) + (1,) * (stacked.ndim - 1)
    mean = jnp.sum(stacked * normalized.reshape(reshape), axis=0)
    alpha = (1.0 - float(interval_mass)) / 2.0
    # Weighted quantiles so the band reflects the posterior, not the raw grid
    # spread. An unweighted jnp.quantile here would ignore ``normalized``.
    lower = weighted_quantile(stacked, normalized, alpha)
    upper = weighted_quantile(stacked, normalized, 1.0 - alpha)
    return PosteriorPredictiveResult(
        trajectories=stacked,
        mean=mean,
        lower=lower,
        upper=upper,
        weights=normalized,
    )


def _jax() -> Any:
    try:
        import jax
    except ModuleNotFoundError as exc:
        msg = (
            "JAX is required for hpfracc.prob calibration utilities. Install "
            "the package with its runtime dependencies before calling this "
            "function."
        )
        raise ModuleNotFoundError(msg) from exc
    return jax


def _jnp() -> Any:
    try:
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        msg = (
            "JAX is required for hpfracc.prob calibration utilities. Install "
            "the package with its runtime dependencies before calling this "
            "function."
        )
        raise ModuleNotFoundError(msg) from exc
    return jnp
