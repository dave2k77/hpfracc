"""Independent algebraic residuals for stable-weight float32 integration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.numerical.implicit_validation import independent_residuals
from hpfracc.solvers import ImplicitPredictorCorrector, simulate
from hpfracc.solvers.predictor_corrector import _history_dot


def _decay(t, state, params, *, rng_key=None, inputs=None):
    return -params * state


def test_stable_float32_stiff_decay_meets_independent_residual():
    """Algebraic acceptance is separate from the coarse startup's IVP error."""
    with jax.enable_x64(False):
        times = jnp.linspace(0., 1., 257, dtype=jnp.float32)
        initial = jnp.ones(1, dtype=jnp.float32)
        order = jnp.asarray(.5, dtype=jnp.float32)
        rate = jnp.asarray(30., dtype=jnp.float32)
        result = simulate(
            model=_decay, ts=times, initial_state=initial, params=rate,
            solver=ImplicitPredictorCorrector(
                dt=1/256, order=order, newton_iterations=8,  # type: ignore[arg-type]
                newton_rtol=1e-5, newton_atol=1e-7, weight_evaluation="stable",
            ),
        )
        assert result.latent_state.dtype == jnp.float32
        assert _decay(times[0], initial, rate).dtype == jnp.float32
    assert result.solver_info is not None
    assert result.solver_info.diagnostics["newton_converged"]
    assert np.all(np.isfinite(np.asarray(result.latent_state)))
    reference_states = np.asarray(result.latent_state, dtype=np.float64)
    residuals = independent_residuals(
        "half_decay", np.asarray(times), reference_states,
        30., .5, np.ones(1, dtype=np.float64), 1e-5, 1e-7,
    )
    assert np.max(residuals) <= 1.


@pytest.mark.parametrize("shape", [(257,), (257, 1), (257, 2, 3)])
def test_history_reduction_retains_small_positive_contributions(shape):
    """Full history must not discard a representable aggregate of small terms."""
    with jax.enable_x64(False):
        history = jnp.full(shape, 2.**-25, dtype=jnp.float32).at[0].set(1.)
        weights = jnp.ones(257, dtype=jnp.float32)
        actual = jax.jit(_history_dot)(weights, history)
        assert actual.dtype == jnp.float32
    expected = np.sum(np.asarray(history, dtype=np.float64), axis=0)
    # Case-specific absolute budget: eight float32 ulps at the unit scale.
    # This is not a bound for arbitrary ill-conditioned sums.
    np.testing.assert_allclose(actual, expected, rtol=0.,
                               atol=float(8*np.finfo(np.float32).eps))
