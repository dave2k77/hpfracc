"""Experimental adaptive product integration with a prescribed-forcing bound.

The bound concerns interpolation error at accepted nodes, conditional on a
supplied global C2 curvature bound. It excludes floating-point and RHS-evaluation
error and does not apply to general state-dependent FDE dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Any

from hpfracc.ops.orders import as_order
from hpfracc.solvers.base import SimulationResult, SolverInfo


@dataclass(frozen=True, slots=True)
class BoundedForcing:
    """Prescribed q(t, params, *, rng_key=None, inputs=None) and curvature bound.

    ``curvature_bound`` is a concrete nonnegative scalar/array bounding
    ``abs(d2q/dt2)`` componentwise over the whole interval and parameter domain
    used. q must be twice continuously differentiable in time. The wrapper never
    forwards the evolving state to q. Bounds are caller-supplied assumptions,
    not inferred or certified by sampling derivatives. Experimental.
    """

    forcing: Any
    curvature_bound: Any

    def __post_init__(self) -> None:
        import numpy as np

        if not callable(self.forcing):
            raise TypeError("forcing must be callable.")
        try:
            value = np.asarray(self.curvature_bound)
        except (TypeError, ValueError) as exc:
            message = "curvature_bound must be concrete model configuration."
            raise TypeError(message) from exc
        if (not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value)
                or value.size == 0 or not np.all(np.isfinite(value))
                or np.any(value < 0)):
            raise ValueError(
                "curvature_bound must contain finite nonnegative real values."
            )

        def freeze(item):
            if isinstance(item, list):
                return tuple(freeze(x) for x in item)
            return float(item)

        object.__setattr__(self, "curvature_bound", freeze(value.tolist()))

    def __call__(self, t, state, params, *, rng_key=None, inputs=None):
        del state
        return self.forcing(t, params, rng_key=rng_key, inputs=inputs)


@dataclass(frozen=True, slots=True)
class AdaptiveForcingQuadrature:
    """Adaptive Caputo product integration for ``BoundedForcing`` only.

    Controls a conditional full-history *nodal interpolation-error* envelope
    using positive power-kernel masses and the supplied curvature bound. A
    conservative numerical-state scale also supports relative tolerances.
    Controller/mesh decisions are frozen for gradients. This does not certify
    roundoff error, dense output, or arbitrary nonlinear/weak-startup FDEs.

    Real float32/float64 bounds and initial states are required. ``ts`` supplies
    integration bounds; results have ``max_steps+2`` slots and an accepted-prefix
    mask. Incomplete runs retain their accepted states and interpolation bounds.
    """

    order: Any
    rtol: float = 0.
    atol: float = 1e-5
    max_steps: int = 1024
    first_step: float | None = None
    safety: float = .8
    min_factor: float = .2
    max_factor: float = 5.

    def __post_init__(self) -> None:
        as_order(self.order)
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, Integral):
            raise ValueError("max_steps must be a positive integer.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")
        for name in (
            "rtol", "atol", "first_step", "safety", "min_factor", "max_factor",
        ):
            value = getattr(self, name)
            if value is not None and not isfinite(float(value)):
                raise ValueError(f"Expected finite {name}.")
        if self.rtol < 0 or self.atol <= 0:
            raise ValueError("Expected rtol >= 0 and atol > 0.")
        if self.first_step is not None and self.first_step <= 0:
            raise ValueError("Expected positive first_step.")
        if not 0 < self.safety <= 1 or not 0 < self.min_factor < 1 < self.max_factor:
            raise ValueError(
                "Expected 0 < safety <= 1 and 0 < min_factor < 1 < max_factor."
            )


def _curvature_array(bound, shape, dtype):
    """Validate/broadcast concrete bounds and round a narrowing conversion upward."""
    import jax.numpy as jnp
    import numpy as np

    exact = np.broadcast_to(np.asarray(bound, dtype=np.float64), shape)
    with np.errstate(over="ignore", under="ignore"):
        rounded = exact.astype(np.dtype(dtype))
        rounded = np.where(
            rounded.astype(np.float64) < exact,
            np.nextafter(rounded, np.asarray(np.inf, dtype=dtype)), rounded,
        )
    if (not np.all(np.isfinite(rounded))
            or np.any((exact > 0) & (rounded < np.finfo(dtype).tiny))):
        raise ValueError(
            "curvature_bound is not representable at the working precision."
        )
    return jnp.asarray(rounded, dtype=dtype)


def _simulate_bounded_forcing(*, model, ts, solver, initial_state, params,
                              rng_key=None, inputs=None) -> SimulationResult:
    import jax
    import jax.numpy as jnp

    from hpfracc.solvers.predictor_corrector import (
        _diagnostic_value,
        _nonuniform_pece_weights,
        _validate_increasing_grid,
    )

    if not isinstance(model, BoundedForcing):
        raise TypeError("AdaptiveForcingQuadrature requires a BoundedForcing model.")
    times_in, y0 = jnp.asarray(ts), jnp.asarray(initial_state)
    _validate_increasing_grid(times_in)
    supported = (jnp.dtype("float32"), jnp.dtype("float64"))
    if times_in.dtype not in supported or y0.dtype not in supported or y0.size == 0:
        raise ValueError(
            "Bounded forcing requires real float32/float64 bounds and states."
        )
    alpha = jnp.asarray(as_order(solver.order))
    if alpha.ndim != 0:
        raise NotImplementedError("Bounded forcing supports a scalar fractional order.")

    def force(t):
        return jnp.asarray(model.forcing(t, params, rng_key=rng_key, inputs=inputs))

    initial_f = force(times_in[0])
    if jnp.iscomplexobj(initial_f):
        raise ValueError("Bounded forcing requires real forcing values.")
    dtype = jnp.result_type(times_in.dtype, y0.dtype, initial_f, alpha, float)
    times_in, y0, alpha = times_in.astype(dtype), y0.astype(dtype), alpha.astype(dtype)
    initial_f = jnp.broadcast_to(initial_f, y0.shape).astype(dtype)
    curvature = _curvature_array(model.curvature_bound, y0.shape, dtype)
    t0, end = times_in[0], times_in[-1]
    span = end - t0
    budget, length = int(solver.max_steps), int(solver.max_steps) + 2
    indices = jnp.arange(length)
    rtol = jnp.asarray(solver.rtol, dtype=dtype)
    atol = jnp.asarray(solver.atol, dtype=dtype)
    h0 = (.01 * span if solver.first_step is None
          else jnp.asarray(solver.first_step, dtype))
    h0 = jnp.minimum(h0, span)
    minimum_step = jnp.asarray(1e-12, dtype=dtype) * jnp.abs(span)
    node_times = jnp.zeros(length, dtype=dtype).at[0].set(t0)
    values = jnp.zeros((length, *y0.shape), dtype=dtype).at[0].set(y0)
    rhs_history = jnp.zeros_like(values).at[0].set(initial_f)
    bounds = jnp.zeros_like(values)
    initial_finite = (jnp.all(jnp.isfinite(y0)) & jnp.all(jnp.isfinite(initial_f))
                      & jnp.isfinite(span) & jnp.isfinite(rtol) & jnp.isfinite(atol)
                      & (atol > 0))
    sg = jax.lax.stop_gradient

    def attempt(carry):
        (count, current, h, nodes, states, rhs, bound_history,
         rejected, done, bad, small) = carry
        target = jnp.minimum(sg(current) + sg(h), end)
        actual_h = target - sg(current)
        resolved = target > current
        step = count + 1
        trial_nodes = nodes.at[step].set(target)
        pred, corr, diagonal = _nonuniform_pece_weights(
            step, sg(trial_nodes), indices, alpha, dtype,
        )
        new_f = jnp.broadcast_to(force(target), y0.shape).astype(dtype)
        candidate = y0 + jnp.tensordot(corr, rhs, axes=([0], [0])) + diagonal * new_f

        # |q-Lq| <= M*h_j^2/8 on every panel. Positive predictor weights are
        # integral kernel masses / Gamma(alpha), including the entire old history.
        widths = jnp.concatenate((trial_nodes[1:] - trial_nodes[:-1],
                                   jnp.zeros(1, dtype)))
        pieces = jnp.where(indices < step, pred * widths**2 / 8, 0.)
        old_mass = jnp.sum(jnp.where(indices < count, pieces, 0.))
        old_bound = jnp.where(curvature == 0, 0., curvature * old_mass)
        new_bound = jnp.where(curvature == 0, 0., curvature * pieces[count])
        envelope = sg(old_bound + new_bound)
        # This conservative scale implies E <= atol+rtol*|y_exact| in exact
        # arithmetic whenever E bounds the interpolation error and this test passes.
        scale = (atol + rtol * jnp.abs(candidate)) / (1 + rtol)
        valid_scale = jnp.all(jnp.isfinite(scale) & (scale > 0))
        safe_scale = jnp.where(scale > 0, scale, 1.)
        ratio = sg(jnp.max(envelope / safe_scale))
        finite = (jnp.all(jnp.isfinite(candidate)) & jnp.all(jnp.isfinite(new_f))
                  & jnp.all(jnp.isfinite(envelope)) & jnp.isfinite(ratio) & valid_scale)
        bad = bad | (resolved & ~finite)
        small = small | ~resolved | (finite & (ratio > 1) & (actual_h <= minimum_step))
        accept = resolved & finite & (ratio <= 1)
        nodes = nodes.at[step].set(jnp.where(accept, target, nodes[step]))
        states = states.at[step].set(jnp.where(accept, candidate, states[step]))
        rhs = rhs.at[step].set(jnp.where(accept, new_f, rhs[step]))
        bound_history = bound_history.at[step].set(jnp.where(
            accept, envelope, bound_history[step],
        ))
        count = count + accept.astype(jnp.int32)
        rejected = rejected + (~accept).astype(jnp.int32)
        current = jnp.where(accept, target, current)
        done = done | (accept & (target >= end))

        # Allocate remaining tolerance to the *new* panel. Its known bound has
        # h^(alpha+2) scaling. Recheck the full-history envelope on the next trial;
        # no global scaling law is assumed for the old contribution or relative scale.
        available = scale - old_bound
        local_ratio = jnp.where(new_bound > 0,
                                available / jnp.where(new_bound > 0, new_bound, 1.),
                                solver.max_factor**(alpha + 2))
        factor = solver.safety * jnp.maximum(jnp.min(local_ratio), 0.)**(
            1 / (alpha + 2)
        )
        factor = jnp.where(jnp.any(new_bound > 0), factor, solver.max_factor)
        factor = sg(jnp.clip(factor, solver.min_factor, solver.max_factor))
        next_h = jnp.minimum(actual_h * factor,
                             jnp.maximum(end - current, minimum_step))
        h = jnp.where(done | bad | small, h, next_h).astype(dtype)
        return (count, current, h, nodes, states, rhs, bound_history,
                rejected, done, bad, small)

    def iteration(carry, _):
        active = ~(carry[-1] | carry[-2] | carry[-3])
        return jax.lax.cond(active, attempt, lambda state: state, carry), None

    initial = (jnp.asarray(0, jnp.int32), t0, h0, node_times, values, rhs_history,
               bounds, jnp.asarray(0, jnp.int32),
               jnp.asarray(times_in.size == 1) & initial_finite, ~initial_finite,
               jnp.asarray(False))
    final, _ = jax.lax.scan(iteration, initial, None, length=budget)
    count, current, _, nodes, states, _, bounds, rejected, done, bad, small = final
    valid = indices <= count
    mask = valid.reshape((-1,) + (1,) * y0.ndim)
    nodes = jnp.where(valid, nodes, nodes[count])
    states = jnp.where(mask, states, states[count])
    bounds = jnp.where(mask, bounds, bounds[count])
    exhausted = ~(done | bad | small)
    warnings = []
    for flag, message in (
        (bad, "Bounded-forcing integration encountered nonfinite data/normalization."),
        (small, "Bounded-forcing integration cannot resolve an acceptable step."),
        (exhausted, "Bounded-forcing integration exhausted max_steps."),
    ):
        if _diagnostic_value(flag) is True:
            warnings.append(message)
    info = SolverInfo(
        name="adaptive_forcing_quadrature", method="caputo_linear_forcing_full_history",
        fractional_order=_diagnostic_value(alpha), n_steps=_diagnostic_value(count),
        diagnostics={
            "history": "full", "grid": "adaptive", "estimator": "curvature_bound",
            "estimator_stability": "experimental", "error_norm": "componentwise_bound",
            "bound_scope": "conditional_nodal_interpolation_error",
            "roundoff_included": False,
            "curvature_bound": _diagnostic_value(curvature),
            "interpolation_error_bound": _diagnostic_value(bounds),
            "rtol": float(solver.rtol), "atol": float(solver.atol),
            "max_steps": budget, "n_nodes": _diagnostic_value(count + 1),
            "n_accepted": _diagnostic_value(count),
            "n_rejected": _diagnostic_value(rejected),
            "n_attempted": _diagnostic_value(count + rejected),
            "success": _diagnostic_value(done),
            "reached_t_end": _diagnostic_value(done),
            "nonfinite_detected": _diagnostic_value(bad),
            "step_size_underflow": _diagnostic_value(small),
            "max_steps_reached": _diagnostic_value(exhausted),
            "valid_mask": _diagnostic_value(valid),
            "last_time": _diagnostic_value(current),
            "requested_t_end": _diagnostic_value(end),
            "minimum_step": _diagnostic_value(minimum_step),
        }, warnings=tuple(warnings),
    )
    return SimulationResult(ts=nodes, latent_state=states, solver_info=info)
