"""Recursive SOE application and an opt-in scaled positive-tail fit.

All numerical imports are lazy. The legacy fitter remains in kernels.py so its
released objective and iteration rule can be used as a compatibility reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Any


@dataclass(frozen=True, slots=True)
class _SOEFit:
    rhos: Any
    coefficients: Any
    n: int
    kernel: str
    method: str
    exact_head: bool


def _validate_soe_config(n_poles: int, t_max: float, method: str) -> None:
    if isinstance(n_poles, bool) or not isinstance(n_poles, Integral) or n_poles < 1:
        raise ValueError("soe_poles / n_poles must be a positive integer.")
    if not isfinite(float(t_max)) or t_max <= 0:
        raise ValueError("soe_t_max / t_max must be finite and positive.")
    if method not in ("legacy", "scaled_tail"):
        raise ValueError(f"Unknown soe_fit: {method!r}; use 'legacy' or 'scaled_tail'.")


def _fit_soe(alpha: Any, n: int, *, n_poles: int, t_max: float,
             kernel: str, method: str = "legacy") -> _SOEFit:
    _validate_soe_config(n_poles, t_max, method)
    if n < 1 or kernel not in ("l1", "gl"):
        raise ValueError("Expected positive fit length and kernel 'l1' or 'gl'.")
    if method == "legacy":
        from hpfracc.ops.kernels import _soe_components

        rhos, coefficients = _soe_components(
            alpha, n, n_poles=n_poles, t_max=t_max, kernel=kernel,
        )
        return _SOEFit(rhos, coefficients, n, kernel, method, False)
    return _fit_scaled_tail(alpha, n, int(n_poles), kernel)


def _fit_scaled_tail(alpha: Any, n: int, n_poles: int, kernel: str) -> _SOEFit:
    """Fit positive tails with O(poles^2) forward fitting workspace.

    L1 fits b[k] for k>=1. GL fits cumulative weights g[k], where
    g[0]=1 and g[k]=g[k-1]*(k-alpha)/k, then acts on increments including x[0].
    This avoids fitting a signed, cancellation-sensitive GL derivative kernel.
    Both retain lag zero exactly and fit sum(c[j]*rho[j]**(k-1)) for k>=1.

    The Gram matrix is a geometric sum; the RHS is accumulated in a scan. Column
    normalization and a row-sum Lipschitz bound give a non-expansive projected
    gradient step in exact arithmetic. 400 fixed iterations do not certify
    convergence or fit error.
    The fixed dimensionless pole range is [1/(16*n), 8] in decay rates.
    """
    jax, jnp = _jax(), _jnp()
    dtype = jnp.result_type(alpha, float)
    a = jnp.asarray(alpha, dtype=dtype)
    # These constants depend on shape/configuration/dtype, not on alpha. Evaluate
    # them identically in eager and traced calls: float32 constant folding of
    # exp/log otherwise changes slow poles by an ulp and perturbs long fits.
    with jax.ensure_compile_time_eval():
        rates = jnp.exp(jnp.linspace(jnp.log(jnp.asarray(1 / (16 * n), dtype=dtype)),
                                     jnp.log(jnp.asarray(8., dtype=dtype)), n_poles))
        rhos = jnp.exp(-rates)
    target_kernel = "l1" if kernel == "l1" else "gl_prefix"
    if n == 1:
        return _SOEFit(rhos, jnp.zeros(n_poles, dtype=dtype), n,
                       target_kernel, "scaled_tail", True)

    # Use the actual rounded decay factors, also used in recursive application.
    # If a very slow pole rounds to one, use the exact finite geometric limit.
    with jax.ensure_compile_time_eval():
        log_product = jnp.log(rhos[:, None]) + jnp.log(rhos[None, :])
        decays = log_product < 0
        safe_log = jnp.where(decays, log_product, -1.)
        gram = jnp.where(decays, jnp.expm1((n - 1) * safe_log) / jnp.expm1(safe_log),
                         jnp.asarray(n - 1, dtype=dtype))
        norms = jnp.sqrt(jnp.diag(gram))
        normalized = gram / (norms[:, None] * norms[None, :])
        lipschitz = jnp.max(jnp.sum(jnp.abs(normalized), axis=1))

    def accumulate(k: Any, state: Any) -> Any:
        rhs, basis, previous = state
        lag = jnp.asarray(k + 1, dtype=dtype)
        if kernel == "l1":
            target = lag**(1 - a) * jnp.expm1((1 - a) * jnp.log1p(1 / lag))
        else:
            target = previous * (lag - a) / lag
        return rhs + basis * target, basis * rhos, target

    rhs, _, _ = jax.lax.fori_loop(
        0, n - 1, accumulate,
        (jnp.zeros(n_poles, dtype=dtype), jnp.ones(n_poles, dtype=dtype),
         jnp.asarray(1., dtype=dtype)),
    )
    projected_rhs = rhs / norms

    def update(_: Any, coefficients: Any) -> Any:
        return jnp.maximum(coefficients - (
            normalized @ coefficients - projected_rhs
        ) / lipschitz, 0.)

    normalized_coefficients = jax.lax.fori_loop(
        0, 400, update, jnp.zeros(n_poles, dtype=dtype),
    )
    return _SOEFit(rhos, normalized_coefficients / norms, n,
                   target_kernel, "scaled_tail", True)


def _apply_soe(values: Any, fit: _SOEFit) -> Any:
    """Apply a fitted kernel with a poles-by-features recursive history state."""
    jax, jnp = _jax(), _jnp()
    signal = jnp.asarray(values)
    if fit.kernel == "gl_prefix":
        signal = jnp.concatenate((signal[:1], signal[1:] - signal[:-1]), axis=0)
    flat = jnp.reshape(signal, (signal.shape[0], -1))
    initial = jnp.zeros((fit.rhos.shape[0], flat.shape[1]),
                        dtype=jnp.result_type(signal, fit.rhos, fit.coefficients))

    def step(history: Any, value: Any) -> Any:
        updated = fit.rhos[:, None] * history + value[None, :]
        out = (value + fit.coefficients @ history if fit.exact_head
               else fit.coefficients @ updated)
        return updated, out

    _, result = jax.lax.scan(step, initial, flat)
    return jnp.reshape(result, signal.shape)


def _kernel_errors(fit: _SOEFit, alpha: Any) -> tuple[Any, Any, Any]:
    """Stream actual residuals, avoiding cancellation in Gram-based objectives."""
    jax, jnp = _jax(), _jnp()
    dtype = fit.rhos.dtype
    a = jnp.asarray(alpha, dtype=dtype)

    def step(k: Any, state: Any) -> Any:
        basis, previous, error_sq, target_sq, maximum = state
        lag = jnp.asarray(k, dtype=dtype)
        safe_lag = jnp.where(k > 0, lag, 1.)
        if fit.kernel == "l1":
            target = safe_lag**(1 - a) * jnp.expm1((1 - a) * jnp.log1p(1 / safe_lag))
        else:
            shift = 0 if fit.kernel == "gl_prefix" else 1
            target = previous * (lag - shift - a) / safe_lag
        target = jnp.where(k == 0, 1., target)
        fitted = fit.coefficients @ basis
        if fit.exact_head:
            fitted = jnp.where(k == 0, 1., fitted)
            basis = jnp.where(k == 0, basis, basis * fit.rhos)
        else:
            basis = basis * fit.rhos
        error = jnp.abs(fitted - target)
        return (basis, target, error_sq + error**2, target_sq + target**2,
                jnp.maximum(maximum, error))

    zero, one = jnp.asarray(0., dtype=dtype), jnp.asarray(1., dtype=dtype)
    _, _, error_sq, target_sq, maximum = jax.lax.fori_loop(
        0, fit.n, step, (jnp.ones_like(fit.rhos), one, zero, zero, zero),
    )
    relative = jnp.sqrt(error_sq / target_sq)
    finite = (jnp.all(jnp.isfinite(fit.coefficients)) & jnp.all(jnp.isfinite(fit.rhos))
              & jnp.isfinite(relative) & jnp.isfinite(maximum))
    return maximum, relative, finite


def _soe_info(fit: _SOEFit, alpha: Any) -> dict[str, Any]:
    maximum, relative, finite = _kernel_errors(fit, alpha)

    def export(value: Any, *, boolean: bool = False) -> Any:
        try:
            concrete = bool(value) if boolean else float(value)
        except (TypeError, ValueError):
            return value
        return concrete if boolean or isfinite(concrete) else None

    return {
        "soe_fit": fit.method, "soe_application": "recursive",
        "soe_target_kernel": fit.kernel, "soe_exact_lag_zero": fit.exact_head,
        "soe_fit_samples": fit.n,
        "soe_kernel_max_abs_error": export(maximum),
        "soe_kernel_relative_l2_error": export(relative),
        "soe_fit_finite": export(finite, boolean=True),
    }


def _jax() -> Any:
    import jax

    return jax


def _jnp() -> Any:
    import jax.numpy as jnp

    return jnp
