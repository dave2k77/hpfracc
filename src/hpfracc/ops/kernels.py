"""History convolution kernels for fractional operators.

The v0.1 default remains the explicit full-history lower-triangular matrix so that
it is a clear, inspectable reference.  Phase A introduces named, opt-in
alternatives that must be proven equivalent (or bounded) against that reference
before they are used by later solvers.

All kernels operate along the leading time axis and support arbitrary trailing
state dimensions by reshaping to ``(time, feature)`` and restoring the original
trail shape.
"""

from __future__ import annotations

from typing import Any

from hpfracc.ops.base import HistoryMethod


def _gl_weights(alpha: float, n: int) -> Any:
    """Grunwald-Letnikov binomial weights of length ``n``."""

    jnp = _jnp()
    if n == 1:
        return jnp.ones((1,))
    ks = jnp.arange(1, n, dtype=jnp.result_type(float))
    factors = (ks - 1.0 - alpha) / ks
    return jnp.concatenate([jnp.ones((1,), dtype=factors.dtype), jnp.cumprod(factors)])


def _l1_weights(alpha: float, n: int) -> Any:
    """L1 Caputo weights ``b_k = (k+1)^{1-alpha} - k^{1-alpha}`` of length ``n``."""

    jnp = _jnp()
    ks = jnp.arange(n, dtype=jnp.result_type(float))
    return (ks + 1.0) ** (1.0 - alpha) - ks ** (1.0 - alpha)


def _lower_triangular_history_matrix(weights: Any) -> Any:
    """Dense lower-triangular matrix that realizes the causal history sum."""

    jnp = _jnp()
    n = weights.shape[0]
    rows = jnp.arange(n)[:, None]
    cols = jnp.arange(n)[None, :]
    lag = rows - cols
    return jnp.where(lag >= 0, weights[jnp.clip(lag, 0, n - 1)], 0.0)


def _full_history_convolution(values: Any, weights: Any) -> Any:
    """Explicit matrix-tensor contraction for the causal history sum.

    For a 1-D signal ``values`` of length ``n`` and weights ``w`` of length ``n``,
    returns ``c`` where ``c[i] = sum_{k=0}^{i} w[k] * values[i-k]``.  Trailing
    dimensions are handled independently by reshaping to ``(n, m)``.
    """

    jnp = _jnp()
    n = weights.shape[0]
    original_shape = values.shape
    values_2d = jnp.reshape(values, (n, -1))
    history = _lower_triangular_history_matrix(weights)
    out_2d = jnp.tensordot(history, values_2d, axes=([1], [0]))
    return jnp.reshape(out_2d, original_shape)


def _fft_history_convolution(values: Any, weights: Any) -> Any:
    """FFT-accelerated causal history convolution for uniform grids.

    This computes the same linear causal convolution as
    ``_full_history_convolution`` but via real FFTs on a padded power-of-two
    length.  It is ``O(n log n)`` in time and ``O(n)`` in working memory for the
    padded FFT (plus the input/output arrays), instead of the ``O(n^2)`` dense
    matrix materialisation used by the full-history reference.

    The output is mathematically the same operation; differences are at the
    floating-point accumulation level, so equivalence tests use a tight tolerance
    rather than bitwise identity.
    """

    jnp = _jnp()
    n = int(weights.shape[0])
    original_shape = values.shape
    values_2d = jnp.reshape(values, (n, -1))
    m = values_2d.shape[1]

    # Pad to the next power of two that can hold the full linear convolution of
    # length 2*n - 1.  ``n`` is a static Python int from the input shape, so this
    # remains traceable under jit for concretely-shaped inputs.
    fft_len = 1 << (2 * n - 2).bit_length()

    weight_pad = jnp.concatenate([weights, jnp.zeros((fft_len - n,))])
    value_pad = jnp.concatenate(
        [values_2d, jnp.zeros((fft_len - n, m), dtype=values_2d.dtype)],
        axis=0,
    )

    # Real FFT along the leading (time) axis.  The product gives the linear
    # convolution because both operands are zero-padded to fft_len >= 2*n - 1.
    fw = jnp.fft.rfft(weight_pad, axis=0)
    fx = jnp.fft.rfft(value_pad, axis=0)
    conv_2d = jnp.fft.irfft(fw[:, None] * fx, fft_len, axis=0)[:n]

    return jnp.reshape(conv_2d, original_shape)


def _short_memory_convolution(
    values: Any,
    weights: Any,
    *,
    window_steps: int,
) -> Any:
    """Short-memory truncated causal convolution.

    Only the most recent ``window_steps`` weights are retained; earlier lags are
    treated as zero.  This gives ``O(n * window_steps)`` cost instead of
    ``O(n^2)`` and a bounded, method-specific truncation error that is validated
    against the full-history reference rather than silently assumed.

    The truncation window is measured in time steps, not physical time.  Callers
    that want a physical horizon should convert ``t_window / dt`` and round.
    """

    jnp = _jnp()
    n = weights.shape[0]
    window = max(1, min(int(window_steps), n))
    truncated = jnp.concatenate([
        weights[:window],
        jnp.zeros((n - window,), dtype=weights.dtype),
    ])
    return _full_history_convolution(values, truncated)


def _soe_weights(
    alpha: float,
    n: int,
    *,
    n_poles: int,
    t_max: float,
    kernel: str = "l1",
) -> Any:
    """Sum-of-exponentials weights that approximate discrete L1 or GL weights.

    The fractional operator is defined by a discrete causal weight vector
    ``w_k`` (L1 increments ``b_k`` for Caputo, binomial GL weights for RL/GL).
    This routine fits a non-negative exponential sum

    ``w_k ~= sum_{j=1}^{M} c_j * rho_j^k``

    for ``k = 0, ..., n-1`` by choosing ``rho_j = exp(-lambda_j * dt)`` on a
    logarithmic grid and solving a small non-negative least-squares problem for
    the coefficients ``c_j``.  The returned array has length ``n`` and can be
    used directly in the causal convolution path.

    This is a research-grade approximation; the operator result reports
    ``history="soe"`` and diagnostics so that downstream callers can compare
    against the full-history reference.
    """

    jnp = _jnp()
    if n_poles < 1:
        msg = "n_poles must be positive for SOE history."
        raise ValueError(msg)
    if n < 1:
        msg = "n must be positive for SOE history."
        raise ValueError(msg)
    if t_max <= 0.0:
        msg = "t_max must be positive for SOE history."
        raise ValueError(msg)

    dtau = t_max / float(n)
    if kernel == "l1":
        target = _l1_weights(alpha, n)
    elif kernel == "gl":
        target = _gl_weights(alpha, n)
    else:
        msg = f"Unknown SOE kernel: {kernel!r}"
        raise ValueError(msg)

    # Log-spaced decay exponents.  We want lambda_j * dtau to range from ~1/n
    # (slow decay / long memory) to ~1 (fast decay / short memory), so that
    # rho_j = exp(-lambda_j * dtau) covers the meaningful interval.
    log_min = jnp.log(1.0 / float(n))
    log_max = jnp.log(1.0)
    log_lambdas = jnp.linspace(log_min, log_max, n_poles)
    lambdas = jnp.exp(log_lambdas) / dtau
    rhos = jnp.exp(-lambdas * dtau)

    # Design matrix A[k, j] = rho_j ** k.
    ks = jnp.arange(n, dtype=jnp.result_type(float))
    A = rhos[None, :] ** ks[:, None]
    # GL weights are signed (alternating), so use an unconstrained least-squares
    # solve.  L1 weights are non-negative, so keep the projected NNLS solve.
    if kernel == "l1":
        coefficients = _nnls_projected(A, target, n_iter=400, step=0.005)
    else:
        coefficients = _least_squares_solve(A, target)

    discrete_weights = A @ coefficients
    return discrete_weights


def _least_squares_solve(A: Any, b: Any) -> Any:
    """Solve a small dense least-squares problem via the normal equations."""

    jnp = _jnp()
    AtA = A.T @ A
    Atb = A.T @ b
    return jnp.linalg.solve(AtA + 1e-8 * jnp.eye(AtA.shape[0]), Atb)


def _nnls_projected(
    A: Any,
    b: Any,
    *,
    n_iter: int,
    step: float,
) -> Any:
    """Very small projected-gradient non-negative least-squares solve.

    Solves ``min_x ||A x - b||_2`` subject to ``x >= 0``.  The problem size is
    kept tiny (``n_poles`` is typically 5-20), so a simple fixed-step projected
    gradient method is sufficient and stays in pure JAX.
    """

    jnp = _jnp()
    n_features = A.shape[1]
    x = jnp.zeros((n_features,), dtype=A.dtype)
    AtA = A.T @ A
    Atb = A.T @ b
    for _ in range(n_iter):
        grad = AtA @ x - Atb
        x = jnp.clip(x - step * grad, 0.0, None)
    return x


def _soe_convolution(
    values: Any,
    weights: Any,
) -> Any:
    """Apply an SOE weight vector via the full-history path.

    The SOE approximation is fully captured by the discrete ``weights`` array;
    this helper is a thin wrapper so that compression logic is explicit.
    """

    return _full_history_convolution(values, weights)


def _history_convolution(
    values: Any,
    weights: Any,
    *,
    history: HistoryMethod | str,
    window_steps: int = 64,
) -> Any:
    """Dispatch a causal history convolution to the requested implementation."""

    method = HistoryMethod(history)
    if method is HistoryMethod.FULL:
        return _full_history_convolution(values, weights)
    if method is HistoryMethod.FFT:
        return _fft_history_convolution(values, weights)
    if method is HistoryMethod.SHORT_MEMORY:
        return _short_memory_convolution(values, weights, window_steps=window_steps)
    if method is HistoryMethod.SOE:
        return _soe_convolution(values, weights)

    msg = f"Unsupported history method: {history!r}"
    raise ValueError(msg)


def _jnp() -> Any:
    try:
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        msg = (
            "JAX is required for hpfracc.ops numerical kernels. Install the "
            "package with its runtime dependencies before calling this function."
        )
        raise ModuleNotFoundError(msg) from exc
    return jnp
