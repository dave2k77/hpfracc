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

    When ``weights`` is 2-D with shape ``(n, m)`` the columns are *per-feature*
    fractional-order weights (vector / per-state orders): column ``j`` is applied
    to feature ``j`` independently via a vmap over the feature axis.
    """

    jnp = _jnp()
    n = weights.shape[0]
    original_shape = values.shape
    values_2d = jnp.reshape(values, (n, -1))
    if weights.ndim == 1:
        history = _lower_triangular_history_matrix(weights)
        out_2d = jnp.tensordot(history, values_2d, axes=([1], [0]))
        return jnp.reshape(out_2d, original_shape)

    # Per-feature weights: weights[:, j] convolves values_2d[:, j].
    jax = _jax()

    def _per_feature(w_col: Any, v_col: Any) -> Any:
        history = _lower_triangular_history_matrix(w_col)
        return history @ v_col

    out_2d = jax.vmap(_per_feature, in_axes=(1, 1), out_axes=1)(weights, values_2d)
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

    value_pad = jnp.concatenate(
        [values_2d, jnp.zeros((fft_len - n, m), dtype=values_2d.dtype)],
        axis=0,
    )
    fx = jnp.fft.rfft(value_pad, axis=0)

    if weights.ndim == 1:
        # Shared weights broadcast across every feature column.
        weight_pad = jnp.concatenate([weights, jnp.zeros((fft_len - n,))])
        fw = jnp.fft.rfft(weight_pad, axis=0)[:, None]
    else:
        # Per-feature weights ``(n, m)``: pad and transform each column, then
        # multiply elementwise so column j convolves feature j independently.
        weight_pad = jnp.concatenate(
            [weights, jnp.zeros((fft_len - n, weights.shape[1]))], axis=0
        )
        fw = jnp.fft.rfft(weight_pad, axis=0)

    # Real FFT along the leading (time) axis.  The product gives the linear
    # convolution because both operands are zero-padded to fft_len >= 2*n - 1.
    conv_2d = jnp.fft.irfft(fw * fx, fft_len, axis=0)[:n]

    return jnp.reshape(conv_2d, original_shape)


def _short_memory_convolution(
    values: Any,
    weights: Any,
    *,
    window_steps: int,
) -> Any:
    """Short-memory truncated causal convolution.

    Only the most recent ``window_steps`` weights are retained; earlier lags are
    treated as zero. A scan of shifted, weighted signals uses
    ``O(n * window * m)`` work for ``m`` features without constructing an
    ``n * n`` history matrix or using the native batched-convolution path.
    The input/output remain length ``n``; this is a batch operator, not a streaming
    interface. Weights may contain just the retained window or a longer history.

    This evaluates the same truncated discrete sum as the former dense path up
    to floating-point accumulation error. Truncation error versus full history
    depends on the discarded weights and signal; no universal relative bound is
    implied. The full-history default is unchanged.

    The truncation window is measured in time steps, not physical time.  Callers
    that want a physical horizon should convert ``t_window / dt`` and round.
    """

    jax = _jax()
    jnp = _jnp()
    n = values.shape[0]
    window = max(1, min(int(window_steps), n, weights.shape[0]))
    values_2d = jnp.reshape(values, (n, -1))
    retained = weights[:window]

    indices = jnp.arange(n)

    def add_lag(total: Any, lag: Any) -> tuple[Any, None]:
        source = indices - lag
        shifted = jnp.where(
            (source >= 0)[:, None], values_2d[jnp.maximum(source, 0)], 0,
        )
        # A scalar weight broadcasts over all features; per-state weights have
        # one entry per flattened feature. Complex signals retain their dtype.
        return total + shifted * retained[lag], None

    # Batched jnp.convolve can fail natively on Windows CPU in float64 after
    # mixed buffer allocations (see the pure-JAX diagnostic reproducer). This
    # algebraically identical scan is backend-independent and also safe under
    # an outer vmap, which could batch even an individually mapped convolution.
    initial = jnp.zeros(values_2d.shape, dtype=jnp.result_type(values, weights))
    result, _ = jax.lax.scan(add_lag, initial, jnp.arange(window))
    return jnp.reshape(result, values.shape)


def _soe_weights(
    alpha: float,
    n: int,
    *,
    n_poles: int,
    t_max: float,
    kernel: str = "l1",
) -> Any:
    """Materialize legacy fitted weights for reference/compatibility checks."""

    rhos, coefficients = _soe_components(
        alpha, n, n_poles=n_poles, t_max=t_max, kernel=kernel,
    )
    jnp = _jnp()
    ks = jnp.arange(n, dtype=jnp.result_type(float))
    return (rhos[None, :] ** ks[:, None]) @ coefficients


def _soe_components(
    alpha: float,
    n: int,
    *,
    n_poles: int,
    t_max: float,
    kernel: str = "l1",
) -> tuple[Any, Any]:
    """Legacy SOE fit, retained without changing its objective or updates.

    The fractional operator is defined by a discrete causal weight vector
    ``w_k`` (L1 increments ``b_k`` for Caputo, binomial GL weights for RL/GL).
    This routine fits an exponential sum

    ``w_k ~= sum_{j=1}^{M} c_j * rho_j^k``

    for ``k = 0, ..., n-1`` by choosing ``rho_j = exp(-lambda_j * dt)`` on a
    logarithmic grid. L1 uses projected non-negative least squares; GL uses
    regularized normal equations with signed coefficients. Returns decay factors
    and coefficients so callers can apply the fit recursively. The legacy fit
    still builds a length-by-poles design matrix and has limited accuracy and
    conditioning; the named ``scaled_tail`` fit lives in ``ops._soe``.

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
    # GL has a positive first weight and negative tail, so use signed least-squares
    # solve.  L1 weights are non-negative, so keep the projected NNLS solve.
    if kernel == "l1":
        coefficients = _nnls_projected(A, target, n_iter=400, step=0.005)
    else:
        coefficients = _least_squares_solve(A, target)

    return rhos, coefficients


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

    def update(_: Any, coefficients: Any) -> Any:
        grad = AtA @ coefficients - Atb
        return jnp.clip(coefficients - step * grad, 0.0, None)

    # Preserve the legacy fixed-step iteration rule without unrolling hundreds
    # of updates and their derivatives into the compiled graph.
    return _jax().lax.fori_loop(0, n_iter, update, x)


def _soe_convolution(
    values: Any,
    weights: Any,
) -> Any:
    """Legacy dense reference application of a materialized SOE weight vector.

    The SOE approximation is fully captured by the discrete ``weights`` array;
    public operators now use the recursive application in ``ops._soe``.
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


def _jax() -> Any:
    try:
        import jax
    except ModuleNotFoundError as exc:
        msg = (
            "JAX is required for hpfracc.ops numerical kernels. Install the "
            "package with its runtime dependencies before calling this function."
        )
        raise ModuleNotFoundError(msg) from exc
    return jax
