"""Working-precision normalization for the supported fractional-order interval."""

from __future__ import annotations

from typing import Any

# (-1)**k * (zeta(k) - 1) / k, k=2,...,56, rounded from 80-digit
# reference values. No special-function or high-precision runtime dependency.
_LOG_GAMMA1P_COEFFICIENTS = (
    0.3224670334241132,
    -0.0673523010531981,
    0.020580808427784546,
    -0.007385551028673986,
    0.0028905103307415234,
    -0.001192753911703261,
    0.0005096695247430425,
    -0.00022315475845357939,
    9.945751278180853e-05,
    -4.492623673813314e-05,
    2.050721277567069e-05,
    -9.439488275268397e-06,
    4.374866789907488e-06,
    -2.039215753801366e-06,
    9.55141213040742e-07,
    -4.492469198764566e-07,
    2.1207184805554665e-07,
    -1.0043224823968099e-07,
    4.7698101693639804e-08,
    -2.2711094608943164e-08,
    1.0838659214896955e-08,
    -5.183475041970047e-09,
    2.4836745438024785e-09,
    -1.1921401405860912e-09,
    5.731367241678862e-10,
    -2.7595228851242334e-10,
    1.330476437424449e-10,
    -6.4229645638381e-11,
    3.1044247747322276e-11,
    -1.5021384080754142e-11,
    7.275974480239079e-12,
    -3.527742476575915e-12,
    1.711991790559618e-12,
    -8.315385841420285e-13,
    4.04220052528944e-13,
    -1.9664756310966165e-13,
    9.573630387838556e-14,
    -4.6640760264283744e-14,
    2.2737369600659724e-14,
    -1.1091399470834522e-14,
    5.413659156725363e-15,
    -2.643880017860995e-15,
    1.2918959062789966e-15,
    -6.315935504198448e-16,
    3.089316266963393e-16,
    -1.5117930628108198e-16,
    7.40148685695232e-17,
    -3.625218048120654e-17,
    1.7763568421861633e-17,
    -8.70763157479179e-18,
    4.270088559227004e-18,
    -2.0947604247944643e-18,
    1.0279842823787928e-18,
    -5.046468294792953e-19,
    2.4781763945937917e-19,
)


def _rgamma1p(order: Any) -> Any:
    """Evaluate 1/Gamma(1+order) for real 0 <= order <= 1.

    DLMF 5.7.3 gives log Gamma(1+a) = -log(1+a) + (1-gamma)*a
    + sum((-1)**k * (zeta(k)-1) * a**k/k, k>=2). Horner evaluation
    through degree 24 (float32) or 56 (float64) has an exact-arithmetic
    alternating remainder below 1.2e-9 or 1.3e-19 on this interval.
    Working-precision rounding is additional; this is not an error bound for
    the solver. See https://dlmf.nist.gov/5.7.E3.

    This avoids the cancellation of general-purpose float32 log-Gamma at
    shifted arguments. It preserves JAX tracing/gradients and never enables
    x64 or performs host evaluation of numerical inputs.
    """
    import jax.numpy as jnp

    a = jnp.asarray(order, dtype=jnp.result_type(order, float))
    degree = 56 if a.dtype.itemsize > 4 else 24
    coefficients = jnp.asarray(_LOG_GAMMA1P_COEFFICIENTS[:degree - 1], dtype=a.dtype)
    series = a * a * jnp.polyval(coefficients[::-1], a)
    log_reciprocal = jnp.log1p(a) - 0.42278433509846713 * a - series
    return jnp.exp(log_reciprocal)
