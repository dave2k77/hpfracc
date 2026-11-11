"""Private host-only PL Volterra-defect diagnostics; never a certificate.

This benchmark module is not a solver, controller, public API or differentiable
numerical path. Ordinary NumPy/SciPy float64 evaluation is deliberately separate
from the working precision of the stored trajectory. Input, evaluation and output
roundoff are not enclosed. No result certifies a numerical solution.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy
from scipy.special import gamma, hyp2f1, poch

from hpfracc.config import Provenance, RuntimeTarget


@dataclass(frozen=True, slots=True)
class DefectEvaluation:
    evaluation_times: np.ndarray
    reconstruction: np.ndarray
    fractional_rhs: np.ndarray
    defect: np.ndarray
    provenance: Provenance
    arithmetic_verified: bool = field(default=False, init=False)
    total_numerical_certified: bool = field(default=False, init=False)

    def to_dict(self):
        return {
            "evaluation_times": self.evaluation_times.tolist(),
            "reconstruction": self.reconstruction.tolist(),
            "fractional_rhs": self.fractional_rhs.tolist(),
            "defect": self.defect.tolist(),
            "provenance": self.provenance.to_dict(),
            "arithmetic_verified": self.arithmetic_verified,
            "total_numerical_certified": self.total_numerical_certified,
        }


def _real_array(name, value):
    raw = np.asarray(value)
    if (raw.dtype.kind != "f" or raw.dtype.itemsize not in (4, 8)
            or not np.all(np.isfinite(raw))):
        raise ValueError(f"Expected finite real float32/float64 {name}")
    return np.array(raw, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class _Inputs:
    ts: np.ndarray
    states: np.ndarray
    targets: np.ndarray
    order: float
    matrix: np.ndarray
    initial: np.ndarray
    powers: np.ndarray
    coefficients: np.ndarray
    input_dtype: str


def _prepare(ts, states, evaluation_times, order, matrix, initial_state,
             power_exponents, power_coefficients):
    input_dtype = str(np.asarray(states).dtype)
    ts = _real_array("ts", ts)
    states = _real_array("states", states)
    targets = _real_array("evaluation_times", evaluation_times)
    alpha = _real_array("order", order)
    matrix = _real_array("matrix", matrix)
    initial = _real_array("initial_state", initial_state)
    powers = _real_array("power_exponents", power_exponents)
    if alpha.ndim != 0 or not 0 < float(alpha) < 1:
        raise ValueError("Expected scalar 0 < order < 1")
    if ts.ndim != 1 or ts.size < 2 or not np.all(ts[1:] > ts[:-1]):
        raise ValueError("Expected at least two strictly increasing times")
    if states.ndim != 2 or states.shape[0] != ts.size or states.shape[1] < 1:
        raise ValueError("Expected states shape (time, state_dim), state_dim >= 1")
    d = states.shape[1]
    if matrix.shape != (d, d):
        raise ValueError("Expected square matrix matching state_dim")
    if initial.shape != (d,) or not np.array_equal(initial, states[0]):
        raise ValueError("Expected initial_state to match the first stored state")
    if (targets.ndim != 1 or targets.size == 0
            or np.any(targets < ts[0]) or np.any(targets > ts[-1])):
        raise ValueError("Expected evaluation times within the stored horizon")
    if powers.ndim != 1 or np.any(powers < 0):
        raise ValueError("Expected one-dimensional nonnegative power_exponents")
    if power_coefficients is None:
        coefficients = np.empty((0, d), dtype=np.float64)
    else:
        coefficients = _real_array("power_coefficients", power_coefficients)
    if coefficients.shape != (powers.size, d):
        raise ValueError("Expected power_coefficients shape (power, state_dim)")
    return _Inputs(ts, states, targets, float(alpha), matrix, initial,
                   powers, coefficients, input_dtype)


def _checked(context, operation, *operands):
    """Run inspectable float64 operations, not opaque BLAS/library reductions.

    errstate is a secondary guard: postconditions below also detect silent zero
    products, ratios and positive-base powers. These are range checks, not error
    bounds on rounding, cancellation, summation or special-function evaluation.
    """
    values = tuple(np.asarray(x, dtype=np.float64) for x in operands)
    name = operation.__name__
    if not all(np.all(np.isfinite(x)) for x in values):
        raise FloatingPointError(f"{context}: {name}: Nonfinite operand")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            result = operation(*values)
    except FloatingPointError as exc:
        raise FloatingPointError(f"{context}: {name}: {exc}") from exc
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f"{context}: {name}: Nonfinite result")
    return result


def _add(left, right, context):
    # Exact signed cancellation is legitimate; a zero sum is not underflow proof.
    return _checked(context, np.add, left, right)


def _subtract(left, right, context):
    return _checked(context, np.subtract, left, right)


def _multiply(left, right, context):
    result = _checked(context, np.multiply, left, right)
    nonzero = (np.asarray(left) != 0) & (np.asarray(right) != 0)
    if np.any(nonzero & (result == 0)):
        raise FloatingPointError(f"{context}: multiply: underflow to zero")
    return result


def _divide(numerator, denominator, context):
    result = _checked(context, np.divide, numerator, denominator)
    if np.any((np.asarray(numerator) != 0) & (result == 0)):
        raise FloatingPointError(f"{context}: divide: underflow to zero")
    return result


def _power(base, exponent, context):
    result = _checked(context, np.power, base, exponent)
    if np.any((np.asarray(base) > 0) & (result == 0)):
        raise FloatingPointError(f"{context}: power: underflow to zero")
    return result


def _positive(value, context):
    """Check known-positive basis/special values, without certifying accuracy."""
    if not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise FloatingPointError(f"{context}: expected positive finite value")
    return value


def _power_integral_ratio(beta, alpha):
    """Gamma(beta+1)/Gamma(beta+alpha+1), without subtracting large log-gammas."""
    context = "forcing gamma ratio"
    shifted = _add(beta, 1., context)
    denominator = _positive(_checked(context, poch, shifted, alpha), context)
    return _positive(_divide(1., denominator, context), context)


def _forcing_factors(tau, beta, alpha, context):
    """Shared positive factors; unsafe time powers are not rescued or dropped."""
    exponent = _add(alpha, beta, context)
    time_power = _power(tau, exponent, context)
    return _multiply(time_power, _power_integral_ratio(beta, alpha), context)


def _feedback(states, matrix):
    """Row-stored A*Y: sum Y[i,k]*A[j,k] in increasing k, from exact zero.

    Each product is checked before it can be masked by another component. Fixed
    order checked addition permits genuine signed cancellation; no dot/matmul or
    sum-result zero test substitutes for checking the individual products.
    """
    result = np.zeros_like(states)
    for k in range(states.shape[1]):
        context = f"matrix feedback input component {k}"
        product = _multiply(states[:, k, None], matrix[None, :, k], context)
        result = _add(result, product, context)
    return result


def _interpolate(ts, states, targets):
    """Endpoint-distance barycentric PL evaluation, with exact knot identity."""
    indices = np.searchsorted(ts, targets, side="left")
    knots = ts[indices] == targets
    result = np.empty((targets.size, states.shape[1]), dtype=np.float64)
    result[knots] = states[indices[knots]]
    panels = indices - 1
    for j in np.unique(panels[~knots]):
        context = f"PL reconstruction panel {j}"
        selected = ~knots & (panels == j)
        t = targets[selected]
        width = _subtract(ts[j+1], ts[j], context)
        dl = _subtract(t, ts[j], context)
        dr = _subtract(ts[j+1], t, context)
        # Never form slopes or recover either weight as 1 minus the other.
        wl, wr = _divide(dr, width, context), _divide(dl, width, context)
        yl = _multiply(wl[:, None], states[j], context)
        yr = _multiply(wr[:, None], states[j+1], context)
        result[selected] = _add(yl, yr, context)
    return result


def _panel_weights(targets, left, right, a):
    """Positive integrals of the two local PL endpoint basis functions.

    Active weights use the independent right-target gap. Old-panel small-q and
    near-terminal branches retain their positive hyp2f1 and expm1/log1p formulas;
    the old age complement is independent too, never recovered as 1-q.
    """
    wl, wr = np.zeros_like(targets), np.zeros_like(targets)
    active = targets > left
    if not np.any(active):
        return wl, wr
    context = f"panel [{left!r}, {right!r}]"
    t = targets[active]
    width = _subtract(right, left, context)
    tau = _subtract(t, left, context)
    q = _divide(np.minimum(tau, width), tau, context)
    ap1 = _add(a, 1., context)
    normalization = _positive(_checked(context, gamma, ap1), context)
    mass = _positive(_divide(_power(tau, a, context), normalization, context),
                     context)
    balance = _divide(a, ap1, context)
    cl, cr = np.empty_like(tau), np.empty_like(tau)
    within = t <= right
    fraction = _divide(tau[within], width, context)
    remaining = _divide(_subtract(right, t[within], context), width, context)
    cl[within] = _multiply(
        mass[within],
        _add(remaining, _multiply(fraction, balance, context), context), context,
    )
    cr[within] = _divide(_multiply(mass[within], fraction, context), ap1, context)
    small = ~within & (q < .5)
    factor = _divide(_multiply(_multiply(mass[small], q[small], context), a,
                               context), 2., context)
    one_minus_a = _subtract(1., a, context)
    hl = _positive(_checked(context, hyp2f1, one_minus_a, 1., 3., q[small]),
                   context)
    hr = _positive(_checked(context, hyp2f1, one_minus_a, 2., 3., q[small]),
                   context)
    cl[small], cr[small] = (_multiply(factor, hl, context),
                            _multiply(factor, hr, context))
    near = ~within & ~small
    z = _positive(_divide(_subtract(t[near], right, context), tau[near], context),
                  context)
    logz = _checked(context, np.log, z)
    alogz = _multiply(a, logz, context)
    prefactor = _divide(mass[near], q[near], context)
    left_expm1 = _checked(context, np.expm1,
                          _subtract(alogz, _checked(context, np.log1p, a), context))
    cl[near] = _multiply(prefactor,
                         _add(balance, _multiply(z, left_expm1, context), context),
                         context)
    right_expm1 = _checked(context, np.expm1, _multiply(ap1, logz, context))
    cr[near] = _multiply(prefactor,
                         _add(-_checked(context, np.expm1, alogz),
                              _multiply(balance, right_expm1, context), context),
                         context)
    wl[active], wr[active] = _positive(cl, context), _positive(cr, context)
    return wl, wr


def _panel_ranges(edges, left, right, a):
    """Exact-model extrema: old weights decrease; active left weight has a peak."""
    ll, rl = _panel_weights(edges[:-1], left, right, a)
    lu, ru = _panel_weights(edges[1:], left, right, a)
    lower = [np.minimum(ll, lu), np.minimum(rl, ru)]
    upper = [np.maximum(ll, lu), np.maximum(rl, ru)]
    context = f"panel [{left!r}, {right!r}] active-left peak"
    peak = _multiply(a, _subtract(right, left, context), context)
    # Stay in offset space: left+alpha*width may round to left.
    tau_l = np.maximum(_subtract(edges[:-1], left, context), 0)
    tau_u = np.maximum(_subtract(edges[1:], left, context), 0)
    at_peak = (tau_l <= peak) & (peak <= tau_u)
    ap1 = _add(a, 1., context)
    normalization = _positive(_checked(context, gamma, ap1), context)
    peak_weight = _positive(_divide(_power(peak, a, context),
                                    _multiply(normalization, ap1, context), context),
                            context)
    upper[0][at_peak] = np.maximum(upper[0][at_peak], peak_weight)
    return lower, upper


def affine_power_defect(ts, states, evaluation_times, *, order, matrix,
                        initial_state, power_exponents=(), power_coefficients=None):
    """Evaluate Y-y0-I**alpha(A Y + sum c_beta*(t-t0)**beta).

    State shape is (time, state_dim); all old knots and the original lower
    terminal are retained. Nonnegative powers include weak forcing with beta<1.
    Inputs must be finite real float32/float64 arrays/scalars. Evaluation widens
    represented inputs to float64 before RHS arithmetic; it is host-only and
    does not verify arithmetic or provide a continuous-solution error bound.
    Unsafe intermediate underflow raises even if a rearranged final expression
    would be representable; no zero contribution or certificate is substituted.
    """
    p = _prepare(ts, states, evaluation_times, order, matrix, initial_state,
                 power_exponents, power_coefficients)
    ts, states, targets, a = p.ts, p.states, p.targets, p.order
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        reconstruction = _interpolate(ts, states, targets)
        rhs = np.zeros_like(reconstruction)
        coefficients = _feedback(states, p.matrix)
        # Fixed reduction order: panels in storage order, left then right;
        # forcing terms in supplied order from zero, then added to feedback.
        for j, (left, right) in enumerate(zip(ts[:-1], ts[1:], strict=True)):
            if not np.any(coefficients[j:j+2]):
                continue  # Exactly zero feedback only; no nonzero term is dropped.
            context = f"point feedback panel {j}"
            wl, wr = _panel_weights(targets, left, right, a)
            yl = _multiply(wl[:, None], coefficients[j], context)
            yr = _multiply(wr[:, None], coefficients[j+1], context)
            rhs = _add(rhs, _add(yl, yr, context), context)
        if p.powers.size:
            tau = _subtract(targets, ts[0], "point forcing clock")
            forcing = np.zeros_like(rhs)
            for k, (beta, coefficient) in enumerate(
                zip(p.powers, p.coefficients, strict=True),
            ):
                context = f"point forcing term {k}"
                factors = _forcing_factors(tau, beta, a, context)
                product = _multiply(factors[:, None], coefficient, context)
                forcing = _add(forcing, product, context)
            rhs = _add(rhs, forcing, "point feedback plus forcing")
        defect = _subtract(_subtract(reconstruction, p.initial, "point Y-y0"),
                           rhs, "point defect")
    if not all(np.all(np.isfinite(x)) for x in (reconstruction, rhs, defect)):
        raise FloatingPointError("Nonfinite defect evaluation")
    provenance = Provenance(
        backend="numpy/scipy", runtime_target=RuntimeTarget.CPU,
        config={"scope": "private PL affine-power defect diagnostic",
                "arithmetic": "float64; traps underflow; unverified",
                "input_state_dtype": p.input_dtype, "order": a,
                "numpy_version": np.__version__, "scipy_version": scipy.__version__},
    )
    return DefectEvaluation(targets, reconstruction, rhs, defect, provenance)


@dataclass(frozen=True, slots=True)
class DefectCells:
    """Floating diagnostics of an enclosing real-arithmetic expression.

    These computed endpoints need not enclose even a single exact residual:
    gamma/power/coefficient/summation roundoff is not budgeted or directed.
    """

    verification_times: np.ndarray
    model_lower: np.ndarray
    model_upper: np.ndarray
    provenance: Provenance
    arithmetic_verified: bool = field(default=False, init=False)
    total_numerical_certified: bool = field(default=False, init=False)

    def to_dict(self):
        return {
            "verification_times": self.verification_times.tolist(),
            "model_lower": self.model_lower.tolist(),
            "model_upper": self.model_upper.tolist(),
            "provenance": self.provenance.to_dict(),
            "arithmetic_verified": self.arithmetic_verified,
            "total_numerical_certified": self.total_numerical_certified,
        }


def affine_power_cells(ts, states, verification_mesh, *, order, matrix,
                       initial_state, power_exponents=(), power_coefficients=None,
                       max_work=1_000_000):
    """Evaluate unverified natural-extension model ranges on continuum cells.

    The supplied mesh must cover the whole original horizon; all original knots
    are added. Old positive-basis weights decrease; the active left weight has
    an interior maximum at alpha times the panel width. Coefficient signs select
    their extrema. No sampled maximum is substituted. Unsafe intermediate
    underflow raises rather than silently erasing a finite contribution. The
    resulting floating endpoints are NOT verified numerical enclosures, and no
    feedback propagation or solution-error interpretation is provided here.
    max_work caps cell-term-component evaluations and fails explicitly; it is
    not a measured memory/runtime limit and never authorizes truncating history.
    """
    p = _prepare(ts, states, verification_mesh, order, matrix, initial_state,
                 power_exponents, power_coefficients)
    if (p.targets.size < 2 or not np.all(p.targets[1:] > p.targets[:-1])
            or p.targets[0] != p.ts[0] or p.targets[-1] != p.ts[-1]):
        raise ValueError("Expected increasing verification mesh covering the horizon")
    edges = np.union1d(p.ts, p.targets)
    if type(max_work) is not int or max_work <= 0:
        raise ValueError("Expected positive integer max_work")
    term_count = 3 + 2 * (p.ts.size - 2) + p.powers.size
    work = (edges.size - 1) * term_count * p.states.shape[1]
    if work > max_work:
        raise ValueError(f"Cell work budget exhausted: {work} > {max_work}")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        curve = _interpolate(p.ts, p.states, edges)
        lower = _subtract(np.minimum(curve[:-1], curve[1:]), p.initial, "cell Y-y0")
        upper = _subtract(np.maximum(curve[:-1], curve[1:]), p.initial, "cell Y-y0")
        coefficients = _feedback(p.states, p.matrix)
        # Fixed order: stored panels, left then right basis, supplied powers.
        # Sign selection follows checked products, not a sum-result zero test.
        for j, (left, right) in enumerate(zip(p.ts[:-1], p.ts[1:], strict=True)):
            if not np.any(coefficients[j:j+2]):
                continue  # Exactly zero feedback only; retain every other panel.
            lows, highs = _panel_ranges(edges, left, right, p.order)
            pair = (-coefficients[j], -coefficients[j+1])
            for basis, (lo, hi, coefficient) in enumerate(
                zip(lows, highs, pair, strict=True),
            ):
                context = f"cell feedback panel {j} basis {basis}"
                x = _multiply(lo[:, None], coefficient, context)
                y = _multiply(hi[:, None], coefficient, context)
                lower = _add(lower, np.minimum(x, y), context)
                upper = _add(upper, np.maximum(x, y), context)
        if p.powers.size:
            tau = _subtract(edges, p.ts[0], "cell forcing clock")
            for k, (beta, coefficient) in enumerate(
                zip(p.powers, p.coefficients, strict=True),
            ):
                context = f"cell forcing term {k}"
                factors = _forcing_factors(tau, beta, p.order, context)
                product = _multiply(factors[:, None], -coefficient, context)
                lower = _add(lower, np.minimum(product[:-1], product[1:]), context)
                upper = _add(upper, np.maximum(product[:-1], product[1:]), context)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise FloatingPointError("Nonfinite cell-range evaluation")
    provenance = Provenance(
        backend="numpy/scipy", runtime_target=RuntimeTarget.CPU,
        config={"scope": "private PL affine-power cell diagnostic",
                "arithmetic": ("float64; traps underflow; unverified; "
                               "no outward rounding"),
                "range_method": ("positive panel-basis extrema and "
                                 "signed forcing powers"),
                "cell_term_components": work, "max_work": max_work,
                "input_state_dtype": p.input_dtype, "order": p.order,
                "numpy_version": np.__version__, "scipy_version": scipy.__version__},
    )
    return DefectCells(edges, lower, upper, provenance)
