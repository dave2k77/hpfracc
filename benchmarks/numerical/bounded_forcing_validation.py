"""Conditional full-history forcing-envelope calibration on independent cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import gamma
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from benchmarks.numerical.adaptive_calibration import ROOT, _context
from hpfracc.config import current_provenance
from hpfracc.solvers import AdaptiveForcingQuadrature, BoundedForcing, simulate

CASES = ("affine", "quadratic", "cubic_blind_spot", "oscillatory", "vector_relative")
ORDERS = (.3, .5, .8)
TOLERANCES = (1e-3, 1e-4)
ROUNDOFF_SLACK = {"float32": 3e-6, "float64": 3e-12}


def curvature(case):
    if case == "affine":
        return 0.
    if case == "vector_relative":
        return (0., 2.)
    return 4. if case == "oscillatory" else 2.


def forcing(case, t, params, *, rng_key=None, inputs=None):
    del params, rng_key, inputs
    if case == "affine":
        return 1+2*t
    if case == "quadratic":
        return t*t
    if case == "cubic_blind_spot":
        return t*t-2*t**3/3
    if case == "oscillatory":
        return jnp.sin(2*t)
    return jnp.stack((1+2*t, t*t))


def reference(case, times, alpha):
    t = np.asarray(times, dtype=np.float64)
    affine = t**alpha/gamma(alpha+1)+2*t**(alpha+1)/gamma(alpha+2)
    quadratic = 2*t**(alpha+2)/gamma(alpha+3)
    if case == "affine":
        return affine
    if case == "quadratic":
        return quadratic
    if case == "cubic_blind_spot":
        return quadratic-4*t**(alpha+3)/gamma(alpha+4)
    if case == "vector_relative":
        return np.stack((1+affine, .5+quadratic), axis=-1)
    # Independent continuous integration; no discrete solver weights or trajectory
    # refinement. q(t*u) is integrated against (1-u)^(alpha-1) on [0,1].
    return np.asarray([
        time**alpha/gamma(alpha)*quad(
            lambda u, time=time: np.sin(2*time*u), 0., 1., weight="alg",
            wvar=(0., alpha-1), epsabs=1e-12, epsrel=1e-12,
        )[0] if time > 0 else 0. for time in t
    ])


def reference_envelope(times, alpha, bound):
    """Host integral of panelwise M*h^2/8, using actual accepted nodes."""
    t = np.asarray(times, dtype=np.float64)
    factors = [0.]
    for n in range(1, len(t)):
        width = np.diff(t[:n+1])
        upper, lower = t[n]-t[:n], t[n]-t[1:n+1]
        mass = (upper**alpha-lower**alpha) / gamma(alpha+1)
        factors.append(float(np.sum(width**2 * mass / 8)))
    curve = np.asarray(bound)
    return np.asarray(factors).reshape((-1,) + (1,) * curve.ndim) * curve


@lru_cache(maxsize=32)
def runner(case, atol, max_steps):
    def q(t, params, *, rng_key=None, inputs=None):
        return forcing(case, t, params, rng_key=rng_key, inputs=inputs)

    model = BoundedForcing(q, curvature(case))
    rtol = 1e-3 if case == "vector_relative" else 0.

    @jax.jit
    def run(initial, times, order):
        result = simulate(
            model=model, ts=times, initial_state=initial, params=None,
            solver=AdaptiveForcingQuadrature(order=order, atol=atol, rtol=rtol,
                                              max_steps=max_steps, first_step=1.),
        )
        d = result.solver_info.diagnostics
        keys = ("success", "n_nodes", "n_accepted", "n_rejected", "n_attempted",
                "valid_mask", "max_steps_reached", "nonfinite_detected",
                "step_size_underflow", "interpolation_error_bound")
        return result.ts, result.latent_state, {k: d[k] for k in keys}
    return run


@dataclass(frozen=True, slots=True)
class BoundRow:
    case: str
    dtype: str
    order: float
    atol: float
    rtol: float
    max_steps: int
    accepted: int
    rejected: int
    completed: bool
    max_steps_reached: bool
    nonfinite_detected: bool
    step_size_underflow: bool
    max_abs_error: float | None
    max_interpolation_bound: float | None
    max_bound_budget_ratio: float | None
    max_reference_scaled_error: float | None
    max_error_above_bound: float | None
    max_bound_reference_difference: float | None
    envelope_consistent: bool
    passed: bool


def summarize(output, case, dtype, alpha, atol, max_steps):
    times, values, d = jax.device_get(output)
    n = int(d["n_nodes"])
    accepted, rejected = int(d["n_accepted"]), int(d["n_rejected"])
    initial = np.asarray([1., .5] if case == "vector_relative" else 0., dtype=dtype)
    bound = np.asarray(d["interpolation_error_bound"])
    rtol = 1e-3 if case == "vector_relative" else 0.
    complete = bool(
        d["success"] and times.shape == (max_steps+2,)
        and values.shape == bound.shape == (max_steps+2, *initial.shape)
        and str(times.dtype) == str(values.dtype) == str(bound.dtype) == dtype
        and 2 <= n <= max_steps+1 and accepted == n-1 and rejected >= 0
        and accepted+rejected == int(d["n_attempted"]) <= max_steps
        and np.array_equal(d["valid_mask"], np.arange(max_steps+2) < n)
        and np.all(np.isfinite(values)) and np.all(np.isfinite(bound))
        and np.all(np.isfinite(times)) and np.all(np.diff(times[:n]) > 0)
        and times[0] == 0 and times[n-1] == 1 and np.array_equal(values[0], initial)
        and np.all(times[n:] == times[n-1]) and np.all(values[n:] == values[n-1])
        and np.all(bound[n:] == bound[n-1]) and np.all(bound >= 0)
        and not any(d[k] for k in ("max_steps_reached", "nonfinite_detected",
                                   "step_size_underflow"))
    )
    error = maximum = ratio = scaled = excess = difference = None
    envelope_ok = passed = False
    if complete:
        truth = reference(case, times[:n], alpha)
        errors = np.abs(values[:n].astype(np.float64)-truth)
        reference_bound = reference_envelope(times[:n], alpha, curvature(case))
        envelope = bound[:n].astype(np.float64)
        error, maximum = float(np.max(errors)), float(np.max(envelope))
        ratio = float(np.max(envelope*(1+rtol)/(atol+rtol*np.abs(values[:n]))))
        scaled = float(np.max(errors/(atol+rtol*np.abs(truth))))
        excess = float(np.max(errors-envelope))
        difference = float(np.max(np.abs(envelope-reference_bound)))
        envelope_ok = bool(np.allclose(
            envelope, reference_bound, rtol=2e-5 if dtype == "float32" else 2e-11,
            atol=1e-10 if dtype == "float32" else 1e-14,
        ))
        passed = bool(envelope_ok and ratio <= 1+2e-6 and scaled <= 1
                      and excess <= ROUNDOFF_SLACK[dtype])
    return BoundRow(
        case, dtype, alpha, atol, rtol, max_steps, accepted, rejected, complete,
        bool(d["max_steps_reached"]), bool(d["nonfinite_detected"]),
        bool(d["step_size_underflow"]), error, maximum, ratio, scaled, excess,
        difference, envelope_ok, passed,
    )


def run_case(case, alpha, atol, dtype, *, max_steps=512):
    if dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("float64 validation requires JAX_ENABLE_X64=1")
    if case not in CASES or dtype not in ROUNDOFF_SLACK:
        raise ValueError("Unknown case or dtype")
    dt = getattr(jnp, dtype)
    initial = jnp.asarray([1., .5] if case == "vector_relative" else 0., dtype=dt)
    order = jnp.asarray(alpha, dtype=dt)
    output = runner(case, atol, max_steps)(
        initial, jnp.asarray([0., 1.], dtype=dt), order,
    )
    return summarize(output, case, dtype, float(order), atol, max_steps)


def collect_rows(dtype, max_steps):
    return [run_case(case, a, tol, dtype, max_steps=max_steps)
            for case in CASES for a in ORDERS for tol in TOLERANCES]


def rows_passed(rows, dtype, max_steps):
    expected = {(case, float(np.asarray(a, dtype=dtype)), tol)
                for case in CASES for a in ORDERS for tol in TOLERANCES}
    actual = {(r.case, r.order, r.atol) for r in rows}
    return (len(rows) == len(expected) and actual == expected
            and all(r.dtype == dtype and r.max_steps == max_steps and r.passed
                    for r in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(ROUNDOFF_SLACK), default="float32")
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("max-steps must be positive")
    if args.dtype == "float64" and not jax.config.x64_enabled:
        parser.error("float64 validation requires JAX_ENABLE_X64=1")
    rows = collect_rows(args.dtype, args.max_steps)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(BoundRow.__dataclass_fields__),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    text = stream.getvalue()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"bounded_forcing_{args.dtype}"
        stem.with_suffix(".csv").write_text(text, encoding="utf-8", newline="")
        context = _context(args.dtype, args.max_steps, False)
        context["scope"] = (
            "conditional prescribed-forcing nodal bound; excludes roundoff"
        )
        context["provenance"] = current_provenance(config={
            "report": "bounded_forcing_validation", "dtype": args.dtype,
            "cases": CASES, "orders": ORDERS, "atols": TOLERANCES,
            "curvatures": {case: curvature(case) for case in CASES},
            "max_steps": args.max_steps, "first_step": 1., "safety": .8,
            "roundoff_slack": ROUNDOFF_SLACK[args.dtype],
        }).to_dict()
        for name in ("benchmarks/numerical/bounded_forcing_validation.py",
                     "src/hpfracc/solvers/bounded_forcing.py",
                     "docs/validation/audits/bounded-forcing-2026-09-15/protocol.json"):
            context["source_sha256"][name] = hashlib.sha256(
                (ROOT / name).read_bytes(),
            ).hexdigest()
        context["csv_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        context["rows"] = [asdict(row) for row in rows]
        stem.with_suffix(".json").write_text(
            json.dumps(context, indent=2, allow_nan=False)+"\n", encoding="utf-8",
        )
    sys.stdout.write(text)
    return int(args.check and not rows_passed(rows, args.dtype, args.max_steps))


if __name__ == "__main__":
    raise SystemExit(main())
