"""Selected scientific acceptance, distinct from measurement-integrity checks."""

import argparse
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import version
from math import isclose, isfinite
from numbers import Real
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.numerical import implicit_validation as audit
from benchmarks.numerical.implicit_validation import ImplicitRow
from hpfracc.config import current_provenance
from hpfracc.solvers import ImplicitPredictorCorrector, simulate


@dataclass(frozen=True, slots=True)
class Assessment:
    """A trajectory's scientific decision, not its measurement validity alone."""

    reasons: tuple[str, ...]

    @property
    def decision(self) -> str:
        return self.reasons[0] if self.reasons else "accepted"

    @property
    def accepted(self) -> bool:
        return not self.reasons


def _finite(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _coherent(row: ImplicitRow) -> bool:
    """Reject contradictory summaries; this is not data authentication."""
    if not all(type(flag) is bool for flag in
               (row.measurement_ok, row.finite_trajectory, row.newton_converged)):
        return False
    if (not row.measurement_ok or row.case not in audit.SCALES
            or row.dtype not in audit.RTOL or type(row.n_samples) is not int
            or row.n_samples < 2):
        return False
    counts = (row.newton_failure_count, row.newton_nonfinite_count)
    if not all(type(count) is int and 0 <= count < row.n_samples for count in counts):
        return False
    if (row.newton_nonfinite_count > row.newton_failure_count
            or row.newton_converged != (row.newton_failure_count == 0)):
        return False
    reported = row.reported_max_scaled_residual
    if row.newton_nonfinite_count:
        if reported is not None:
            return False
    elif reported is None or not _finite(reported) or reported < 0:
        return False
    elif row.newton_converged != (reported <= 1):
        return False
    accuracy = (row.max_dimensionless_error, row.endpoint_dimensionless_error,
                row.minimum_dimensionless_state, row.component_max_abs_error)
    if not row.finite_trajectory:
        return bool(row.newton_nonfinite_count and not row.newton_converged
                    and all(value is None for value in accuracy)
                    and row.independent_max_scaled_residual is None
                    and row.residual_flag_disagreements is None)
    error, endpoint, minimum, components = accuracy
    independent = row.independent_max_scaled_residual
    if error is None or endpoint is None or minimum is None or independent is None:
        return False
    if (not all(_finite(value) for value in (error, endpoint, minimum, independent))
            or error < 0 or endpoint < 0 or endpoint > error or independent < 0
            or reported is None):
        return False
    disagreements = row.residual_flag_disagreements
    if type(disagreements) is not int or not 0 <= disagreements < row.n_samples:
        return False
    if (independent <= 1 and row.newton_converged and disagreements
            or independent > 1 and row.newton_converged and not disagreements):
        return False
    if (not isinstance(components, tuple)
            or len(components) != len(audit.SCALES[row.case])
            or not all(_finite(value) and value >= 0 for value in components)):
        return False
    scales = np.asarray(audit.SCALES[row.case], dtype=row.dtype).astype(np.float64)
    normalized = max(value / scale for value, scale in
                     zip(components, scales, strict=True))
    return isclose(error, normalized, rel_tol=1e-12, abs_tol=0.)


def assess(row: ImplicitRow, *, error_limit: float) -> Assessment:
    """Assess a measured trajectory against a declared continuous-error budget."""
    if not _finite(error_limit) or error_limit <= 0:
        raise ValueError("error_limit must be finite and positive")
    if not _coherent(row):
        return Assessment(("invalid_measurement",))
    if not row.finite_trajectory:
        return Assessment(("nonfinite_trajectory",))
    reasons = []
    if not row.newton_converged:
        reasons.append("unconverged_corrector")
    independent = row.independent_max_scaled_residual
    error = row.max_dimensionless_error
    assert independent is not None and error is not None  # Established by _coherent.
    if independent > 1 or row.residual_flag_disagreements:
        reasons.append("independent_residual_failure")
    if error > error_limit:
        reasons.append("inaccurate")
    return Assessment(tuple(reasons))


PROTOCOL_VERSION = "implicit-scientific-v1"


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """Authoritative case identity, solver settings and predeclared criteria."""

    case_id: str
    case: str
    dtype: str
    weight_evaluation: str
    order: float
    strength: float
    n_samples: int
    newton_iterations: int
    newton_rtol: float
    newton_atol: float
    error_limit: float
    role: str
    expected_reason: str


def case_manifest(
    dtype: str, *, weight_evaluation: str = "direct",
) -> tuple[CaseSpec, ...]:
    """Fixed selected coverage, independent of any submitted measurement rows."""
    if dtype not in audit.RTOL or weight_evaluation not in ("direct", "stable"):
        raise ValueError("Unsupported dtype or weight_evaluation")
    nonlinear_limit = 1e-3 if dtype == "float32" else 2e-5
    definitions = (
        ("weak_decay", "half_decay", .5, 1., 257, 8, 1e-3, "accepted"),
        *((case, case, .5, 1., 129, 8, nonlinear_limit, "accepted")
          for case in ("cubic", "coupled", "scaled")),
        ("coarse_stiff_decay", "half_decay", .5, 30., 17, 8, 1e-3, "inaccurate"),
        ("nonfinite_cubic", "cubic", .3, 30., 17, 1, nonlinear_limit,
         "nonfinite_trajectory"),
    )
    return tuple(CaseSpec(
        case_id=identifier, case=case, dtype=dtype, weight_evaluation=weight_evaluation,
        order=float(np.asarray(order, dtype=dtype)),
        strength=float(np.asarray(strength, dtype=dtype)), n_samples=n,
        newton_iterations=iterations, newton_rtol=audit.RTOL[dtype],
        newton_atol=.01 * audit.RTOL[dtype], error_limit=limit,
        role="required_success" if reason == "accepted" else "expected_rejection",
        expected_reason=reason,
    ) for identifier, case, order, strength, n, iterations, limit, reason
        in definitions)


@dataclass(frozen=True, slots=True)
class CaseResult:
    spec: CaseSpec
    measurement: ImplicitRow
    observed_weight_evaluation: str
    effective_dtypes: tuple[str, ...]
    assessment: Assessment
    accepted: bool
    policy_passed: bool


def _matches_fields(actual, expected, keys) -> bool:
    """Match canonical values and types: True/1 or 129.0/129 are not identical."""
    return all(type(getattr(actual, key)) is type(getattr(expected, key))
               and getattr(actual, key) == getattr(expected, key) for key in keys)


def _evaluate(spec, row, observed_evaluation, effective_dtypes) -> CaseResult:
    keys = ("case", "dtype", "order", "strength", "n_samples",
            "newton_iterations", "newton_rtol", "newton_atol")
    matches = (_matches_fields(row, spec, keys)
               and observed_evaluation == spec.weight_evaluation
               and effective_dtypes == (spec.dtype,) * 5)
    assessment = (assess(row, error_limit=spec.error_limit) if matches else
                  Assessment(("configuration_mismatch",)))
    accepted = assessment.accepted
    if spec.role == "required_success":
        passed = accepted
    elif spec.expected_reason == "inaccurate":
        passed = bool(matches and "inaccurate" in assessment.reasons
                      and row.finite_trajectory and row.newton_converged
                      and row.minimum_dimensionless_state is not None
                      and row.minimum_dimensionless_state < 0
                      and row.max_dimensionless_error is not None
                      and row.max_dimensionless_error > .3)
    else:
        passed = assessment.reasons == ("nonfinite_trajectory",)
    return CaseResult(spec, row, observed_evaluation, effective_dtypes,
                      assessment, accepted, passed)


@lru_cache(maxsize=48)
def _runner(spec: CaseSpec):
    scales = jnp.asarray(audit.SCALES[spec.case], dtype=spec.dtype)
    times = jnp.linspace(0., 1., spec.n_samples, dtype=spec.dtype)

    def model(t, state, params, *, rng_key=None, inputs=None):
        return audit.dynamics(spec.case, t, state, params,
                              rng_key=rng_key, inputs=inputs)

    @jax.jit
    def run(initial, strength, order):
        result = simulate(
            model=model, ts=times, initial_state=initial,
            params=(strength, order, scales),
            solver=ImplicitPredictorCorrector(
                dt=1 / (spec.n_samples - 1), order=order,
                newton_iterations=spec.newton_iterations,
                newton_rtol=spec.newton_rtol, newton_atol=spec.newton_atol,
                weight_evaluation=spec.weight_evaluation,
            ),
        )
        assert result.solver_info is not None
        d = result.solver_info.diagnostics
        diagnostics = {key: d[key] for key in audit.FIELDS}
        diagnostics["weight_evaluation_code"] = jnp.asarray(
            int(d["weight_evaluation"] == "stable"), dtype=jnp.int32,
        )
        return result.ts, result.latent_state, diagnostics
    return run


def _execute(spec: CaseSpec):
    scales = np.asarray(audit.SCALES[spec.case], dtype=spec.dtype)
    initial = jnp.asarray(audit.reference(
        spec.case, np.asarray(0.), spec.strength, scales,
    ), dtype=spec.dtype)
    strength = jnp.asarray(spec.strength, dtype=spec.dtype)
    order = jnp.asarray(spec.order, dtype=spec.dtype)
    output = jax.device_get(_runner(spec)(initial, strength, order))
    times, states, diagnostics = output
    row = audit.summarize(output, spec.case, spec.dtype, float(order),
                          float(strength), spec.n_samples, spec.newton_iterations)
    code = int(diagnostics["weight_evaluation_code"])
    observed = "stable" if code == 1 else "direct" if code == 0 else "unknown"
    dtypes = tuple(str(value.dtype) for value in
                   (times, states, initial, order, strength))
    raw = {"times": times, "states": states, "initial": np.asarray(initial),
           "order": np.asarray(order), "strength": np.asarray(strength),
           **{key: np.asarray(value) for key, value in diagnostics.items()}}
    return _evaluate(spec, row, observed, dtypes), raw


def collect_rows(dtype: str, *, weight_evaluation: str = "direct", raw_runs=None):
    """Execute exactly the manifest under its effective working precision."""
    manifest = case_manifest(dtype, weight_evaluation=weight_evaluation)
    results = []
    with jax.enable_x64(dtype == "float64"):
        for spec in manifest:
            result, raw = _execute(spec)
            results.append(result)
            if raw_runs is not None:
                raw_runs[spec.case_id] = raw
    return results


def rows_passed(rows, dtype: str, *, weight_evaluation: str = "direct") -> bool:
    """Require exact coverage and recompute decisions; do not trust result flags."""
    expected = {s.case_id: s for s in
                case_manifest(dtype, weight_evaluation=weight_evaluation)}
    rows = list(rows)
    if len(rows) != len(expected) or not all(
        isinstance(row, CaseResult) and isinstance(row.spec, CaseSpec) for row in rows
    ):
        return False
    if {row.spec.case_id for row in rows} != set(expected):
        return False
    for result in rows:
        spec = expected[result.spec.case_id]
        if (not _matches_fields(result.spec, spec, CaseSpec.__dataclass_fields__)
                or type(result.accepted) is not bool
                or type(result.policy_passed) is not bool):
            return False
        recomputed = _evaluate(spec, result.measurement,
                               result.observed_weight_evaluation,
                               result.effective_dtypes)
        if not recomputed.policy_passed or result != recomputed:
            return False
    return True


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted((root / "src/hpfracc").rglob("*.py")),
             *sorted((root / "benchmarks").rglob("*.py")),
             root / "tests/unit/test_scientific_acceptance.py",
             root / "pyproject.toml", root / "uv.lock"]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _runtime(manifest) -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True,
        )
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    return {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "python": platform.python_version(), "platform": platform.platform(),
        "jax": jax.__version__, "jaxlib": version("jaxlib"),
        "numpy": np.__version__, "scipy": version("scipy"),
        "git_dirty": dirty, "source_sha256": _source_hashes(),
        "provenance": current_provenance(config={
            "protocol_version": PROTOCOL_VERSION,
            "cases": [asdict(spec) for spec in manifest],
        }).to_dict(),
    }


def _csv_text(rows) -> str:
    extras = ("case_id", "role", "weight_evaluation", "observed_weight_evaluation",
              "effective_dtypes", "error_limit", "expected_reason", "decision",
              "rejection_reasons", "accepted", "policy_passed")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream,
                            fieldnames=[*extras, *ImplicitRow.__dataclass_fields__],
                            lineterminator="\n")
    writer.writeheader()
    for result in rows:
        spec = result.spec
        writer.writerow({
            **asdict(result.measurement), "case_id": spec.case_id, "role": spec.role,
            "weight_evaluation": spec.weight_evaluation,
            "observed_weight_evaluation": result.observed_weight_evaluation,
            "effective_dtypes": ";".join(result.effective_dtypes),
            "error_limit": spec.error_limit, "expected_reason": spec.expected_reason,
            "decision": result.assessment.decision,
            "rejection_reasons": ";".join(result.assessment.reasons),
            "accepted": result.accepted, "policy_passed": result.policy_passed,
        })
    return stream.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(audit.RTOL), default="float32")
    parser.add_argument("--weight-evaluation", choices=("direct", "stable"),
                        default="direct")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    manifest = case_manifest(args.dtype, weight_evaluation=args.weight_evaluation)
    raw_runs = {}
    with jax.enable_x64(args.dtype == "float64"):
        runtime = _runtime(manifest)
        rows = collect_rows(args.dtype, weight_evaluation=args.weight_evaluation,
                            raw_runs=raw_runs)
        if runtime["source_sha256"] != _source_hashes():
            raise RuntimeError("Source inputs changed during scientific validation")
    passed = rows_passed(rows, args.dtype, weight_evaluation=args.weight_evaluation)
    text = _csv_text(rows)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / (
            f"scientific_acceptance_{runtime['backend']}_{args.dtype}_"
            f"{args.weight_evaluation}"
        )
        stem.with_suffix(".csv").write_text(text, encoding="utf-8", newline="")
        artifacts = {}
        for i, result in enumerate(rows):
            path = stem.with_name(f"{stem.name}_case_{i}.npz")
            np.savez_compressed(path, **raw_runs[result.spec.case_id])
            artifacts[result.spec.case_id] = {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        protocol = {"version": PROTOCOL_VERSION,
                    "cases": [asdict(spec) for spec in manifest]}
        report = {
            "scope": "selected implicit acceptance; not Phase A or release sign-off",
            "protocol_version": PROTOCOL_VERSION, "manifest": protocol["cases"],
            "manifest_sha256": hashlib.sha256(json.dumps(
                protocol, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest(),
            "runtime": runtime, "rows": [asdict(row) for row in rows],
            "raw_artifacts": artifacts,
            "csv_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "summary": {
                "configurations": len(rows), "expected_configurations": len(manifest),
                "scientifically_accepted": sum(row.accepted for row in rows),
                "expected_rejections_verified": sum(
                    row.spec.role == "expected_rejection" and row.policy_passed
                    for row in rows
                ),
                "policy_passed": passed,
            },
        }
        stem.with_suffix(".json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    sys.stdout.write(text)
    return int(args.check and not passed)


if __name__ == "__main__":
    raise SystemExit(main())
