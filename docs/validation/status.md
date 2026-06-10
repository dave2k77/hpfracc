# Validation Status

Status: Phase 6 v0.1 alpha summary.

HPFRACC validation is scoped to research software claims for fractional
operators, fixed-step fractional differential equation solvers, differentiable
workflows, and small experimental probabilistic utilities. This page separates
validated v0.1 behavior from provisional or out-of-scope capabilities.

HPFRACC is not clinical, diagnostic, or subject-specific decision software.
Validation here does not establish biological realism or fitness for medical
use.

## Validated in the v0.1 Alpha Scope

| Area | Current status | Evidence |
| --- | --- | --- |
| Caputo derivative of constants | Validated for the v0.1 full-history L1 implementation. | Unit tests and `uv run python -m benchmarks.numerical.operator_validation.report`. |
| Caputo power-law references | Validated against analytic `t**power` references with refinement checks. | Unit tests and operator validation report. |
| Observed convergence order | Validated that the Caputo L1 operator attains max-norm order `2 - alpha`, the Grunwald-Letnikov / Riemann-Liouville operator attains first order `O(h)`, and the predictor-corrector solver attains endpoint order `1 + alpha`, by float64 grid-refinement slope estimates. | Unit tests in `tests/unit/test_convergence_order.py` and `uv run python -m benchmarks.numerical.convergence`. |
| Riemann-Liouville correctness | Validated against the analytic RL power-law reference and the decisive constant case (RL of a constant is `t**(-alpha) / Gamma(1 - alpha)`, nonzero, distinguishing it from Caputo). Replaces the earlier tautological RL-equals-GL check. | Unit tests in `tests/unit/test_fractional_operators.py` and `uv run python -m benchmarks.numerical.operator_validation.report`. |
| JAX JIT compatibility | Smoke-tested for representative operators and solver calls. | Unit tests for JIT-compatible operator and solver paths. |
| Input/parameter differentiability | Validated with finite-difference checks for representative operator and solver quantities. | `uv run python -m benchmarks.numerical.gradient_checks` and unit tests. |
| Scalar linear Caputo FDE refinement | Validated against a truncated Mittag-Leffler reference for selected grids. | `uv run python -m benchmarks.numerical.solver_validation.report`. |
| Basic numerical stability checks | Smoke-tested for constant zero response and non-amplifying scalar decay. | `uv run python -m benchmarks.numerical.stability`. |
| Baseline benchmark context | Local CPU-oriented timing rows include backend/platform context. | `uv run python -m benchmarks.numerical.baseline`. |
| Stochastic reproducibility mechanics | Reproducibility with fixed PRNG keys and variation across keys are tested for the additive-noise helper. | Unit tests in `tests/unit/test_probabilistic_phase5.py`. |
| Scalar-grid probabilistic calibration behavior | Tested for Gaussian likelihood preference, nearby scalar parameter recovery, normalized posterior weights, and posterior-predictive summary shape. | Unit tests in `tests/unit/test_probabilistic_phase5.py` and the calibration example. |
| Runtime provenance capture | Tested for JSON-compatible runtime context and graceful operation without git. | Unit tests in `tests/unit/test_provenance.py`. |

## Provisional Capabilities

The following capabilities exist, but should not be treated as validated stable
research infrastructure yet:

- Gradients with respect to fractional order `alpha`.
- Numerical behavior for long time horizons or very large state dimensions.
- General stochastic fractional differential equation accuracy beyond the
  additive-noise reproducibility smoke tests.
- Broad Bayesian inference workflows beyond scalar-grid Gaussian calibration.
- Performance claims outside the local benchmark context.
- Biological, neural, EEG, clinical, diagnostic, or subject-specific realism.
- Placeholder domain namespaces such as `hpfracc.brain`, `hpfracc.observe`,
  `hpfracc.train`, `hpfracc.data`, and `hpfracc.viz`.

## Required Validation Commands

Run the aggregate validation summary:

```bash
uv run python -m benchmarks.numerical.validation_summary
```

Run detailed operator and solver reports:

```bash
uv run python -m benchmarks.numerical.operator_validation.report
uv run python -m benchmarks.numerical.solver_validation.report
```

Run component validation checks when detailed rows are needed:

```bash
uv run python -m benchmarks.numerical.gradient_checks
uv run python -m benchmarks.numerical.stability
```

Run the observed-convergence-order report (estimates empirical orders against
the expected `2 - alpha` operator and `1 + alpha` solver rates):

```bash
uv run python -m benchmarks.numerical.convergence
```

Run the baseline benchmark for local timing context:

```bash
uv run python -m benchmarks.numerical.baseline
```

See [Validation Methodology](methodology.md) for report formats, command
options, and provenance capture guidance. See the
[v0.1.0a0 Report Bundle](reports/v0.1.0a0/README.md) for committed example
artifacts from the current alpha-readiness pass.

## Interpreting Validation Output

Validation reports write plain CSV to stdout. A passing validation row means the
specific documented case passed its configured tolerance. It does not imply a
general theorem about all fractional systems, all discretizations, all time
horizons, or all hardware backends.

When saving validation artifacts for release notes, save runtime provenance as a
separate JSON artifact next to the CSV output:

```bash
uv run python - <<'PY'
import json
import hpfracc as hp

print(json.dumps(hp.config.current_provenance().to_dict(), indent=2))
PY
```

## Release-Readiness Rule

For Phase 6, a v0.1 alpha candidate should not make a claim unless that claim is
covered by one of:

1. a unit test;
2. a validation report command;
3. a documented benchmark with runtime context;
4. an explicit provisional/experimental label.
