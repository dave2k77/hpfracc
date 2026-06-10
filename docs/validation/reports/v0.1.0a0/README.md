# v0.1.0a0 Validation Report Bundle

This directory contains a small validation and benchmark artifact bundle for the
HPFRACC `0.1.0a0` pre-alpha release-readiness pass.

The bundle is intentionally small and text-based so it can be reviewed in git.
It records representative local CPU validation output; it is not a general
performance claim.

## Files

| File | Purpose |
| --- | --- |
| `provenance.json` | Runtime context captured with `hp.config.current_provenance()`. |
| `validation_summary.csv` | Aggregate pass/fail rows for operator, solver, convergence-order, gradient, and stability checks. |
| `operator_validation.csv` | Detailed operator correctness rows, including Caputo and Riemann-Liouville analytic-reference errors. |
| `solver_validation.csv` | Detailed scalar Caputo FDE solver refinement rows. |
| `baseline.csv` | Local CPU-oriented operator and solver timing rows. |

The artifact CSV files are linked below for repository browsing:

- [`validation_summary.csv`](validation_summary.csv)
- [`operator_validation.csv`](operator_validation.csv)
- [`solver_validation.csv`](solver_validation.csv)
- [`baseline.csv`](baseline.csv)
- [`provenance.json`](provenance.json)

## Generation Commands

Generated from the repository root with the project `uv` environment:

```bash
uv run python -m benchmarks.numerical.validation_summary \
  > docs/validation/reports/v0.1.0a0/validation_summary.csv

uv run python -m benchmarks.numerical.operator_validation.report \
  > docs/validation/reports/v0.1.0a0/operator_validation.csv

uv run python -m benchmarks.numerical.solver_validation.report \
  > docs/validation/reports/v0.1.0a0/solver_validation.csv

uv run python -m benchmarks.numerical.baseline \
  > docs/validation/reports/v0.1.0a0/baseline.csv

uv run python -c "import json, hpfracc as hp; print(json.dumps(hp.config.current_provenance(config={'report_bundle': 'v0.1.0a0'}).to_dict(), indent=2))" \
  > docs/validation/reports/v0.1.0a0/provenance.json
```

## Interpretation

- `validation_summary.csv` is the fastest artifact to inspect for the current
  validation boundary.
- Passing rows apply only to the documented cases and tolerances.
- `baseline.csv` is local timing context for the machine recorded in
  `provenance.json`; do not compare it across machines as a benchmark claim.
- HPFRACC remains research software only. These artifacts do not establish
  clinical, diagnostic, biological, EEG, neural-mass, neural-field, or
  subject-specific validity.

See also:

- [Validation Status](../../status.md)
- [Validation Methodology](../../methodology.md)
