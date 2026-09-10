# Developer Notes

Active phase: Phase A

Latest tagged release: `0.1.0a1` (2026-06-26).

The numerical core should remain domain-agnostic. Neural mass, neural field,
probabilistic, and phantom-brain layers should build on top of validated core
operators and solvers.

Start with the [current development state](roadmap.md#current-development-state)
and [validation status](../validation/status.md). Tagged features, unreleased
hardening, and remaining gates are separate.
The Phase 2/6 plans are historical baselines, and the
[release checklist](release-checklist.md) is a template, not release sign-off.

Before domain-model work starts (Phase C), review the
[Domain Model Guardrails](domain-model-guardrails.md). Future neural mass,
neural field, and phantom-brain modules must remain experimental until
independently validated, and examples must avoid biological, clinical,
diagnostic, or subject-specific claims from toy models.

## Test tiers

The suite is organized into tiers so each kind of check has a clear home. New
operators and solvers should add to the relevant tiers as they land.

- `tests/unit/` — per-component correctness: analytic/limiting cases,
  convergence order, finite-difference gradient checks, and `jit`/`grad` smoke
  tests. This is the scientific contract.
- `tests/property/` — `@pytest.mark.property` Hypothesis tests of algebraic
  invariants that must hold for every admissible input (for example operator
  linearity and the Caputo-of-a-constant identity).
- `tests/regression/` — `@pytest.mark.regression` golden-artifact tests pinning
  operator outputs to committed references in `tests/regression/golden/`.
  Regenerate intentionally with `tests/regression/golden/_generate.py` and
  review the diff; an unexplained diff is a regression.
- `tests/performance/` — `@pytest.mark.performance` timing/scaling guards.
  These are **deselected by default** (`-m "not performance"` in
  `pyproject.toml`) because wall-clock results are machine-dependent. Run them
  with `uv run python -m pytest -m performance`; in CI they run in a separate
  non-gating job.

Shared fixtures live in `tests/conftest.py`: `jax_mod` (skip if JAX is missing)
and `enable_x64` (scoped float64 for refinement and gradient checks).

