# Phase 6: Research-Ready v0.1 Alpha Task Backlog

Status: Complete

Goal: make the completed numerical, differentiable-model, and experimental
probabilistic foundations coherent enough for a credible v0.1 alpha release
before adding domain-specific brain-model layers.

This backlog is ordered by dependency and release risk. Each task should be
completed with a small focused change, tests or docs build verification, and a
short commit.

## Completion Gate

Phase 6 is complete when all of the following pass from a clean checkout:

```bash
uv run --extra test python -m pytest
uv run --extra docs mkdocs build --strict
uv run --extra test python -m benchmarks.numerical.validation_summary
uv run --extra test python -m benchmarks.numerical.operator_validation.report
uv run --extra test python -m benchmarks.numerical.solver_validation.report
uv run --extra test python -m benchmarks.numerical.baseline
```

The release candidate should also have a clean working tree except for intended
release artifacts.

## P0: Release-Blocking Tasks

### Task 1: Add API stability documentation

Objective: make stability expectations visible before users depend on pre-alpha
APIs.

Files:

- Modify: `docs/api/contract.md`
- Possibly modify: `src/hpfracc/stability.py`
- Possibly modify: `src/hpfracc/__init__.py`
- Test: `tests/unit/test_public_api.py`

Work:

1. Add a stability-tier table to `docs/api/contract.md`.
2. Classify public namespaces and objects:
   - stable: top-level namespace names, research-use policy
   - provisional: `hp.ops` operators, `hp.solvers.PredictorCorrector`,
     `hp.solvers.simulate`, result dataclasses
   - experimental: `hp.nn`, `hp.prob`, `hp.experimental`, placeholder brain and
     workflow namespaces
   - private: underscored modules and helpers
3. Decide whether `hp.StabilityTier` should be exported at top level or remain
   internal to the package.
4. Add or adjust tests if any export behavior changes.

Acceptance criteria:

- API contract states stability tier definitions and current assignments.
- Any code-level stability export is covered by `test_public_api.py`.
- `uv run --extra test python -m pytest tests/unit/test_public_api.py` passes.
- `uv run --extra docs mkdocs build --strict` passes.

### Task 2: Add a release checklist document

Objective: define the exact validation, documentation, and packaging steps for a
v0.1 alpha candidate.

Files:

- Create: `docs/developer/release-checklist.md`
- Modify: `mkdocs.yml`
- Possibly modify: `README.md`

Work:

1. Create a checklist with sections for:
   - clean git state
   - unit tests
   - docs build
   - validation report commands
   - benchmark commands
   - examples smoke test
   - version and changelog check
   - citation metadata check
   - research-use disclaimer check
2. Add the checklist to MkDocs developer navigation.
3. Link it from the roadmap Phase 6 section if useful.

Acceptance criteria:

- Checklist is visible in generated docs navigation.
- All commands are copy-pasteable and use the project-preferred `uv run` form.
- `uv run --extra docs mkdocs build --strict` passes.

### Task 3: Make provenance metadata release-ready

Objective: ensure result metadata and validation outputs carry enough context for
research reproducibility.

Files:

- Modify: `src/hpfracc/config/base.py`
- Modify: `src/hpfracc/config/__init__.py`
- Modify: validation methodology documentation for report provenance capture
- Modify: validation report modules under `benchmarks/numerical/` only if CSV
  format changes are explicitly needed
- Test: `tests/unit/test_contracts.py`
- Test: `tests/unit/test_provenance.py`

Work:

1. Review `Provenance` fields: package version, commit hash, backend, runtime
   target, timestamp, config.
2. Add a small helper for constructing runtime provenance if needed, e.g.
   current JAX backend/platform and best-effort git commit hash.
3. Keep default construction deterministic enough for tests; avoid hidden global
   mutation.
4. Ensure `to_dict()` remains JSON-compatible.
5. Document why validation CSV rows remain plain CSV and how to capture
   provenance JSON alongside report artifacts.

Acceptance criteria:

- Provenance helper behavior is tested without requiring a git command to
  succeed.
- Existing structured result contracts still pass.
- Validation output format remains documented, including separate provenance
  capture for CSV reports.
- `uv run --extra test python -m pytest tests/unit/test_contracts.py` passes.

### Task 4: Smoke-test and document all examples

Objective: ensure every documented example actually runs in the release
candidate environment.

Files:

- Modify: `examples/fractional_ode/README.md`
- Modify: `docs/quickstart/index.md`
- Possibly modify example scripts under `examples/fractional_ode/`
- Test: add or extend example smoke tests if appropriate

Work:

1. Run each example through `uv run --extra test python ...`:
   - `examples/fractional_ode/caputo_operator.py`
   - `examples/fractional_ode/caputo_solver.py`
   - `examples/fractional_ode/train_neural_fode.py`
   - `examples/fractional_ode/probabilistic_calibration.py`
2. Fix output or imports if an example fails.
3. Document expected high-level output in the examples README.
4. Expand quickstart beyond the operator-only snippet to include solver,
   metadata, and links to training/probabilistic examples.

Acceptance criteria:

- All example commands complete successfully.
- Examples README includes commands and expected output shape/meaning.
- Quickstart covers first operator call and first Caputo FDE simulation.
- `uv run --extra docs mkdocs build --strict` passes.

## P1: High-Value Release Polish

### Task 5: Align README, docs index, and roadmap language

Objective: keep user-facing project positioning consistent across entry points.

Files:

- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `docs/developer/roadmap.md`
- Possibly modify: `CONTRIBUTING.md`

Work:

1. Ensure each entry point says v0.1 alpha is research software, not clinical
   software.
2. Ensure each entry point says Phase 6 is release-readiness, not domain-model
   implementation.
3. Mention that neural mass and neural field work is planned after the alpha
   cleanup.
4. Keep wording concise; avoid duplicating the whole engineering contract.

Acceptance criteria:

- No stale references imply Phase 6 is currently neural mass / neural field
  implementation.
- Research-only language remains visible.
- `uv run --extra docs mkdocs build --strict` passes.

### Task 6: Add a validation status summary page

Objective: give readers one concise page explaining what has been validated and
what remains provisional.

Files:

- Create: `docs/validation/status.md`
- Modify: `mkdocs.yml`
- Possibly modify: `docs/validation/methodology.md`

Work:

1. Summarize validated areas:
   - Caputo constants and power-law references
   - RL/GL consistency under the v0.1 baseline
   - scalar linear Caputo FDE refinement
   - finite-difference gradient checks
   - basic stability checks
   - stochastic reproducibility smoke tests
   - scalar-grid probabilistic calibration behavior
2. Summarize provisional areas:
   - gradients with respect to fractional order
   - long-time performance claims
   - stochastic numerical validity beyond the additive-noise baseline
   - domain/biological realism
3. Link report commands back to `docs/validation/methodology.md`.

Acceptance criteria:

- Validation status page is visible in docs nav.
- Page distinguishes validated claims from provisional capabilities.
- Page links report commands back to `docs/validation/methodology.md`.
- `uv run --extra docs mkdocs build --strict` passes.

### Task 7: Capture benchmark/report outputs for release notes

Objective: make the release candidate auditable without requiring readers to run
every benchmark immediately.

Files:

- Possibly create: `docs/validation/reports/v0.1.0a0/README.md`
- Possibly create generated text/CSV report files under `docs/validation/reports/v0.1.0a0/`
- Modify: `.gitignore` only if generated reports should be excluded instead

Work:

1. Decide whether report outputs should be committed or generated on demand.
2. If committed, run the validation and benchmark commands and save outputs with
   date, Python version, JAX version, backend, and commit hash.
3. If not committed, document the exact generation commands and expected output
   shape.

Acceptance criteria:

- Release notes can point to either committed report artifacts or reproducible
  commands.
- No broad performance claim appears without benchmark context.

### Task 8: Review package metadata for alpha release

Objective: ensure package metadata matches the release candidate.

Files:

- Modify: `pyproject.toml`
- Modify: `CITATION.cff`
- Modify: `CHANGELOG.md`
- Possibly modify: `README.md`

Work:

1. Check version string consistency between `pyproject.toml`,
   `src/hpfracc/_version.py`, README, and changelog.
2. Check license, authors, project URLs, classifiers, and Python version support.
3. Check citation metadata for the current version and project title.
4. Add a release date only when the tag date is known.

Acceptance criteria:

- Version metadata is consistent.
- Changelog accurately describes completed work.
- Citation metadata is not stale.

## P2: Optional If Time Allows

### Task 9: Add docs/code consistency tests

Objective: prevent future docs from drifting away from documented commands and
public API names.

Files:

- Possibly create: `tests/unit/test_docs_consistency.py`

Work:

1. Test that all docs command references to benchmark modules import cleanly.
2. Test that public API names listed in `docs/api/contract.md` exist.
3. Keep tests simple and avoid brittle full-document parsing.

Acceptance criteria:

- Tests catch missing public objects or renamed benchmark modules.
- `uv run --extra test python -m pytest tests/unit/test_docs_consistency.py`
  passes.

### Task 10: Add a no-claims guardrail note for future brain modules

Objective: reduce risk before Phase 7 domain-model work starts.

Files:

- Modify: `docs/developer/index.md`
- Possibly create: `docs/developer/domain-model-guardrails.md`
- Modify: `mkdocs.yml` if a new page is created

Work:

1. State that neural mass, neural field, and phantom-brain modules must remain
   experimental until independently validated.
2. State that no biological, clinical, diagnostic, or subject-specific claims
   should be made from toy models.
3. Require model-relative tests and sensitivity checks for future domain models.

Acceptance criteria:

- Future Phase 7 contributors have clear guardrails.
- `uv run --extra docs mkdocs build --strict` passes.

## Suggested Work Order

1. Task 1: API stability documentation.
2. Task 2: release checklist.
3. Task 4: example smoke tests and quickstart expansion.
4. Task 3: provenance metadata review.
5. Task 5: README/docs/roadmap language alignment.
6. Task 6: validation status summary.
7. Task 8: package metadata review.
8. Task 7: report output capture decision.
9. Task 9 and Task 10 if time allows.

Do not start Phase 7 neural mass / neural field implementation until P0 is
complete and the Phase 6 completion gate passes.
