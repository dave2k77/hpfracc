# Repository agent guidance

This file provides shared guidance for any coding agent working in this repository.

## Agent-neutral collaboration and attribution

- Keep shared instructions in `AGENTS.md` and use agent-neutral names and wording
  for project files and documentation.
- Credit human contributors only. Do not add AI assistants, models, or vendors as
  authors, co-authors, or contributors in commit trailers, project metadata, or
  contributor lists. Preserve legitimate human attribution.
- Do not append agent branding or generated-by signatures to commit messages,
  pull request descriptions, or project documentation.

## What this is

HPFRACC is a pre-alpha (v0.1) JAX-native library for fractional calculus and
fractional dynamical systems. It is **research software, not clinical/diagnostic
software**. The numerical core (operators + Caputo FDE solver) is the product;
the neural, probabilistic, and brain-model layers are deliberately experimental
prototypes that must not drive the architecture of the core.

The authoritative spec is `v0.1_engineering_contract.md`. When in doubt about
scope, API shape, validation expectations, or what is in/out of scope for v0.1,
read it before deciding. The v0.1 priority order — **correctness > stability >
differentiability > reproducibility/provenance > performance** — governs every
tradeoff.

## Environment & commands

This project uses `uv` with a committed lockfile. Always run via `uv run` so the
reproducible environment is used.

```bash
uv sync --locked --extra dev             # set up the locked dev environment
uv run --frozen python -m pytest         # full test suite
uv run python -m pytest tests/unit/test_caputo_solver.py            # one file
uv run python -m pytest tests/unit/test_caputo_solver.py::test_name  # one test
uv run --frozen ruff check .             # locked dev tool; E,F,I,UP,B
uv run --frozen mkdocs build --strict    # build docs (must pass --strict)
```

Validation / benchmark entry points (all are `python -m` modules, not scripts):

```bash
uv run --frozen python -m benchmarks.numerical.validation_summary --check
uv run --frozen python -m benchmarks.numerical.core_hardening --check
uv run --frozen python -m benchmarks.numerical.history_validation --check
uv run python -m benchmarks.numerical.operator_validation.report   # operator detail
uv run python -m benchmarks.numerical.solver_validation.report     # solver detail
uv run python -m benchmarks.numerical.operator_scaling             # scaling smoke test
uv run python -m benchmarks.numerical.baseline                     # CPU baseline
```

Requires Python >=3.11. Only runtime dependency is `jax`; SciPy/mpmath/hypothesis
are test-only validation dependencies and must not leak into the core runtime.
Ruff is declared in the dev extra and locked. CI pins uv 0.11.21 and uses
locked/frozen project commands for Python 3.11–3.13 CPU tests, lint and strict
docs. Run `uv lock` only for intentional dependency changes and review its diff.

## Architecture

Public API is reached through `import hpfracc as hp`. Top-level namespaces map to
`src/hpfracc/<name>/`:

- `hp.ops` — fractional derivative operators (the numerical foundation)
- `hp.solvers` — fixed-step Caputo FDE solver
- `hp.config` — `Provenance`, `ExperimentMetadata`, runtime targets
- `hp.metrics`, `hp.typing` — supporting contracts
- `hp.nn`, `hp.prob` — **experimental** model/probabilistic layers
- `hp.experimental`, plus placeholder namespaces `hp.brain`, `hp.observe`,
  `hp.train`, `hp.data`, `hp.viz`

### Key design conventions (these recur across the codebase)

- **JAX-only.** Public numerical functions take JAX arrays / pytrees and stay
  `jit`/`grad`-compatible where declared. No hidden global state or Python-side
  mutation in numerical paths. JAX is imported lazily inside functions (see the
  `_jnp()`/`_jax()` helpers in `ops/fractional.py` and `solvers/`) so that
  import-time and non-numerical code paths don't hard-require JAX — preserve this
  pattern when adding operators/solvers.
- **Full-history defaults on purpose.** GL/RL/Caputo-L1 and deterministic solvers
  retain their full-history reference defaults. Uniform operators have named
  opt-ins: FFT evaluates the same discrete convolution up to roundoff;
  short-memory and SOE are approximations. Short-memory now uses windowed
  execution, and SOE uses recursive application with legacy or opt-in
  `scaled_tail` fitting. Solver histories remain uncompressed. Never silently
  substitute an approximation or promote a selected-case validation claim.
- **Structured results, not tuples.** Operators return `OperatorResult`
  (with `return_info=True`) carrying `OperatorInfo`; the solver returns
  `SimulationResult` carrying `SolverInfo`. Metadata objects are frozen
  slotted dataclasses with a `to_dict()` JSON-compatible export. New public
  results follow this pattern and attach `Provenance`.
- **Shape conventions** (from contract §8): single state `(..., state_dim)`;
  trajectory `(time, ..., state_dim)` with **time as the leading axis only after
  simulation**; stepwise dynamics operate on a single time-slice. Operators take
  inputs with a leading time axis `(time, ...)`.
- **Dynamics signature** is fixed: `f(t, state, params, *, rng_key=None,
  inputs=None)`. Randomness always uses explicit JAX PRNG keys, even where v0.1
  behavior is deterministic.
- **Current numerical scope:** `0 < alpha < 1`. Uniform operators support scalar
  or per-state orders for full/FFT/short-memory history; SOE remains scalar.
  Nonuniform Caputo L1 supports scalar/full history only. Deterministic solvers
  remain scalar-order: uniform explicit/implicit (matching `dt`), supplied
  nonuniform, or adaptive meshes. Uniform solvers expose opt-in stable weight
  evaluation; adaptive gradients freeze the realized mesh. Use
  `docs/validation/capabilities.md` for the supported/evidenced combinations.
- **Stability tiers** (`hpfracc.stability.StabilityTier`): every public object is
  stable / provisional / experimental / private. Most operator and solver APIs
  are *provisional*; neural/prob/brain APIs are *experimental* until validated.

## Validation is part of the contract

Tests in `tests/unit/` are the scientific contract, not just dev hygiene. New
operators/solvers need analytic/limiting-case validation, finite-difference
gradient checks, and `jit`/`grad` smoke tests (contract §9). Validation reports
record raw error values; tolerances are case-specific in tests.

Several tests enforce documentation and release consistency — be aware when
editing docs, version metadata, or the public API:

- `test_docs_consistency.py` checks that benchmark commands in README/docs import,
  that every `hp.*` name in `docs/api/contract.md` resolves, that the version in
  `pyproject.toml` / `_version.py` / `CHANGELOG.md` / `CITATION.cff` all agree,
  and that `docs/developer/domain-model-guardrails.md` contains required
  non-clinical guardrail phrasing. Bumping the version or touching public API
  means updating these in lockstep.
- `test_public_api.py`, `test_contracts.py`, `test_provenance.py` pin the public
  surface and result-object contracts.

`docs/validation/status.md` defines the current validation boundary;
`docs/developer/release-checklist.md` is the RC checklist.
