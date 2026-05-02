# ADR 0001: JAX-Only v0.1 Backend

Status: Accepted

## Context

HPFRACC v0.1 prioritizes numerical correctness, stability, differentiability,
and execution on CPU plus a single accelerator. Supporting multiple numerical
backends would multiply API, autodiff, randomness, and testing complexity before
the fractional operators and solvers are validated.

## Decision

HPFRACC v0.1 is JAX-only.

Public numerical APIs should use JAX-compatible arrays and pytrees. Stochastic
APIs must use explicit JAX PRNG keys. Multi-device support is deferred, but APIs
should avoid choices that block future JAX sharding support.

## Consequences

- The core runtime dependency is JAX.
- Validation can use SciPy and mpmath as test-only dependencies.
- Neural models should follow pytree conventions.
- Non-JAX backends are out of scope until after the v0.1 numerical foundation is
  stable.

