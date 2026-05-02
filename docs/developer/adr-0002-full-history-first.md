# ADR 0002: Full-History Implementations First

Status: Accepted

## Context

Fractional operators and fractional dynamical systems are history dependent.
Fast memory approximations such as short-memory truncation, FFT convolution, and
sum-of-exponentials compression are important, but they can obscure baseline
correctness.

## Decision

Initial v0.1 operators and solvers use explicit full-history discretisations.
Approximate memory strategies must be added later as named and configurable
methods.

## Consequences

- Initial methods may have quadratic time or memory scaling.
- Complexity must be documented.
- Baseline validation is easier to reason about.
- Later high-performance methods can be compared against a clear reference.

