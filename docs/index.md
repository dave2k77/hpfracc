# HPFRACC

HPFRACC is a pre-alpha research-support library for fractional calculus and
fractional dynamical systems in differentiable scientific computing.

The v0.1 alpha line focuses on a validated JAX-native numerical core before
adding domain-specific brain-model or phantom-brain workflows.

HPFRACC is research software only. It is not clinical, diagnostic, or
subject-specific decision software.

## Current Development Focus

Active phase: Phase A

Latest tagged release: `0.1.0a1` (2026-06-26).

Phases 0-6 delivered the original alpha baseline. The tagged release adds nonuniform/adaptive
and implicit solvers, per-state operator orders, and selected order-gradient
validation. Further precision, input-validation, solver-diagnostic, and reporting
hardening is unreleased, not part of that tag.

- The [roadmap](developer/roadmap.md#current-development-state) separates released
  capabilities, unreleased changes, and remaining Phase A gates.
- The [validation status](validation/status.md) defines case-specific evidence
  and limitations; a passing check is not a universal accuracy guarantee.
- The [API contract](api/contract.md) records supported interfaces and stability
  tiers. Numerical validation does not make the experimental layers stable.
- The [release checklist](developer/release-checklist.md) is a candidate template,
  not an executed release record. Runtime provenance can be captured with
  `hp.config.current_provenance()` and must identify the source used.

Neural mass, neural field, EEG, and phantom-brain workflows remain future domain
phases and should not be interpreted as current validated capabilities.
