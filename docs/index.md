# HPFRACC

HPFRACC is a pre-alpha research-support library for fractional calculus and
fractional dynamical systems in differentiable scientific computing.

The v0.1 alpha line focuses on a validated JAX-native numerical core before
adding domain-specific brain-model or phantom-brain workflows.

HPFRACC is research software only. It is not clinical, diagnostic, or
subject-specific decision software.

## Current Release-Readiness Focus

Phase 6 is a research-ready v0.1 alpha pass. The active work is to make the
existing numerical, differentiable-model, and experimental probabilistic
foundations coherent enough for alpha release:

- API stability tiers are documented in the API contract.
- Validation scope and provisional capabilities are summarized in the validation
  status page.
- Runtime provenance can be captured with `hp.config.current_provenance()`.
- The release checklist records the required tests, validation reports,
  benchmarks, examples, version metadata, and research-use checks.

Neural mass, neural field, EEG, and phantom-brain workflows remain future domain
phases and should not be interpreted as current validated capabilities.
