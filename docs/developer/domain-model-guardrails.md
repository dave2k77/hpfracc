# Domain Model Guardrails

Status: Required guardrails for Phase 7 and later domain-model work.

HPFRACC's numerical core is domain-agnostic research software. Future neural
mass, neural field, and phantom-brain modules must build on the validated core
without implying biological or clinical validity before independent validation
exists.

## Required Boundaries

- Neural mass, neural field, and phantom-brain APIs remain experimental until independently validated against clearly identified references or datasets.
- Make no biological, clinical, diagnostic, or subject-specific claims from toy
  models, synthetic examples, smoke tests, or unvalidated parameter choices.
- Do not present model output as a patient-level interpretation, treatment
  recommendation, diagnosis, prognosis, or substitute for empirical biomedical
  assessment.
- Keep examples explicit about what they demonstrate: numerical wiring,
  differentiability, reproducibility, or sensitivity behavior, not biological
  realism.

## Minimum Evidence Before Claims Expand

Future domain-model contributions should include model-relative tests and
sensitivity checks before any stronger documentation language is added.

At minimum, domain-model work should document:

1. the exact governing equations and parameter meanings;
2. the reference implementation, paper, or dataset used for comparison;
3. model-relative tests for known equilibria, limiting cases, or conserved
   quantities when applicable;
4. sensitivity checks for timestep, fractional order, parameter perturbations,
   random seeds, and observation transforms;
5. validation status labels that distinguish numerical correctness from domain
   realism.

## Review Checklist

Before merging Phase 7 neural mass, neural field, or phantom-brain work, confirm:

- [ ] public APIs are labeled experimental;
- [ ] examples and docs avoid biological, clinical, diagnostic, or
  subject-specific claims;
- [ ] model-relative tests cover the documented equations or assumptions;
- [ ] sensitivity checks are present or explicitly deferred with a tracked gap;
- [ ] validation docs state what has and has not been validated;
- [ ] README and quickstart language still describe HPFRACC as research software
  only.

These guardrails do not block exploratory research code. They prevent exploratory
code from being described as validated biomedical infrastructure before the
required evidence exists.
