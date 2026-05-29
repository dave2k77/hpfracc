# Developer Notes

The numerical core should remain domain-agnostic. Neural mass, neural field,
probabilistic, and phantom-brain layers should build on top of validated core
operators and solvers.

Before Phase 7 domain-model work starts, review the
[Domain Model Guardrails](domain-model-guardrails.md). Future neural mass,
neural field, and phantom-brain modules must remain experimental until
independently validated, and examples must avoid biological, clinical,
diagnostic, or subject-specific claims from toy models.

