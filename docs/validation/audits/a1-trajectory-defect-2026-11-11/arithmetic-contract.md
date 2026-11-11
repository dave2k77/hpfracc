# A1 private diagnostic: arithmetic repair contract

## Authorization and state

Owner-authorized plan, 2026-11-11: "okay, agreed. proceed with your recommendation."
This authorizes arithmetic-contract review followed by **one** bounded
implementation/review cycle. Independent technical review **passed** before
numerical implementation. The exact reviewed draft has SHA-256
`b98f781ae2a64131345640997b2ee563f7e1526cccdbfefdc838fc8d28a293f9` and is retained
in `arithmetic-contract-review.tar.gz`; `arithmetic-contract-review.json` records
the scoped verdict and verified evidence handles. This version only adds the
approval record and clarifies that forcing coefficients may be signed. Technical
contract clearance is not approval of a numerical implementation. The rejected source is
`92ff3dceb217f6097bae8ccde6cf159d8daa8d14`, not an approved oracle. Its 981-case
CPU/CUDA default passes, all six independent RED cases and three failed reviews
remain immutable evidence. `pre-arithmetic-checkpoint.tar.gz` preserves mutable
checkpoint files; `pre-arithmetic-layout.json` resolves its original 54-file
manifest. No scientific acceptance or scope reduction is authorized.

## Mathematical and execution boundary

- Keep the original full-history PL reconstruction through the stored states,
  original lower terminal/forcing clock, scalar `0 < alpha < 1`, constant real
  matrix feedback and finite nonnegative forcing exponents with arbitrary finite
  signed coefficients. Preserve both APIs,
  result/provenance contracts, mesh validation and explicit work exhaustion.
- Interpret float32/float64 inputs as their represented binary values. Numerical
  work remains a private host float64 calculation, separate from solver dtype
  and CUDA execution. Do not promote JAX precision or change core/runtime,
  defaults, public exports, dependencies or lockfile.
- This cycle supplies no global relative-accuracy theorem, verified rounding
  budget, outward-rounded cell enclosure, feedback propagation, solution-error
  certificate, JAX/AD contract or broader A1 acceptance. Existing case-specific
  comparison tolerances remain unchanged. Ordinary rounding/conditioning and
  special-function error remain distinct from detected arithmetic-range failure.
- Do not introduce order cutoffs, magnitude whitelists, hidden history dropping
  or new mathematical-domain exclusions to make the tests pass. A per-evaluation
  explicit arithmetic failure is not approval of a narrower completion scope.

## Shared reconstruction primitive

For each panel `[l,r]`, use independently computed distances `dL=t-l`, `dR=r-t`
and width `h=r-l`. Its exact-arithmetic PL formula is
`Y=(dR/h)*Y_l+(dL/h)*Y_r`.

- Locate the panel without changing the represented coordinates; return the
  stored state exactly at a stored knot, including both original endpoints.
- Do not form a state/time slope before multiplying by the target offset.
- Do not obtain either distance/weight by subtracting the other normalized
  fraction from one. Preserve independently resolvable terminal gaps.
- Guard the differences, ratios, weighted products and addition. A stable scaled
  equivalent is permitted; do not silently let a nonzero ratio/product disappear
  through underflow. Use the same primitive for point reconstruction and cell
  endpoint reconstruction, with no public behavior split.
- Canonical VAL-011 B1/B2 inputs should successfully recover their normal
  representable values (about 1000 and `5e-101`), not be blanket-rejected merely
  because the former library primitive failed. Their high-precision references
  use exact integer ratios of the stored binary inputs.

## Active-panel basis conversion

For `l<t<=r`, let `tau=t-l`, `g=r-t`, `h=r-l` and
`m=tau**alpha/Gamma(1+alpha)`. The exact positive weights in the original endpoint
basis are

- `wL=m*(g/h+(tau/h)*alpha/(1+alpha))`;
- `wR=m*(tau/h)/(1+alpha)`.

Compute the remaining gap `g` independently, never as `h-tau` or as the complement
of a rounded normalized fraction. Retain the separately validated old-panel
branches, including their independent age complement and near-terminal
`log1p`/`expm1` evaluation. Check both branches at their boundary. Do not change
kernel terminal or reset history.

The same weight primitive must feed point integrals and cell extrema. Recheck
old-panel monotonicity and the active-left interior maximum at `alpha*h`, using
range-checked peak arithmetic. Recover the canonical coupled B3 point value and
cover its reference in the model cell range at the existing comparison slack;
this is not a claim that floating cell endpoints are certified enclosures.

## Arithmetic-range failure policy and sibling audit

- `np.errstate` is a secondary guard, not proof that called C/BLAS/library internals
  report intermediate underflow. Remove reliance on `np.interp` for reconstruction.
- Audit explicit differences/divisions, powers, positive gamma-ratio/basis values,
  weighted state products, matrix feedback, forcing contraction and accumulation.
  Use shared, inspectable float64 primitive operations rather than opaque
  contractions when the latter bypass the promised range checks.
- Reject invalid/divide/overflow and unsafe intermediate underflow explicitly as
  `FloatingPointError` with operation context. Positive-base powers, positive
  ratios and nonzero-operand products that silently become zero must not erase a
  contribution. Exact algebraic zeros may remain zero; do not call cancellation
  itself an underflow detector or confuse legitimate zeros with range loss.
- Conservative rejection is allowed when an intermediate cannot be evaluated
  safely even if a rearranged final result is representable. Preserve the
  existing time-power underflow rejection regressions: this cycle does not claim
  to rescue all scaled finite products. Do not catch an arithmetic exception and
  return a zero, truncated history or a certificate instead.
- The contract does not assert that range checks bound all cancellation,
  conditioning, summation or special-function errors. Independently demonstrated
  gross point-value failures remain blocking, regardless of false certificate
  flags. Future verified arithmetic/evaluation/input/RHS/output budgets remain
  required for certification.

## Test and review sequence

1. Preserve the pre-repair checkpoint and review this contract before modifying
   numerical code. Keep its owner authorization separate from technical approval.
2. Add all six canonical failures to the default regression suite, retaining
   independent references and original raw failed evidence. Require successful
   recovery for the canonical examples; separately test explicit rejection for
   genuinely unsafe range configurations. Snapshot exact tests/source before RED.
3. Add sign/coupling, endpoint identity, severe panel/time/amplitude ratios,
   shifted origins, near-zero/near-one orders and matched float32/float64 input
   checks. Distinguish new actual RED tests from already-passing compatibility
   assertions. Never reinterpret an unchanged passing assertion as a RED cycle.
4. Implement only this shared-arithmetic repair. Exercise both APIs and sibling
   matrix/power/reduction paths. Do not relax old tolerances or overwrite fixtures.
5. Freeze corrected source, run focused/adversarial checks, independently review
   the complete corrected helper and run full CPU/CUDA suites with actual process
   exits and exact collection/source readback. Documentation/static checks and
   record integrity are separate gates.
6. If the one new final-source review finds a blocker, preserve it and stop for
   another owner decision. Do not turn a full-suite pass into source approval.
   Even a passing scoped artifact review leaves `arithmetic_verified=False`,
   `total_numerical_certified=False`, A1 open and per-state solver scope undecided.

## Delivery limits

Any later commit/push requires the relevant successful review and verification;
no phase closure, release, tag or Phase B authorization is implied. Track fixes,
failed hypotheses, commands/results and remaining limits in VAL-011/SCI-001 and
the linked audit. Keep credentials and personal messaging identifiers out of
repository evidence; progress receipts belong outside the repository.
