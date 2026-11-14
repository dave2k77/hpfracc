# A2 compression-budget tracer: preparation contract

Status: **corrected DRAFT, pending independent FINAL review and owner approval;
implementation not authorized; A2 acceptance false**. This is preparation only.
Baseline: `a3951bde6a633f22ee2c1e006ea78c62421f599b`.
Protocol ID: `a2-glrl-exact-reference-v1-draft`. The closed constants below are
predeclared proposals for re-review, not an approved implementation freeze.

The original failed reviews and reviewed draft remain immutable in
[the failed-review preservation record](../validation/audits/a2-preparation-2026-11-11/initial-contract-reviews-preservation.json)
and its archived bytes. They have **not** been relabeled as passing. The exact
finite mathematics in section 4 is unchanged. Final preparation repair cycle
2 of 2 addresses both failed corrected reviews, including all MATH-REREVIEW and
V-B/V-S01 findings. The rejected corrected packet and full verdicts remain in
`corrected-contract-reviews-failed.tar.gz` with its preservation record; neither
review is relabeled passing. No further automatic repair cycle is authorized.

## 1. Purpose and unchanged acceptance boundary

Prepare one private, CPU-first tracer: independent exact finite compression
assessment, exhaustive evaluation of supplied candidates and deterministic
minimum configured-pole/ID ranking. There is no selector, validator or numerical
worker implementation in this preparation. Proposed later files remain
`benchmarks/numerical/_compression_budget.py` and
`tests/unit/test_compression_budget.py`; this document does not authorize them.

The [A2 programme](phase-a-completion.md#a2-history-approximation-and-memory-open)
remains authoritative. This tiny diagnostic is not an alternative acceptance
scope. A1/A2/A3/A4, scientific certification, Phase A completion, broader
improving-fit refinement and measured CPU/CUDA forward/reverse peak memory remain
open. Per-state solver scope and Phase B questions remain unanswered. SCI-003
and the private trajectory-defect helper are not reference oracles for this work.
Full-history defaults, public APIs, lazy JAX imports, JAX as sole runtime
dependency, fitting/application code and global precision policies are unchanged.
Research software only; no clinical or diagnostic claim.

## 2. Closed, portable preparation packet

All support lives under
`docs/validation/audits/a2-preparation-2026-11-11/`:

- [protocol.json](../validation/audits/a2-preparation-2026-11-11/protocol.json):
  scope, ordered status precedence, exact candidate identity, caps and accounting;
  includes the 125 pinned numerical-input identities and index-entry digest.
- [fixtures.json](../validation/audits/a2-preparation-2026-11-11/fixtures.json):
  exact case IDs, raw input/fit IEEE hex, dtype/shape/layout/hash, configuration,
  ordered candidates, reduced rational budgets, declarative negative mutations,
  expected outcome roles and complete future evidence/comparison inventories.
- [comparison-policies.json](../validation/audits/a2-preparation-2026-11-11/comparison-policies.json):
  every working comparison operand, reference, dtype, shape and tolerance;
  objectives, scalar-order handling and all predeclared FD steps.
- [record-schemas.json](../validation/audits/a2-preparation-2026-11-11/record-schemas.json):
  closed required-key/type/enum record definitions **plus the mandatory canonical
  parse and semantic predicates**. JSON Schema alone is insufficient.
- [packet-notes.md](../validation/audits/a2-preparation-2026-11-11/packet-notes.md):
  case expansion, evidence binding, resource-control and replay requirements.
- [schema-control-contract.md](../validation/audits/a2-preparation-2026-11-11/schema-control-contract.md):
  normative occurrence/stage, disposition, full-history, provenance-DAG, phase,
  output ownership and byte-lifecycle interlocks.
- [schema-control-fixtures.json](../validation/audits/a2-preparation-2026-11-11/schema-control-fixtures.json):
  labeled STATIC/HYPOTHETICAL REPRESENTABILITY witnesses and tamper recipes;
  these are not actual worker/process/fitter outputs.
- [packet-manifest.json](../validation/audits/a2-preparation-2026-11-11/packet-manifest.json):
  final path/byte/SHA-256 identities, external to every file it hashes; it does
  not hash itself. No self-referential hash is required.

Packet paths are relative to this audit directory unless explicitly named
`repo_path`, which is relative to the repository root. No scratch executable,
absolute workstation path or original review directory is a replay dependency.
The packet is static input/specification, not an observation or a shipped replay
CLI. A portable implementation and tamper-tested semantic replay remain future
gates after separate authorization. If prose and packet disagree, stop and
re-review; do not choose the more permissive interpretation.

## 3. Initial reference domain and proposed resource limits

Exactly scalar alpha=1/2, uniform dt=1/4, lower terminal zero, zero GL/RL
prehistory; GL and its current RL alias use the same finite reference. Shapes are
(N,) or (N,M), 1<=N<=17, 1<=M<=3, actual float32/float64 values decoded exactly.
No decimal-intent substitution. The 204 EXACT-GRID cases cover every N/M pair in
both dtypes and aliases; scalar signals and the explicit stacked case are also
frozen. This is not a continuous-operator accuracy claim at the initial sample.

Eligible candidates have method `scaled_tail`, kernel `gl_prefix`, genuine bool
`exact_head=True`, matching N/order/dt and complete authenticated producer
binding, 1..8 poles, one-dimensional equally sized finite c/rho arrays, c>=0,
0<rho<=1. The existing `_SOEFit` does **not** store alpha/dt/t_max; capture the
source-bound producer call rather than infer those fields from that record.
Actual-fit calls use `_fit_soe(alpha,17,n_poles=p,t_max=1,kernel="gl",
method="scaled_tail")`, unchanged 400 updates and dimensionless rate grid.
`scaled_tail` does not use t_max as an independent accuracy control; preserve it
as supplied configuration, not an attainability parameter.

| Proposed per-request limit | Value |
| --- | --- |
| Supplied candidates / configured poles per candidate | 8 / 8 |
| Exact rational primitives, cumulative | 100,000 |
| Normalized numerator or denominator | 32,768 bits |
| All serialized per-request evidence, including raw inputs and failure index | 16 MiB |
| Worker wall watchdog, launch through confirmed exit | 60 seconds |
| Pre-work input bundle / separate failure-index reservation | 256 KiB / 512 KiB |
| Individual raw array / tau numerator or denominator | 4 KiB / 256 bits |
| ID / ordinary ASCII metadata string | 64 / 256 bytes |
| Inventory envelope / source entries / evidence slots per request | 32 / 125 / 2,048 |
| JSON depth / node count / each stdout or stderr capture | 16 / 32,768 / 16 KiB |

All are **proposed diagnostic caps**, not measured feasibility, global numerical
accuracy, memory-complexity or peak-memory guarantees. Beyond the tiny finite
reference domain is `unsupported_reference`, not a mathematically invalid
problem or changed public support. Empty candidate list is invalid; 9..32
inventoried candidates or >8 poles are unsupported assessments. Inputs beyond
bounded transport/inventory are `input_envelope_exceeded`/indeterminate, with a
bounded failure receipt; do not promise unlimited raw-byte/ID retention for an
unadmitted malicious envelope. For every admitted case, persist all supplied IDs
and its complete slot inventory **before arithmetic**.

Count each exact rational add/subtract/multiply/divide/abs/comparison as one,
including shared reference setup, every candidate, signed/prefix/residual
cross-checks, all qualification comparisons and analytic gradient/transpose
work. Repeated pole powers are charged multiplications; no hidden uncharged
power/dot/sum. Integer work in the independent binomial cross-check is charged
likewise. Validate operand bits before work and normalized result bits after;
normalization/GCD and transient integer products are not covered by a primitive
count (a product can reach twice the operand bit cap). Parsing/decoding/hashing/
serialization are wall/byte bounded, not rational primitives. Shared results may
be reused free only if no new arithmetic occurs. Mode-specific exact arrays and
any rerun charge the same cumulative request counter.

The byte limit includes **all files** produced for the request: input/fit/working
and exact arrays, descriptors, metadata, hashes/manifests, commands, bounded
logs, partial files and the reserved failure index, including whitespace and
newlines. Referenced frozen packet/source bytes need not be copied; any copies
are charged. Precheck each bounded serialized result before committing it.
Never hide required arrays or omit charged files to fit a cap.

## 4. Exact finite mathematical contract

Use stdlib exact rational arithmetic for decoded samples and fit arrays. At
alpha=1/2 and dt=1/4 the normalization is exactly `s=2`; no Gamma, quadrature,
working-precision fitter objective or SCI-003 diagnostic is a reference.
For each time index and real feature independently:

```text
w[0] = g[0] = 1
w[k] = w[k-1] * (k-1-alpha)/k
g[k] = g[k-1] * (k-alpha)/k
v[0] = x[0]; v[j] = x[j]-x[j-1]
g_hat[0] = 1
g_hat[k] = sum_p c[p]*rho[p]**(k-1), k >= 1
epsilon = max_{0 <= k < N} abs(g_hat[k]-g[k])
D[i] = s * sum_{k=0}^i w[k]*x[i-k]
D_hat[i] = s * sum_{k=0}^i g_hat[k]*v[i-k]
B[0] = 0
B[i] = s * epsilon * sum_{j=0}^{i-1} abs(v[j]), i >= 1
```

For N=1, epsilon=0. The identity `g[k]=sum_{j=0}^k w[j]` gives the exact
signed-GL/prefix-increment equivalence. The exact head makes the current-increment
residual zero; the remaining terms obey `abs(D_hat[i]-D[i]) <= B[i]` by the
triangle inequality. Retain the initial value in v[0]. Cross-check every output
against both the independent signed full sum and the residual-weighted sum;
never rely on favorable endpoints, signed variation or floating sums of products.

A finite nonnegative exact request tau is in derivative-output units. A valid
candidate qualifies iff every time/feature B is <= tau exactly. Decimal display
rounding cannot determine qualification, and no floating slack is allowed.
Bound rejection does not prove actual error exceeds tau or another fit cannot
meet it. Separate kernel epsilon, exact finite compression error, working-output
error against D_hat/D, and any later continuous-discretization error.
An exact finite compression comparison is not a working-roundoff certificate.

## 5. Explicit decision precedence and identity

### DECISION-01: ownership and precedence

1. Global request syntax/type/tau errors and duplicate IDs or complete content
   make the request `invalid_request` (no selected ID). Retain all observable
   reasons. Genuine canonical fields reject bool/float integer lookalikes and
   integer boolean lookalikes; candidate-local malformed fields are isolated as
   candidate errors, not silently coerced into the global envelope.
2. A well-formed unsupported REQUEST (e.g. order 1/3, nonuniform, Caputo or
   complex input, N/M outside the tiny domain) makes the whole request
   `indeterminate/unsupported_reference`.
3. In a supported request, authenticated provided candidate bytes with known
   L1/kernel/method/head/N/order/dt/producer mismatch, malformed shapes/types,
   nonfinite c/rho, negative c or invalid rho are `known_ineligible`. They do not
   turn a valid GL request into an unsupported request. Continue the other
   candidates. Eligibility checks still run for zero signals and singletons.
4. MISSING/CORRUPT raw artifacts, failed claimed byte digest/length, unavailable
   producer binding, unsupported assessment, exhausted resources or incomplete
   required evidence are `unknown`; the whole request is `indeterminate` even
   after another qualifier. A supplied authenticated NaN is a known bad fit;
   an unreadable file is **not** evidence of a bad numeric fit. An explicitly
   supplied malformed array shape is known-ineligible, not a missing artifact.
   Known producer/config incompatibility differs from missing/corrupt producer
   provenance. Request input artifact integrity failure is globally unknown.
5. Only after every required assessment is known, select minimum
   `(pole_count, stable_id)` among eligible qualifiers. Otherwise return
   `selected_id=null` and `no_supplied_candidate_meets_bound`. This does not mean
   no fit can meet the budget; bound rejection need not imply actual-error
   failure. No fallback, first-success or monotone-fit-quality assumption.

### DECISION-02: canonical content identity

IDs match ASCII `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`, case-sensitive unsigned-byte
lexical order, at most 64 bytes. No Unicode/locale normalization. Equal IDs OR
equal **ID-free** content tuples invalidate the request, including identical
content renamed under different IDs. Same pole count alone is not a duplicate.

The ordered tuple is `(configuration, producer, rhos_descriptor,
coefficients_descriptor)` with every nested field fixed in `protocol.json`.
Configuration includes N/P/kernel/method/exact_head/alpha/dt/t_max/grid/lower
terminal/prehistory. Producer includes kind, entrypoint, source repo path/hash,
lock/policy, iterations/grid, backend, working dtype, x64, JAX version, device
class and mode. Each array descriptor includes dtype, exact shape, byte order,
C layout, byte count and verified SHA-256 of raw bytes. Tuple JSON has sorted
object keys, compact separators and canonical types. Explicitly excluded:
stable/case/request/observation/run IDs, PID/times/elapsed/command/exit/hostname,
artifact location and logs; no other untyped provenance or extra metadata exists.
Do not normalize signed zero, pole permutations, zero coefficients or equivalent
kernels. Missing content cannot be hashed into a guessed identity: it is unknown.

### DECISION-03: mandatory EXHAUSTIVE policy, not a theorem of uncertainty

Any unknown assessment makes the result indeterminate whether its rank is lower
**or higher** than a known qualifier. A known lower minimum cannot mathematically
be displaced by an unknown strictly higher rank. The stricter outcome here is
nevertheless mandatory because this tracer requires exhaustive completed
assessment, not because a minimum is universally unknowable. Both rank cases
are frozen fixtures. Pole count is a configured cost proxy, not measured cost.

## 6. Closed fixture matrix and outcome roles (B1)

`fixtures.json` enumerates 560 request cases: 204 EXACT-GRID, 32 FIT-REAL,
28 REF-CAPS, 16 REF-CORE, 148 REF-EDGES, 100 REF-RANK, 16 REF-SIGNAL and
16 narrowly scoped SCHEMA-CONTROLS (rank-two M=1 and combined precedence).
There are 10 separate raw parser controls and 4 identity controls. The static
library contains 840 raw arrays and 160 candidate specs; 12 are future real-fit
producer specs with **no arrays observed**. The packet preinventories 24,336
evidence slots and 1,312 working-comparison slots; no case exceeds 954 evidence
slots. These are planning identities, not executed test/worker counts.

There are 1,056 fit-binding checks and 64 candidate-free future full-history
observations. Reference-list mutations precede dereferencing; every supplied
occurrence is cloned. Empty-list cases now have no stale candidate inventory;
duplicate invalid raw occurrences retain their IDs and distinct indices.

REF-FAIL remains x=(1,1,1), c=rho=(1/2), tau=3/16: epsilon=1/8,
B=(0,1/4,1/4), D=(2,1,3/4), D_hat=(2,1,1/2), no qualifier. REF-PASS adds
c=(1/4,1/4), rho=(1/2,1), exact over three lags; selected only among the supplied
list. Equality tau=1/4 qualifies; tau=1/4-2^-100 does not. Never convert that
comparison to binary64. A distinct one-pole c=1/2,rho=3/4 is also exact over three
lags, so no global pole-minimality claim follows from REF-PASS.

All 24 permutations of the four-candidate rank list are explicit in each
dtype/alias, including same-pole lexical ties, a larger exact candidate and a
bad smaller candidate; separate good-smaller/bad-larger controls reject monotone
quality assumptions. Zero, singleton, changed and stacked signals, missing
intermediate evidence rows, canonical invalid types/tau, duplicate IDs/content,
known incompatibilities, nonfinite provided arrays, missing/corrupt artifacts,
unknown higher/lower rank, outside-domain and controlled resource outcomes are
concrete requests/mutations, not headings awaiting favorable examples.

REF-SIGNAL and FIT-REAL use N=17, ramp k/16, step 1[k>=8], alternating
(-1)^k/8, each separately and stacked (17,3), in **both float32/float64 and
GL/RL**. FIT-REAL has no zero signal. Its supplied order is always poles (8,2,4).
For EACH declared signal/dtype/alias and mode, tau=1 requires at least one real
qualifier; tau=0 requires no qualifier. These are **predeclared preserve-and-stop
gates, not observed or proved attainability**. Do not invent arrays, guess the
winning pole count, change signal, skip cases or retune if contradicted. Preserve
all actual returned candidates and exact assessments first.

The prefix bound is monotone in time per feature: final B alone suffices for the
mathematical threshold. Requiring every intermediate row is an evidence-
completeness rule, not a claim that omitting one necessarily changes selection.
Actual error need not be monotone. Shortcut controls distinguish these facts.

## 7. Working and derivative comparison contract (B2)

All policies are frozen **proposals for re-review**, narrow empirical gates, not
roundoff/gradient bounds or extensions of B. Check every finite component with
exact equal shape/rank, **no broadcasting**:

`abs(a-r) <= atol + rtol*abs(r)`

`a` is the named observed operand and `r` the named reference in
`comparison-policies.json`; rtol never silently uses the other operand. Compare
decoded represented observations to exact rational targets without first
rounding the latter to the worker dtype. M=1 for shape (N,).

| Comparison | float32 rtol / atol | float64 rtol / atol |
| --- | --- | --- |
| Recursive vs own exact D_hat; full vs D; JIT vs eager forward | 2e-5 / 2e-5 | 1e-10 / 6e-10 |
| Input AD vs own analytic exact gradient | 2e-5 / 2e-5 | 2e-7 / 2e-8 |
| JIT vs eager input and order gradients | 2e-5 / 2e-5 | 2e-7 / 2e-8 |
| Input directional AD vs central FD; adjacent FDs | 2e-3 / 2e-4 | 2e-6 / 2e-7 |
| Order AD vs central FD; adjacent FDs | 5e-3 / 5e-4 | 2e-5 / 2e-6 |

Base objective is mean(D_hat squared) over N*M entries. For each mode's actual
base fitted kernel, set g_hat[-1]=0, w_hat[k]=g_hat[k]-g_hat[k-1] and
L_hat[i,j]=2*w_hat[i-j] for j<=i, else zero. The exact input gradient is
`2/(N*M) * L_hat.T @ D_hat`, applied independently to each feature and retained
in input shape. This is the explicit analytic target for **both dtypes**.

Choose q=cos(x_base) once in working dtype, save its raw bytes and hold it fixed
with stop-gradient, including input perturbations. Directional AD is sum(grad*q).
For input derivatives the fit is fixed: each same-mode base/perturbed composition
must return identical actual base fit bytes or stop. For order derivatives,
hold selected configuration fixed but refit at every a. Strong scalar alpha,
dt and h have shape [] in the target dtype; JIT takes alpha dynamically. Recompute
`dt**(-a)` **inside** every differentiated/perturbed objective; 2 is exact only
at base alpha=1/2. Do not freeze coefficients or reselect inside AD/FD.

The source-bound private worker composition is fit -> apply -> scale, returning
its actual rhos/coefficients as auxiliary arrays for EACH eager/JIT/AD/FD
invocation. The selected ad-base observation jointly returns both input/order
cotangents and actual forward auxiliaries, separately from base selection.
Retain its own arrays and require authenticated same-mode input/fit/alpha/dt
identity before reusing an exact reference cache. Base auxiliary arrays used
for gradient/reference binding must match
the same-mode base output producer; any mismatch stops. No eager-array borrowing
for JIT, and no claim that a public SOE wrapper has been instrumented. Preserve
branch-specific producer/config/alpha/dt/input/q/h, actual arrays, outputs,
objectives, cotangents, FD values and coefficient-zero patterns.

Float32 h=(1/128,1/256,1/512), all three; float64 h is the actual binary64
encoding of 1e-5 retained in the packet. For each mode, ALL steps must meet the
AD policy and every adjacent pair the corresponding FD policy, using larger-h
FD as reference. Float64 has no adjacent pair: explicitly not applicable.
Fixed-fit input objectives are quadratic, so central FD has **zero real-
arithmetic truncation error**; floating perturbation/evaluation remains.
O(h^2)+O(u/h) is only a sufficiently smooth order-objective heuristic, not a
bound for projected-fit kinks or active-set changes. Final zero-pattern stability
is a limited diagnostic, not evidence that intermediate active sets agree.
Any unstable/nonfinite/nonsmooth/unresolved comparison: preserve and stop,
without tolerance relaxation or scientific acceptance.

Scope x64 off/on outside JIT for float32/float64 workers and restore it. Assert
actual input/alpha/dt/h/q/perturbation/fit/output/objective and both cotangent
dtypes/shapes; a float32 cotangent alone does not establish float32 forward
execution. All host rational work has a separate context from the future JAX
worker. No working stage has been run by this preparation.

## 8. Closed records and bounded partial evidence (B3/B4)

All required typed keys and enums are in `record-schemas.json`, with
`additionalProperties=false`. Its canonical predicate rejects duplicate JSON
keys and NaN/Inf tokens, integer 1.0/true and boolean 0/1 coercion. Nonfinite
provided IEEE bytes remain distinct from nonfinite JSON syntax. Field failure
belongs to its request/candidate ownership boundary; unsupported finite domain
values are not relabeled malformed by a restrictive schema. Raw malformed input
is retained in the bounded input bundle, not coerced into a canonical record.

Rationals are `{num,den}` lowercase hexadecimal strings, no plus/0x/leading
zeros/-0, positive denominator, gcd=1. Zero is uniquely `{num:"0",den:"1"}`.
Check field lengths before conversion (internal 8,192 magnitude hex digits,
plus optional numerator minus), then normalized bits. Tau is finite nonnegative
and capped at 256 bits per part. Raw arrays require dtype, shape, explicit byte
order, C layout, byte count, hex bytes and SHA-256 with exact size relationships.
Every supplied candidate, observation branch, quantity and comparison has a
unique preinventoried slot; missing rows cannot become a completed rejection.
Unavailable slots use null plus explicit unknown/not_evaluated/reason, not zero.
Nonselected or genuinely unused slots use explicit not_applicable and reason.

The mandatory typed disposition table preserves actual known-ineligible reasons,
not a conflicting generic not_evaluated reason. All observable reasons survive
combined failures. Transport shapes 0..1024 (rank <=2) are metadata only: admitted
N=18 expectations [18] and [18,18] remain exportable without extending the tiny
numerical domain. Full-history observations use a candidate-free configuration
and actual full-history source, not invented fitter attribution. Standalone slot
artifacts have record_type=slot and are accepted by the root schema. Every emitted
artifact has a closed route. Exact producer-binding canonical UTF-8 bytes bind
invocation/run/process/case/input/source/config/mode/branch/actual arrays in an
acyclic DAG; hashes establish consistency, not execution authentication.

Before arithmetic, persist the bounded raw input/config/source bundle, all IDs
and slot inventory; initialize the separately reserved failure index. Serialize
results into bounded buffers, check remaining bytes, commit arrays atomically,
then update index. The initialized and post-kill index also retain ALL unfinished
comparison and fit-binding dispositions. On cap/kill, retain completed arrays and valid partial prefixes
within budget; explicitly inventory missing remainders and ALL unfinished slots.
Do not promise complete exact arrays after exhaustion, unlimited failure logs,
or no-supplied-candidate/minimum results. Keep partial file bytes charged too.

Supervisor wall time starts immediately before process launch and includes
imports/decode/arithmetic/serialization/flush until confirmed exit. At 60s,
terminate, allow separately recorded 1s grace, then kill if needed; timeout is
indeterminate. Failure-index write/kill latency uses its reserved bytes and is
separately recorded, not hidden in a claimed 60s total guarantee. Bounded preflight
input capture is separately timed. Concrete controls set operation cap 0, bit
cap 1, evidence cap 786,432 bytes (only input/index reservations), watchdog 0
with a startup barrier, and an explicit fault checkpoint after candidate 0's
qualifier commit. The last is fault injection, **not** measured exhaustion or
feasibility. No exhaustion protocol has been executed here.

Typed phase receipts distinguish preflight, watchdog, grace and
kill-through-confirmed-exit; a later supervisor-export receipt measures the
completed index write without self-digest/write-duration cycles. Unavailable and
not-applicable timings are explicit. The zero-watchdog barrier is never released.
No 60-second total elapsed guarantee is made. Fresh privately owned per-request
roots/subdirectories, no-follow handle-relative creation/readback, containment,
regular-file/ownership checks, immutable no-overwrite arrays and atomic owned
index replacement are mandatory future invariants, not an existing exploit claim.
Temporary/partial/coexisting files count toward live AND cumulative 16 MiB writes;
no hidden/deleted/unowned-file or unlimited-rewrite loophole is permitted. The
whole index transaction (including coexistence and later receipts) must fit its
512 KiB reservation, otherwise bounded reservation_exceeded/indeterminate before
arithmetic. Full templates must include every field/check, not omit them to fit.

The preserved independent reviews established only their schedules: 612 exact
synthetic Fraction cases/7,344 outputs; a represented-extreme charged schedule
used 55,480 operations with 32,768-bit/16 MiB caps plausible for that schedule.
Neither is a general feasibility certificate. The earlier 136 parent rows are
planning arithmetic, not production results. This revision's own scratch checks
are also preparation only. No measured fit/application/AD or peak-memory result
can be inferred from any of these counts.

## 9. Preparation exit gates and unresolved limits

Independent re-review of this corrected digest and packet is still required,
including empirical-policy sufficiency, closed schemas, byte accounting and
resource controls; then explicit owner approval before implementation. Portable
semantic replay/tamper controls, behavioral RED, numerical workers, all future
FIT-REAL outcomes and working comparisons remain unimplemented/unmeasured.
Documentation checks/build are separate from scientific acceptance.

Final-cycle static representability self-checks cover 17 linked hypothetical
bundles, 38 typed status witnesses, all 17 schema roots/status paths, all 64
future full-reference bindings and non-no-op structural/semantic tamper controls.
They do not run a worker or observe real fit arrays. Full transaction templates
also expose a residual limit: eight float32 tau=1 FIT-REAL templates exceed the
conservative 512 KiB whole-index-transaction reservation (maximum 792,318 bytes;
complete metadata can exceed the 32,768-node envelope). They must stop with
bounded reservation_exceeded/indeterminate, not omit comparisons/checks or relax
caps. The detailed change/check ledger and limitations are in the linked
schema-control contract. This is not a demonstrated FIT-REAL pass or universal
admission guarantee; independent FINAL review and owner approval remain open.

`implementation_authorized`, `a2_acceptance_passed`, `arithmetic_verified` and
`total_numerical_certified` remain false in the preparation packet. No current
core/public/default/dependency change, full/GPU benchmark, commit/push or closure
of A2/Phase A or unanswered scientific/per-state/Phase B scope is authorized.
