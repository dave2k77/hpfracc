# A2 corrected preparation packet: reading and replay contract

**Corrected DRAFT; independent FINAL review and owner approval pending.** Static
planning data only. Implementation authorization and A2/scientific acceptance
remain false. No fixture value is an observed production-fitter output.

## Files, lineage and portable resolution

Read `protocol.json`, `fixtures.json`, `comparison-policies.json` and
`record-schemas.json` together with the
[preparation contract](../../../developer/a2-compression-budget-protocol.md).
All local references resolve from this audit directory, except `repo_path`
which resolves from the repository root. A fixture `_ref` indexes the exact
named library object; no external array is implied. Source hashes cover the
unchanged repository source, not a synthetic-fitter implementation.

`packet-manifest.json` lists exact file byte counts and hashes after edits and
excludes itself. It is a proposed review identity, not an approved freeze or an
execution receipt. Its included protocol/source identities do not authenticate
worker observations. The preserved `initial-contract-reviews-failed.tar.gz` and
`initial-contract-reviews-preservation.json` remain unchanged and failed.
Original review files are accessible from the archive without scratch paths.

## Expansion and explicit expected roles

Each case names actual raw input bytes, shape, dtype and candidate order, exact
rational tau, required comparisons, initial evidence-slot inventory and expected
outcome role. Apply explicitly labeled `references` mutations BEFORE
dereferencing, independently deep-clone every supplied candidate occurrence,
then apply labeled `expanded` and `evidence` mutations at existing targets only.
The exact stages/targets are normative in
[schema-control-contract.md](schema-control-contract.md).
No replace is an insertion and no coercion is allowed. `candidate:i` is supplied-order index;
`request` is global ownership. Mutation `artifact_status=missing/corrupt` withholds
or fails authentication of the referenced bytes deliberately. It does NOT turn
those bytes into an observed malformed numerical fit. A provided raw replacement
has its own frozen bytes/shape/SHA; malformed-shape controls keep deliberately
inconsistent dimensions as authenticated supplied metadata, not silently repair
it. A complex request replacement has interleaved real/imaginary IEEE bytes and
is unsupported by the tiny reference even though structurally decodable.

Negative syntax/type/canonical fixtures deliberately are not valid canonical
record instances; preserve raw request bytes in `input_bundle` and a null parsed
request when needed. A canonical export of an error need not pretend its input
satisfied the success-record schema. Unknown producer bindings are not completed
ineligibility findings; authenticated incompatible producer configuration is.

Every required slot is explicit. Request expansion sets `record_type="request"`,
`request_id=case_id`, `case_id` from its enclosing fixture, `policy_id` from the
packet and `packet_manifest_sha256` from the external final manifest. Dereference
`input_ref`, ordered `candidate_refs` and `caps_ref`; retain every other explicitly
specified request field. No provider/default fields are inferred. Actual producer
context (backend/device/JAX version/mode and actual returned arrays) is captured
from the future worker, not invented during expansion. Source lists need only
the actual producer/lock/specification bindings; the complete 125-file numerical
pin remains in the shared protocol and need not be duplicated into every bundle.

Every required slot is explicit. Case ID plus slot ID is the unique evidence
key. Full-output reference observations have candidate ID null; they are not
attributed to a selected SOE candidate. For each actual-fit case, BOTH modes
assess all three base candidates. The tau=1 working stage preinventories every
candidate's branches; after base exhaustive selection, unselected AD/FD branches
are explicit not_applicable. Base forward comparisons still apply to ALL
candidates. Tau=0 stops after base no-qualifier gate: no derivative stage is
implied. Full base references remain required for both tau requests.

Full-reference observations use the disjoint full_configuration/full_producer
variant and pinned fractional.py full-history source, with null candidate ID AND
index. All 64 future bindings are explicit in required_full_observations; none is
an observed result. Duplicate invalid supplied occurrences retain duplicate IDs
and distinct indices. Empty supplied lists have only shared slots, not stale
two-candidate inventories. Both remain invalid raw requests, not coerced records.

For every mode/branch, preserve actual returned fit arrays, not copied eager
arrays or a later replay fit. The same source-bound composition must return
fit/output/objective auxiliaries; gradient auxiliaries must bind to that exact
same-mode base producer, or stop. Recompute/refit at perturbed order without
reselection; base strong alpha and dt encodings are frozen in `working_scalars`.
FD h encodings are in `comparison-policies.json`. Direction q is a FUTURE raw
cos(x_base) output, not a synthetic precomputed cos array. Input directional AD
uses fixed saved q, and the analytic directional target uses its decoded bytes.

No observation arrays are provided for the 12 actual-fitter configurations.
All 32 FIT-REAL cases bind nonzero ramp/step/alternating/stacked inputs. Tau=1
qualifier and tau=0 nonqualifier are required future stop gates, not claims of
attainability. A contradiction stops the stage, not a search for a nicer signal.

## Shapes and canonical parse (mandatory, not Schema-only)

Input-shaped quantities: input/v/D/D_hat/error/B/analytic input gradient/output/
full output/input AD/direction. Lag arrays: w/g/g_hat/residual/signed fitted
weights [N]. L_hat [N,N], containing scale 2. Fit arrays/zero-pattern [P]. Scalar
quantities (including alpha/dt/h/objective/order AD/directional AD/FD/epsilon)
have shape []. Exact matrix/vector arrays are flat C-order rational lists with
cardinality equal to shape product. Zero-pattern is exact 0/1 rational data,
not an untyped bool-to-number conversion. A partial time-leading prefix has
shape [K,...] with K<N and status partial; missing [K:N,...] is explicitly
unknown in the index using the expected full shape from this packet. Never
call a missing endpoint or component a complete array.

The canonical parse predicate in `record-schemas.json` is part of this contract:
duplicate keys/nonfinite JSON rejected, type(value) is int or bool as appropriate,
closed keys/enums, bounded parse depth/nodes/strings, reduced hex rationals,
explicit byte order/layout/length/SHA. Schema acceptance of integer 1.0 is not
sufficient. Canonical structure and tiny-domain support are separate: finite
N=18/M=4/P=9/complex requests are unsupported, not invalid mathematical problems.
Candidate-local authenticated malformed fields are known-ineligible; global
request syntax/type/tau/duplicates are invalid; corrupt/missing bytes are UNKNOWN.

Failure-index expected/completed shapes use bounded transport dimensions
0..1024, not the numerical N<=17 bound. Unsupported N=18 retains [18] and
[18,18] with null numerical payloads. Explicit (1,1)/(3,1) controls preserve rank
in both dtypes and aliases. The typed disposition table permits actual
malformed_fit/configuration_mismatch reasons for known-ineligible not_applicable
slots and preserves all observable reasons; not_evaluated is not their forced
replacement. Missing producer evidence still makes a locally incompatible
candidate unknown, with both causes retained.

Identity tuple excludes stable IDs and volatile receipt/location fields but
includes every declared configuration/producer/source/dtype/shape/byte-order/
layout/byte-count/raw digest field. Signed-zero bytes and pole order are identity
data. Content-equivalent renamed candidates are forbidden duplicates; equal
kernels with distinct bytes/configs remain distinct. Do not silently canonicalize
mathematics into byte identity. Unknown tuples are not guessed.

## Resource controls and bounded evidence

The proposed 8 candidates, 8 poles, 100k exact primitives, 32,768 normalized bits,
16 MiB evidence and 60s watchdog remain diagnostic caps pending review. Pure
exact planning does not measure fitter feasibility or application/AD peak memory.
Counters include shared reference, cross-checks, analytic gradients and all
mode/branch reference arithmetic. Decode/GCD/hash/export costs are wall bounded,
not free unlimited work. The manifest/source SHA fields and raw arrays belong
to byte coverage, including JSON formatting and partial files.

The reserved failure index contains compact `index_slot` references, never copies
of raw/exact array payloads. Each completed or partial slot record is stored at
`arrays/{slot_id}.json` and referenced by exact byte count/SHA-256, expected shape
and completed prefix shape. Unknown/unused rows carry null references and explicit
reasons. Pre-render worst-case index metadata before arithmetic and refuse admission
if it cannot fit 512 KiB; do not fill the reserve with duplicated rational arrays.

Standalone arrays are record_type=slot root variants. Every emitted artifact has
an explicit path-to-schema route. The initialized/final index also contains ALL
comparison and fit-binding dispositions. Full transaction sizing includes all
new fields, process/phase receipts and old/new index coexistence; if the reserved
transaction cannot fit, bounded reservation_exceeded/indeterminate preflight
refuses numerical work rather than omitting records.

Persist bounded raw input/all admitted IDs plus compact slot inventory before
work, and a separate initialized failure index within the reserved 512 KiB.
The entire input bundle including any raw-request escape copy must fit 256 KiB;
use compact JSON and do not duplicate raw bytes unnecessarily. Oversize envelope
gets a bounded prefix/digest-of-retained-prefix/received-length-if-known failure
receipt, never a falsely labeled full-input digest or promise of unlimited IDs.
No arithmetic begins before successful preflight persistence.

The controlled exhaustion protocol is predeclared, NOT executed:

| IDs (each dtype/alias) | Trigger | Required future receipt |
| --- | --- | --- |
| CAP-ops | rational primitives=0 | first attempted primitive refused; all unknown |
| CAP-bits | normalized bits=1 | normalization of 1/2 exceeds cap; no rejection proof |
| CAP-bytes | evidence=786432 bytes | 256 KiB+512 KiB reservations leave no result allowance |
| CAP-wall | watchdog=0, held startup barrier | supervisor does not release arithmetic; timeout, confirmed exit |
| CAP-after-qualifier | injected fault at candidate-0 commit | retain qualifier; next unknown; whole request indeterminate |

After normal preflight, fault/cap/kill retains each completed array and any
budget-fitting valid prefix. ALL other candidate/quantity slots remain explicit
UNKNOWN/not_evaluated. Atomic temporary files also count toward bytes; failed
large allocations must not evade the byte limit. Supervisor terminates at 60s,
waits separately recorded <=1s grace then kills; complete process confirmation
and failure-index receipt are mandatory. Reservation is not a claim that all
required exact arrays always fit. Never claim no qualifier or a minimum after
an unknown, even if an earlier lower-ranked qualifier is mathematically minimal.

## Semantic replay and limits

A future authorized replay must first validate bounded canonical structure and
source/case/policy/producer/raw-byte relationships, then independently recompute
exact shared identities, signed/prefix/fitted/residual outputs, each B, analytic
input gradient, candidate completeness and rank. Working comparisons use every
named reference/shape/dtype/atol/rtol and all declared FD steps. Only then checksum
completed immutable artifacts and write the external manifest. Missing files,
rebound producer/alpha/dt, swapped eager/JIT fits, duplicate/omitted rows, changed
raw bytes, signed-zero normalization and dropped candidates need explicit tamper
controls. Static schemas/checksums do not implement this replay or authenticate
execution. A portable repository CLI remains a separate implementation gate.

Thresholds are empirical; fixed-fit input FD has zero real truncation while
smooth-order h-squared heuristics do not cover projected-fit kinks. No derivative,
roundoff, continuum, A1/A2/A3/A4, per-state solver or Phase B closure is granted.
