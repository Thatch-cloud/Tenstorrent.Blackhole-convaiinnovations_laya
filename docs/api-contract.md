# Decision API contract v1

Status: executable schema proposal for the shared-service pilot. No platform route,
model numerical parity, physical card acceptance or authentication implementation
is claimed by these files.

The path is client -> Administration shared gateway -> host-initiated reverse
tunnel -> Management -> local Compute-managed Laya runtime. Administration remains
the authority; neither Compute nor this runtime is a public tenant-auth endpoint.

## Native wire contract

`POST /v1/decisions` uses `src/laya_tt/schemas/decision-request.schema.json` and
`src/laya_tt/schemas/decision-response.schema.json` (JSON Schema Draft 2020-12). Require
`model: "laya-english"`; unknown aliases never auto-select another checkpoint.
The deployment resolves this alias to an independently pinned checkpoint revision.
One request has one state and a map of stable question IDs. Iterate questions and
criteria in their received insertion order; never sort them before tokenization.
State can be text, an object or a list; it is model input, never authority metadata.

The native pilot deliberately accepts a strict subset of upstream inputs:
nonempty questions, safe bounded IDs, string instructions and criterion descriptions,
unique choice labels, lowercase boolean keys, and no arbitrary extra fields. Limits
are 64 rows, 64 criteria per row, 4096 characters per instruction/description,
50000 characters for string state. These are admission ceilings, not proven hardware
batch sizes. The runtime may split a request into smaller serialized shape buckets.
Enforce a streaming 2 MiB body cap before JSON parsing, bounded nesting, duplicate-key
rejection and finite numbers. JSON Schema cannot enforce the wire byte/depth limits
or detect keys overwritten by a JSON parser.

Tokenizer admission additionally rejects any row exceeding 512 sequence tokens or
192 head tokens, without silently truncating options or context. The worst-case
encoded-token reservation is 64 * 512; charge/report observed encoded tokens,
including the repeated state for each question row. No generated tokens exist.
Schema character limits do not establish token limits.

Response answers retain the upstream shape and semantics. Choice returns the
selected label and label-keyed probabilities. Score is the expected zero-based
level index, with its legend and distribution. Noul is the probability of true,
not a string or boolean classification. Every answer includes the upstream
confidence fields and action probability. Do not reinterpret confidence as a
single interchangeable metric across all three question types.

Response metadata identifies request, pinned checkpoint, runtime revision/generation
and actual backend. `cpu-reference` is explicit development execution, never a
silent substitute for `tt-blackhole`. Usage includes successful request count,
question rows, observed encoded tokens, zero output tokens, queue and execution
milliseconds. Include accelerator milliseconds only when measured. Pricing and
retention policy remain platform decisions.

Beyond schema validation, verify result IDs/types exactly match the request;
choice keys equal input criteria and selected label exists; score legend and
probability keys match level indices; score lies within the actual level range;
all numbers are finite and distributions sum to one within upstream rounding error
(up to 0.00005 per option, plus numerical tolerance). Never normalize the rounded
upstream values merely to force an exact sum. Full response fixtures must come from
the pinned reference runner; no invented prediction fixtures are shipped here.

## Compatibility mapping

The optional `POST /v1/systemone` adapter is not implemented by these schemas.
Its compatibility target is the pinned upstream source:

- https://github.com/NandhaKishorM/laya/blob/970dc8c5f63d7b886a68409493f37d569424f933/laya/serve.py
- https://github.com/NandhaKishorM/laya/blob/970dc8c5f63d7b886a68409493f37d569424f933/laya/agent.py

Preserve `state`, question IDs, question definitions, criteria order and upstream
`answers` unchanged when converting an accepted compatible request. Choice accepts
a label-to-description object or a label list; score accepts descriptions in level
order; noul accepts optional true/false descriptions and display labels. Keep
`confidence`, `answer_confidence` and `action.act_probability` unchanged. Output
`usage.input_tokens` maps from the reference's actual encoded-token accounting and
`usage.output_tokens` remains zero. Preserve upstream model/routing response metadata
when compatibility is implemented; native runtime metadata must be additive or
out-of-band, not replacements masquerading as upstream values.

Intentional platform policy differences must be documented to clients: the pilot
requires a supported explicit model alias and rejects unknown models rather than
upstream auto-routing; native validation is stricter; no implicit truncation is
allowed. Compatibility must not be advertised as exact for requests outside that
supported subset. Upstream's `english` can be an explicitly documented compatibility
alias for `laya-english`; multilingual and typed-decisions are unsupported here.
Native schema validation must not silently erase upstream extra fields to claim
compatibility. Test any adapter against captured upstream wire fixtures before release.

## Trusted admission context

`src/laya_tt/schemas/admission-context.schema.json` is a SEPARATE internal envelope. It must
never be accepted from public request JSON or merged with state/questions. The
schema describes structurally valid metadata; it does not authenticate it.
Administration supplies tenant/key identity, entitlement policy, resolved checkpoint,
runtime generation, reservation, server-owned request/attempt/usage IDs, request
SHA-256 binding and deadline over an authenticated platform channel. No raw API key
is forwarded. The SHA-256 covers exact validated public-request UTF-8 bytes retained
for transport; intermediaries must not reserialize those bytes. If a different
canonicalization is adopted, version it explicitly across all producers/consumers.

Before queueing, Management must verify trusted issuer/channel, tenant/key binding,
entitled model/revision, policy freshness, target runtime generation, payload digest,
nonexpired deadline (issued <= now < deadline), reservation ownership and remaining
row/token budgets. Admission aggregates all API keys for the tenant. A structurally
valid envelope claiming tenant B is not evidence it was authorized for B. Context
validation is not a substitute for those checks. Replay/reservation consumption and
usage receipt dedupe must use server-owned identities, never caller correlation IDs.
Convert remaining deadline duration to the worker's monotonic clock at admission;
never compare Unix milliseconds directly to a monotonic timestamp.

The native JSON schema excludes caller authority fields at the request/question
level. Such field names inside arbitrary state are legal data and must remain
opaque. Tests explicitly show both tenant contexts can satisfy a schema: integration
must prove issuer/tenant/payload binding rejects a spoofed context before worker
submission. This prevents confusing schema validation with authentication coverage.

## Errors, evolution and checks

The public gateway proposal uses the platform machine-readable error envelope
and request correlation. Its target mappings below are not yet implemented by a
public gateway. The internal runtime transport has its own explicit status mapping
in [internal-asgi.md](internal-asgi.md). Native
malformed/unsupported inputs: 400; absent identity: 401; denied entitlement: 403;
unknown model/task: 404; body/token limits: 413; tenant quota: 429 with Retry-After;
no capacity/queue saturation: 503; expired execution deadline: 504. Compatibility
may require upstream 422 for invalid question definitions; freeze that difference
with adapter fixtures. No exception traces, input text or credentials in errors.

Schemas are version 1 and closed to unknown properties. Changes to accepted fields
need coordinated schema/consumer releases; do not assume additive fields are accepted
by old closed-schema clients. Serve discovery/capabilities only for ready observed
runtimes and their supported contract/shape versions.

Run `python -m unittest discover -s tests -p test_contracts.py -v` with `jsonschema`
installed (CI-tested with 4.26.0). Tests exercise all question shapes, malformed inputs,
resource ceilings, injection and envelope separation. They contain request fixtures
and synthetic metering shape only, no model probability claims. Remaining integration
gates include authenticated context binding, tenant fairness across keys, ordered
response binding, deadline/cancel behavior, durable idempotent usage, and numerical
reference/hardware parity.
