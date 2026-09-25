# Internal decision service

`DecisionService` composes the existing `AdmissionAdapter`, a concrete backend and
`SerializedWorker`. It exposes Python methods, not an external HTTP endpoint.
Platform-owned verifier, preparation and reservation callbacks remain mandatory.
The runtime descriptor comes from trusted startup and must describe the loaded
checkpoint/backend; a `tt-blackhole` label does not implement or validate a TT port.

Construction starts in `starting`. Call `start(self_test)` with a callback that
actually executes a representative decision on the loaded model and returns exactly
`True`. A failed test moves to `failed`; there is no permissive default or automatic
device fallback. Only `ready` admits requests. Backend exceptions or invalid output
withdraw readiness and prevent further queued backend executions. A new service
generation is required after failure.

`submit(raw_request_bytes, authenticated_envelope_bytes)` validates the platform
grant, performs trusted tokenization, consumes the platform reservation and queues
work under the verified tenant and execution attempt. A successful Future contains
the bundled `decision-response` contract. Question identities/order, actual prepared
token counts, native answer validity and the entire response schema are checked.
Response objects are independent copies; no response cache or prompt logging exists.

Preparation has its own bounded pool: `max_admitting` defaults to eight concurrent
admission operations and must be a positive integer. Capacity is checked before
parsing, authentication, tokenization or reservation consumption. Excess work raises
`AdmissionCapacityExceeded`; it does not wait in an unbounded preprocessing queue.
Every admitted preparation releases its slot on success or failure. Worker request
and token-cost queues are separately bounded. This pool bounds local concurrency,
not authenticated per-tenant fairness before tokenization; gateway admission/rate
policy remains required.

`queue_ms` measures the observed interval from worker submission to backend start.
`execution_ms` measures the observed backend call, including its decoding. Both are
monotonic elapsed time rounded down to whole milliseconds. Preparation/admission
time and post-backend schema checking are outside those intervals. `encoded_tokens`
comes from validated preparation and must agree with the backend. `output_tokens`
is zero because this is a decision model. `accelerator_ms` is omitted: host elapsed
time cannot establish accelerator usage, including on a real TT backend.

`drain(wait=True, cancel_queued=False, timeout=None)` closes admission and waits for
the model, active preparation and startup self-test. False means ownership remains;
it never authorizes card reassignment. Future cancellation stops delivery, while
the worker retains in-flight payload/device ownership until execution returns.
A racing admission may consume a reservation and then be rejected by draining.

## Platform work still required

- Persist/reconcile reservation outcomes when queue admission fails, draining races,
  deadlines expire, clients cancel or a process crashes after reservation consumption.
- Persist usage with authenticated tenant, request, execution attempt and usage event
  identity; reconcile executed work even if response delivery was cancelled. The
  response usage object is not a durable ledger, billing event or exactly-once proof.
- Publish readiness through Compute/Management, reconcile runtime generations and
  exclusivity, and retain device ownership until actual backend/process teardown.
- Bind the existing shared gateway/reverse-tunnel path to host Management and the
  internal service. Transport authentication, rate policy, secret management and
  public error sanitization remain platform responsibilities.

There is no process-local substitute for durable replay prevention, no TT-to-CPU
fallback, and no claim of measured accelerator usage or hardware acceptance here.


An optional durable [execution journal](execution-journal.md) now records local
admission, execution and receipt outcomes, including cancelled delivery and process
crashes. It does not implement the external reservation authority or authenticated
receipt receiver. Without an explicitly configured receipt sink, this service
retains the in-memory behavior described above.

When the journal is configured, startup checks for unresolved prior execution
before invoking the model self-test. Recovery failures leave readiness failed.
Optional [runtime readback](runtime-readback.md) requires an explicit observed
backend provider and never marks a service ready.
