# Shared runtime lifecycle

`laya_tt.worker.SerializedWorker` serializes one backend callable. It is a local
execution primitive, not an authenticated server, scheduler, or device allocator.
One trusted model process holds one exclusively allocated P150A. Sharing happens
between requests to that process; no subdevice tenant-isolation claim is made.

## Admission boundary

The platform adapter authenticates the gateway and validates its signed or otherwise
authenticated internal admission envelope. It binds tenant/key, request and attempt
IDs, payload hash, checkpoint revision, runtime generation, policy revision,
reservation, limits and deadline before calling this worker. Public body fields
cannot establish tenant identity. JSON schema validation alone is not authentication.
The API owner defines the envelope separately from the upstream Laya request body.

After validation, call:

```python
future = worker.submit(verified_tenant_id, execution_attempt_id, owned_payload,
                       cost=validated_encoded_token_cost,
                       deadline=absolute_monotonic_deadline)
```

Translate the authenticated Unix deadline once at admission using the remaining
wall-clock budget and `time.monotonic()`. Reject already expired requests. The
worker independently rejects expired monotonic deadlines. Estimate or compute cost
using trusted tokenization; never accept caller-claimed cost. Bound request bytes,
question rows and tokenization work before this queue. Defaults allow 128 global
and 16 per-tenant queued requests, with cumulative queued costs of 65,536 global
and 8,192 per tenant. These are configurable bootstrap limits, not benchmark results.

Budgets count queued requests only; one executing request additionally owns input
and device buffers. The payload must be immutable or exclusively transferred to
the worker until completion. No payload logging or response cache is implemented.
Backend implementations must independently avoid shared mutable per-request state
and cross-tenant caches. Backend outputs are opaque to the worker.

Tenant queues are FIFO and rotate one request per tenant per turn. This guarantees
request-count fairness among waiting tenants, not equal device time: workloads of
different costs still have different execution durations. Production scheduling
may need cost-weighted fairness once measured latency and policy are available.

## Cancellation and deadlines

`cancel(tenant_id, execution_attempt_id)` and `future.cancel()` cancel delivery.
Queued cancellation promptly releases queue budgets. The same request ID is allowed
for different tenants. Duplicate IDs within a tenant are rejected while queued or
executing; these are not durable idempotency or billing records.

An in-flight cancellation or deadline completes delivery immediately, but the
backend call continues. Its payload, allocation and request identity remain owned
until it returns. No next backend call starts concurrently. Results arriving after
cancellation or expiry are discarded. Backend exceptions are returned through the
Future and are never logged by the worker; the external adapter must sanitize error
responses. Callbacks attached to futures must be short and nonblocking.

The two worker threads are daemon threads to avoid blocking interpreter exit. This
is not evidence of successful drain or device release. A process exit, forced kill,
or stuck backend requires supervisor reconciliation and may require operator-led
recovery before the card can safely be reused.

## Drain and shutdown

`shutdown(wait=True, cancel_queued=False, timeout=None)` closes admission and drains
existing work. `cancel_queued=True` rejects outstanding queued delivery while
allowing in-flight ownership to finish. It returns true only when both worker
threads have stopped; a timeout returns false and never authorizes releasing or
reassigning the card. Call shutdown again after the backend returns to verify drain.
The caller owns backend/device teardown and readiness publication.

Production readiness requires checkpoint verification, actual model loading and a
decision self-test. Liveness alone must not advertise readiness. Deployment remains
the platform's responsibility: shared gateway/reverse tunnel to host Management,
local Compute runtime lifecycle, and the fleet's serving-pod target. This module
does not create a parallel deployment controller.
