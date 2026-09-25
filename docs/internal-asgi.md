# Internal ASGI transport

`DecisionASGI(service)` wraps an already configured `DecisionService`. It creates no
listener, credentials, model fallback or permissive verifier. Platform startup owns
the ASGI server, private binding, transport security, loaded backend identity, model
self-test and authenticated admission callbacks. This is the runtime-side seam,
not an externally exposed customer gateway.

| Route | Method | Result |
| --- | --- | --- |
| `/v1/decisions` | POST | Validated decision response or sanitized machine-readable error |
| `/healthz` | GET | Process liveness, HTTP 200, including current readiness state |
| `/readyz` | GET | HTTP 200 only when service is ready; otherwise 503 |

Decision requests require `Content-Type: application/json` (optional UTF-8 charset)
and `X-Thatch-Admission-Grant`. That internal header carries canonical standard
base64 encoding of opaque grant bytes, at most 16,384 encoded bytes (12,288 decoded).
Duplicate selected headers, malformed base64 and absent/empty grants are rejected.
The existing platform verifier still authenticates the decoded bytes; base64 is
not authentication. The platform must strip customer-supplied internal headers
before minting/forwarding its grant. No caller-provided tenant field is trusted.

The exact request bytes, including whitespace, reach `DecisionService` so the
authenticated SHA-256 binding remains intact. Both declared and fragmented bodies
are capped at 2 MiB. Content-Length must match if supplied. Body collection has a
configurable 30-second default timeout. ASGI servers decode HTTP transfer framing;
the application bounds the resulting body chunks independently of Content-Length.

`max_pending=16` bounds concurrent body readers, submission threads and responses
waiting on execution. A full pool returns 429 before admitting the request. This
is separate from the service preparation bound and worker tenant/cost queues.
Submission runs through `asyncio.to_thread`; waiting for the returned concurrent
Future is asynchronous and does not block the event loop.

Disconnect cancels response delivery. If preparation is still running, the handler
retains its bounded slot until submission returns, then cancels the returned Future.
Cancellation cannot preempt Python preparation or an active model call. The worker
retains input/device ownership until the backend returns. Server cancellation of
the request handler also arranges cancellation of the eventual submitted Future.
Graceful server shutdown must drain the service **before** stopping the event loop;
abrupt loop/process termination is not successful model/device teardown.

All error bodies have the form `{"error":{"code":"..."}}`. No raw verifier,
backend exception, prompt, credential or stack trace is returned. Major statuses:
400 malformed transport/public JSON/schema or invalid input semantics, 403 rejected
trusted admission/grant, 408 body timeout, 409 duplicate request, 413 oversized body,
schema size limit or prepared token/row overflow, 415 unsupported media type,
429 capacity exhausted, 503 unavailable/cancelled, 504 execution deadline,
500 internal preparation/execution failure. Native input that would be truncated
is rejected with 413. Typed public-request and preparation errors remain subclasses
of `AdmissionRejected` for existing callers, but transport checks these categories
before the generic grant rejection. Responses
carry `Cache-Control: no-store`. Readiness probes require no grant inside this
private seam; deployment must not publish them or the decision route by default.

The existing reservation and durable usage gaps remain unchanged. Transport
delivery, disconnect and cancellation do not establish exactly-once billing or
authorize releasing reservations/device claims. Administration/Management must
reconcile those outcomes using the established platform identity infrastructure.


An optional durable [execution journal](execution-journal.md) now records local
admission, execution and receipt outcomes, including cancelled delivery and process
crashes. It does not implement the external reservation authority or authenticated
receipt receiver. Without an explicitly configured receipt sink, this service
retains the in-memory behavior described above.

Runtime collectors can use `GET /internal/decision-runtime`; see
[runtime readback](runtime-readback.md) for observed identity, precision, limits
and lifecycle semantics. A successful readback does not establish readiness;
collectors add their observation time and validate the authorized generation.
