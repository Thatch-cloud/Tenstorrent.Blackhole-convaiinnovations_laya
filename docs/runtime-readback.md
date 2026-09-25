# Internal runtime readback

`GET /internal/decision-runtime` supplies the runtime-authored fields of the
neutral `DecisionRuntimeObservation` contract, version 1. A trusted collector
adds `observed_at_unix_ms` after successful readback. The response intentionally
does not contain a timestamp, tenant identity, entitlement, device lease, or
physical acceptance claim. Collection must verify the expected runtime endpoint
and generation; a successful HTTP response alone does not authorize dispatch.

This route belongs on the same platform-protected internal listener as the
existing runtime seam. There is no default listener or new authentication scheme.
Hosting must restrict network access to trusted collectors and dispatchers; do
not publish this route directly as a customer API. Decision execution continues
to require the existing authenticated, payload-bound admission grant.

Bootstrap supplies `DecisionService(..., backend_readback=backend.readback)` for
the backend returned by `load_cpu_backend`. The CPU factory enables readback only
after pinned source/checkpoint and loaded-state integrity verification. Readback
rechecks CPU device, parameter precision, disabled AMP and absence of fast mode.
Other backends must implement their own observed provider; there is no inferred
TT precision, CPU fallback, or default provider. Missing or invalid provider facts
return a sanitized 503 without starting the model or changing lifecycle state.

Identity comes from the service's trusted bootstrap `RuntimeIdentity`; precision,
supported question types and encoding ceilings come from `BackendReadback`.
Admission supplies actual configured row/token ceilings and native schema bounds.
The response intersects these caps, including rows times sequence length for the
total token cap. Limits are independent ceilings, not a promise that every input
below each ceiling fits: rendered option length, shared head budget, and complete
untruncated sequence checks can reject smaller inputs. Per-tenant grant limits
and queue/preparation capacity are separate admission policy, not model capability.

Lifecycle states are starting, ready, draining, failed and stopped. GET never
marks a service ready; the explicit loaded-model startup test remains mandatory.
All states can be read with HTTP 200 when facts are available. Existing `/healthz`
and `/readyz` response bodies and status behavior are unchanged. Collectors must
require state ready, exact authorized identity/precision, sufficient limits and
fresh observation time before advertising `decisions.v1`. Runtime readback does
not wire a collector, tunnel dispatcher, durable billing receiver or accelerator.
