# Local execution journal and receipt outbox

`ExecutionJournal(path)` supplies an optional `DecisionService(receipt_sink=journal)`
boundary. It stores local execution evidence and replay tombstones on explicit
durable host storage, using SQLite WAL, FULL synchronous commits and transactions.
It does **not** replace the mandatory authenticated platform verifier or authority
reservation-consume callback. With no receipt sink, the earlier in-memory behavior
remains; deployments must opt into and provision durable storage explicitly.

## Execution and crash semantics

After authority admission returns, the service journals the exact verified context,
runtime identity and prepared row/token counts before enqueuing. Tenant-scoped
reservation, execution-attempt and usage-event IDs are each unique in this database.
Restart/replay cannot cause the same local attempt to be dispatched twice. Same
identifier strings under different verified tenants remain separate. All runtimes
that require shared host-local protection must use the same durable database.
An expected local replay returns sanitized HTTP 409 without withdrawing readiness;
storage, corruption and receipt-commit failures still fail readiness. A replay may
reach this local guard after the external authority callback has run; the platform
must reconcile that authority outcome rather than silently authorizing new work.

Immediately before invoking the model, an atomic `admitted -> running` transition
must succeed. Cancellation competes through the same transactional state machine:
`admitted -> not_started` prevents a subsequent start; once `running`, cancellation
only changes delivery state and cannot invent a nonexecution receipt. In-flight
execution still writes terminal observed evidence even if its Future was cancelled.

Completion and its outbox receipt commit in one transaction **before** a successful
result is delivered. Receipt-storage failure withdraws readiness and withholds the
result. A transaction failure after execution leaves `running` unresolved, rather
than manufacturing completed or failed usage. Failed model calls produce a distinct
`failed` receipt with observed host elapsed time; they do not claim a successfully
processed token count. Queue rejection and pre-start cancellation produce
`not_started` evidence without a usage claim.

A process crash can leave `admitted` or `running` records. `unresolved()` exposes
these for reconciliation; neither is a completed receipt or a billable usage fact.
The service calls `check_startup()` before its model self-test. Any unresolved row
blocks startup and leaves readiness failed, without changing journal records.
The supervisor must reconcile and fence the old owner before creating a replacement
service. This guard is not a cross-process lease; an empty journal cannot prove
exclusive device ownership. Unacknowledged terminal receipts do not block startup.
`running` means the durable start fence passed: a crash may occur before the actual
model call, during execution, or after execution but before completion commit.
No automatic replay, quota release or conversion to completed usage occurs. This
is conservative at-most-once local dispatch, not exactly-once execution. Platform
recovery must fence the previous runtime/device owner before interpreting records.

## Durable observations and acknowledgment

Completed receipt observations contain validated question rows, actual encoded
tokens, zero generated tokens and measured host queue/execution milliseconds.
Queue time ends at entry to the service execution wrapper; the durable start commit
then happens before timing the backend call. Journal I/O is not represented as model
execution time. No accelerator duration, cache-hit measurement, money or billing
rate is fabricated. Backend failures have elapsed-time evidence only.

Receipts preserve the platform-minted `usage_event_id`, tenant/key identity,
reservation/attempt identity, request hash and runtime identity. Observed producer
epoch timestamps are stored once, so replay never mints a new usage event or changes
its event time. `result_ready` records local Future completion, not client receipt.
Delivery state is distinct from execution state and is retained locally.

`pending_receipts(limit)` returns immutable JSON bytes represented as a string,
receipt ID and SHA-256. `acknowledge(receipt_id, payload_sha256)` accepts only that
exact persisted receipt payload, including identity, and is idempotent. The caller
must authenticate the platform's acknowledgment; knowing a correlation ID alone
is insufficient. Receipt hash mismatch blocks reading and acknowledgment. This is
corruption detection, not protection against an administrator rewriting the local
database. No prompts, questions, answers or opaque credential/grant bytes are stored.
The retained context is a snapshot of **verified claims**, not a signature artifact.

## Existing platform integration remains open

### Frozen receipt wire contract

`load_schema("execution-receipt")` defines the closed version-1 journal payload.
The journal validates it in the same transaction before writing terminal state
and outbox evidence. Invalid evidence leaves the attempt unresolved. Completed
work carries observed row/token counts and host timings; failed work carries
timings only; `not_started` requires a null start timestamp and a bounded reason.
Unknown fields, generated-token claims, and inferred accelerator time are rejected.
Numeric wire values fit unsigned 64-bit integers. Wall-clock timestamps need not
be monotonic across a clock correction; elapsed durations remain separately measured.

`schemas/fixtures/execution-receipts/` contains synthetic CPU-reference examples
produced by the real journal with a fixed test clock. `test_execution_receipt.py`
reproduces their exact payload bytes, checks restart persistence, and tests invalid
evidence rollback. They contain no model result or physical-execution claim.
The fixture files end with a newline; the journal payload string does not. Hash
the actual transmitted payload bytes, never a parsed/reformatted representation.

A receiver must authenticate the host, match all identities and runtime fields
against its own durable reservation, and validate measured work against the granted
ceilings. A schema-valid payload or matching SHA-256 is not producer authentication.
Retain original bytes, usage identity and occurrence time through durable replay;
acknowledge only after the authoritative receiver commits those exact bytes.

The Management `UsageReceipt` vocabulary is workload/host/attempt-oriented, while
Administration usage requires verified tenant/key/model identity and deduplication
on stable `(key_id, event_uid)`. An authoritative adapter and service-class/dimension
mapping are still needed; this outbox deliberately invents neither. Preserve
`usage_event_id` and original producer timestamps when mapping; do not generate a
fresh metering event on each retry.

Management's in-memory enqueue or a successful generic transport response is **not**
a receipt-specific durable acknowledgment. Neither may clear this outbox. Until an
authenticated payload-bound durable receiver exists, receipts stay pending.

The crash window between remote authority consumption and the service's local
`admitted` commit remains an authority reconciliation responsibility. Local SQLite
cannot atomically commit with a remote authority. Database deletion/replacement,
host failover and multiple independent databases are outside local replay guarantees.
Provision persistent local storage, backup/recovery and authenticated uploader
ownership through existing platform operations; no uploader, credentials, network
listener, automatic pruning or quota authority is introduced here.
