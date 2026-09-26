# Local ledger client

`laya_tt.ledger_client.LocalLedgerClient` is an explicit Linux runtime adapter to
a trusted host-owned ledger bridge. Bootstrap supplies an absolute socket path,
the service UID and the independently observed loaded-runtime metadata. It does
not receive node credentials, verify grants, select a host or reserve capacity.

Bind `client.journaled_consumption(journal)` as the `AdmissionAdapter` consumption
callback, using the same `ExecutionJournal` instance/path as the service receipt sink.
The adapter must authenticate the grant and prepare the input first. The client
encodes those verified claims, the exact prepared counts and its owned runtime
snapshot, then makes one request. Only an authenticated local bridge's 204 permits
the callback to return `True`. Every error denies execution. A timeout or lost
response may follow a committed consumption: reconcile it, never retry it or infer
that its reservation can be released.

For each immutable `Receipt` from the execution journal, call
`client.deliver_receipt(receipt)`. Only `True` permits the caller to acknowledge the
same receipt ID and SHA-256 in its journal. Original bytes are preserved. Missing,
oversized, duplicate-field or mismatched acknowledgments keep the outbox entry
pending. Receipt redelivery must preserve the original ID, bytes and digest.
Receipt delivery does not acknowledge journal entries itself.

Each call rechecks the canonical private socket directory and socket ownership
and permissions, then checks the connected server's Linux `SO_PEERCRED` UID before
sending any metadata. Use a 0700 directory and 0600 socket owned by the configured
UID. Unsupported platforms fail closed. The launcher must protect that path for
its entire lifetime. Programs sharing the UID belong to the same trust domain;
this is not isolation against them or privileged host processes.

Requests are capped at 16 KiB by evidence validation. Response headers are capped
at 8 KiB and receipt acknowledgments at 4 KiB. The client accepts the bridge's
HTTP/1.1 length-delimited responses, with no chunking, informational responses,
redirect following or proxy support. A fresh Unix connection is used per call.
Consumption uses the remaining grant deadline capped at 15 seconds; receipt
delivery is capped at 15 seconds. Errors expose fixed text without remote content.

The explicit [runtime composition](runtime-composition.md) wires the shared journal
and bounded receipt batches. The host must still provide the grant verifier,
launcher, receipt scheduling and authoritative recovery. Recover old-generation receipts using their
original trusted runtime assignment; do not relabel them as a new generation.
Successful local or CI tests do not establish platform deployment, hardware
exclusion, accelerator parity or end-to-end physical acceptance.


The journal upgrades schema versions 1 and 2 to 3 transactionally, preserving existing
execution/receipt rows and adding consumption intents. Before sending, the callback
persists exact consumption metadata and verified claims. A lost response, exception,
or crash retains a pending intent. A confirmed ACK changes it to acknowledged;
only matching durable admission replaces it with an execution row in one transaction.
Both intent states block startup and duplicate consumption across journal connections.
Use `pending_consumptions()` as recovery evidence; never replay those requests or
infer that expiry released their reservations. No automatic intent deletion is provided. A crash before a network send may also
leave an intent, so an intent alone does not prove remote consumption.

The lower-level `consume_reservation` method is transport only. Direct use without
the journal wrapper retains the consumption-to-journal crash gap. Production bootstrap
must select the wrapper and the same durable journal; existing test adapters remain
available, so this library addition is not automatic production activation.

Before constructing `RuntimeApplication`, a trusted supervisor may call
`client.recover_pending(journal, limit=8)` with the original generation's trusted
runtime assignment. The batch limit is 1?64. Each request sends the original stored
consumption bytes to `/v1/ledger/recover`, with an independent 15-second deadline;
the original grant may have expired. A closed acknowledgment must bind SHA-256 of
those exact bytes. Duplicate fields, unknown fields, invalid counts, identities,
and mismatched digests fail closed.

Only `unconsumed_fenced` changes a pending intent to a durable fenced tombstone.
Startup then ignores that tombstone, but its attempt, reservation and usage-event
identities remain occupied. An acknowledged intent cannot become unconsumed.
`consumed` and `settled` are returned as observations and keep startup blocked;
they never authorize execution or acknowledge local receipts. Those cases still
require supervisor/process fencing and terminal evidence reconciliation.

A failed or lost recovery reply leaves the intent unresolved. Recovery can be
retried with the same bytes; consumption cannot. Earlier verified fences in a
partially failed batch remain committed. The journal uses an exact-context/body
compare-and-set, rejecting any intervening acknowledgment or admission. This API
is a trusted supervisor seam, not a public HTTP endpoint or a device lease.
