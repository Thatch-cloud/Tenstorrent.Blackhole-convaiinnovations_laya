# Local ledger client

`laya_tt.ledger_client.LocalLedgerClient` is an explicit Linux runtime adapter to
a trusted host-owned ledger bridge. Bootstrap supplies an absolute socket path,
the service UID and the independently observed loaded-runtime metadata. It does
not receive node credentials, verify grants, select a host or reserve capacity.

Bind `client.consume_reservation` as the `AdmissionAdapter` consumption callback.
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
The client does not erase or acknowledge journal entries itself.

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

Host bootstrap, the grant verifier, journal recovery and receipt delivery loop
still need explicit composition. Recover old-generation receipts using their
original trusted runtime assignment; do not relabel them as a new generation.
Successful local or CI tests do not establish platform deployment, hardware
exclusion, accelerator parity or end-to-end physical acceptance.
