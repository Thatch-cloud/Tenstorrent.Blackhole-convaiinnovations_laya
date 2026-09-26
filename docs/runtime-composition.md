# Runtime composition

`laya_tt.bootstrap.RuntimeApplication` connects the runtime adapters using one
durable execution journal. The host supplies an already verified loaded backend,
observed `RuntimeIdentity`, authenticated grant verifier, policy revision, journal
path, private ledger socket and service UID. There are no default credentials,
grant verifier, model loader, device choice, socket listener or CPU fallback.

The trusted launcher must establish exclusive device ownership **before loading
an accelerator backend**. This composition layer cannot establish that ownership.
It checks backend readback and refuses composition if the journal contains earlier
execution or consumption requiring recovery. It wires journaled consumption into
admission and uses that same journal for execution receipts.

Call `application.start(loaded_model_self_test)` with a real loaded-model test.
Readback is checked before and after the callback; startup remains unready on
failure. Serve `application.asgi` only through the host's authenticated internal
runtime transport. Supplying a backend label or a callback that merely returns
True is not hardware/model validation.

The host schedules `application.deliver_pending(limit=8)` outside its async event
loop. Each call handles a bounded batch, forwards original receipt evidence and
acknowledges the journal only after an exact authenticated bridge acknowledgment.
It stops on failure, retains evidence and prevents overlapping batches on the same
instance. The host owns retry/backoff and shutdown scheduling. Old-generation
receipts require their original trusted bridge assignment; never relabel them.

`application.drain(timeout=..., cancel_queued=...)` closes admission and waits for
model ownership to end. False forbids backend destruction or reassignment. Even
True does not resolve authoritative reservations, prove a hardware lease was
released or imply that receipt delivery has finished.

The verifier, runtime launcher, scheduler, authority reconciliation and production
deployment are still platform responsibilities. Tests use explicit test grants and
a synthetic backend to verify composition ordering; they do not establish deployed
authentication, physical accelerator isolation or numerical parity.
