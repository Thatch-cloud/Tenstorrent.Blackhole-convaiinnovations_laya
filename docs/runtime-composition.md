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
deployment are still platform responsibilities. Default contract tests use explicit
test grants and a synthetic backend to verify composition ordering.

The opt-in Linux test below loads the pinned CPU/FP32 checkpoint and verifies
startup and service answers against the committed mixed-question-widths reference
(choice, score, noul and a one-option choice). It uses a real Ed25519 grant,
the ASGI entry point, Linux Unix-socket peer credentials, journaled consumption
and the durable receipt outbox. The journal is reopened before and after receipt
acknowledgment to check retention and acknowledgment across connections.

```sh
LAYA_CPU_INTEGRATION=1 PYTHONPATH=src python -m unittest discover -s tests -p test_cpu_runtime.py -v
```

Install the reference dependencies and provision the exact assets named by
`configs/checkpoint-lock.json` first. The manually dispatched CPU reference
workflow runs this test after its independent reference capture passes. Ordinary
contract CI skips it because that lane does not download checkpoint assets.

The Unix ledger peer in this test is a recording fixture. It verifies the original
consumption/receipt bytes and returns exact acknowledgments, but is not a durable
reservation authority. A pass demonstrates the real CPU runtime boundary, not
shared-platform admission/accounting, production deployment or TT acceptance.
