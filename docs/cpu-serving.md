# Local CPU reference serving

`laya-cpu-serve` composes the existing signed-admission runtime, durable journal,
local ledger client and hosted lifecycle into one loopback HTTP process. It is an
explicit CPU reference entry point. It rejects `tt-blackhole`; it neither claims
a device nor provides an accelerator fallback. The TT image and launcher remain
separate unfinished work.

Install the package with its `reference` and `serve` extras in the pinned reference
environment. Provision the source, checkpoint lock and golden reference tree
before starting. The entry point performs no downloads. The root must contain
`configs/checkpoint-lock.json`, `tests/fixtures/cpu-reference/` and the assets at
the paths in that lock. It checks the committed reference digest, exact lock
identity, checkpoint bytes and loaded state. Startup executes the mixed-question
golden case before readiness.

The host launcher supplies a protected JSON assignment with exactly these fields:

```json
{
  "schema_version": 1,
  "runtime": {
    "checkpoint_revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    "runtime_revision": "operator-pinned-build",
    "runtime_generation": "operator-unique-generation",
    "backend": "cpu-reference",
    "model": "laya-english"
  },
  "issuer": "configured-authority",
  "host_id": "provisioned-host",
  "policy_revision": "assigned-policy",
  "public_keys": {"configured-key-id": "REPLACE_WITH_64_LOWERCASE_HEX_CHARACTERS"},
  "journal_path": "/var/lib/laya/generation/journal.sqlite",
  "ledger_socket": "/run/laya/ledger.sock",
  "ledger_uid": 1000
}
```

The example is deliberately invalid until the real public verification key and
assignment values are supplied. No signing key is loaded. Protect the assignment
and assets from workloads; their paths are operator input, not a customer API.
The launcher must provision the journal directory and real ledger peer identity.
Retain journals across failure and reconcile uncertain consumption before restart.

```sh
laya-cpu-serve --assignment /run/laya/assignment.json --root /opt/laya --port 8091
```

The process binds only `127.0.0.1`, uses one worker, requires ASGI lifespan,
disables reload, proxy-header trust, WebSockets and access logs, and bounds HTTP
concurrency/backlog. Run it in the host service's network namespace. Local reachability
does not grant admission: requests still require signed context and acknowledged
ledger consumption. Startup/shutdown use `HostedRuntime`; signal-driven shutdown
drains execution and retains unacknowledged receipts. Shutdown does not prove
authoritative ledger reconciliation or release a hardware claim.

This is reference-service composition, not a production image, remote control
plane, automatic assignment source or physical accelerator acceptance.
