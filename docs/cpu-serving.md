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

## Reference image acceptance

`docker/cpu-reference.Dockerfile` builds an amd64 CPU image with the upstream
checkout, checkpoint and golden fixtures included. The Python base is pinned by
image digest; model inputs and direct reference dependencies are pinned. Build
outputs record the source revision and installed Python/OS versions. Transitive
dependencies and apt package resolution are not yet a reproducible build lock.

The `CPU reference image acceptance` workflow builds the image and runs all three
real CPU runtime tests as UID/GID 10001, without network, capabilities, privilege
escalation or a writable root filesystem. Only `/tmp` is writable for test journals.
It retains the image ID, inspection metadata, dependency versions and test log.
It does not publish an image or deploy a service. A passed job proves CPU container
acceptance only. Production journals require a persistent writable volume; never
use the test's temporary journal layout for serving tenant work.

The entry point binds loopback within its network namespace. A host service must
share that namespace or provide an explicitly designed local transport; exposing
a container port alone does not make a loopback listener reachable.

Production image publication, remote control plane, automatic assignment,
Compute lifecycle binding and physical accelerator acceptance remain unfinished.
