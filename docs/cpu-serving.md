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
Orchestrated launches must also pass `--assignment-sha256` with the trusted hash
of the exact file bytes. The launcher verifies the hash on the same bounded read
it parses, before loading the model. This detects replacement between assignment
selection and mounting; an immutable ConfigMap name alone is insufficient.
The launcher must provision the journal directory and real ledger peer identity.
Retain journals across failure and reconcile uncertain consumption before restart.

```sh
laya-cpu-serve --assignment /run/laya/assignment.json --root /opt/laya --port 8091
```

The process defaults to `127.0.0.1`, uses one worker, requires ASGI lifespan,
disables reload, proxy-header trust, WebSockets and access logs, and bounds HTTP
concurrency/backlog. For a separate serving pod, the trusted launcher may explicitly
select `--listen-host 0.0.0.0` and must enforce the host's ingress network policy.
No proxy-header trust or grant/admission checks change with the listen address.
Local reachability
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
It additionally runs the same tests as UID/GID 1000 with the explicit pod listener.
Both profiles must pass before publication. The tests assert the effective UID;
the local bridge and worker share that UID and private socket directory. A group
permission workaround does not satisfy the existing ledger peer contract.
It retains the image ID, inspection metadata, dependency versions and test logs.
By default it does not publish an image. A manual dispatch from `main` with
`publish=true` pushes the already tested image to
`ghcr.io/thatch-cloud/laya-cpu-reference` using the repository's scoped Actions
token. Tags include source commit, run ID and attempt; there is no moving `latest`
tag. It resolves a registry digest, checks the registry manifest's config digest
against the tested image ID, pulls by digest, and verifies source/user identity.
The retained `release.json` and `pull-reference.txt` identify the verified artifact.
Consumers must pin the registry pull reference, not the local image config ID.
Package visibility and fleet pull credentials are separate provisioning concerns.

A passed job proves CPU container acceptance only; publication does not deploy a
service. Production journals require a persistent writable volume; never
use the test's temporary journal layout for serving tenant work.

For the default loopback listener, a host service must share its network namespace;
exposing a container port alone does not make a loopback listener reachable. The
explicit pod listener enables pod-IP routing, but its container test does not
prove Kubernetes network policy, volume ownership, host affinity or orchestration.

TT image publication, remote control plane, automatic assignment,
Compute lifecycle binding and physical accelerator acceptance remain unfinished.
