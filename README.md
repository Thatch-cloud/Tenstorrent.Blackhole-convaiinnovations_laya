# Laya on Tenstorrent Blackhole

Work in progress toward one-card Laya inference with a shared-service integration
contract. The TT backend and deployed shared endpoint are not yet accepted.

Implemented foundations:

- Immutable upstream/checkpoint pinning and tensor/JSON reference capture.
- Native typed-decision request/response and separate internal admission schemas.
- Required platform verifier and reservation hooks, exact request-byte binding,
  runtime/revision/deadline checks, and bounded JSON parsing.
- Serialized tenant-fair worker with bounded queue costs and cancellation/drain.
- Optional durable execution journal and receipt outbox with crash/replay checks.
- Internal decision service composing verified admission, observed timing, response
  validation, model self-test readiness, failure and drain ownership.
- Explicit CPU reference backend, preserving upstream decision semantics and
  rejecting silently truncated native inputs.
- Strict PyTorch graph-export experiment and guarded TT-XLA experiment harness.

## Local development

Python 3.11 or newer is required. The tested Windows reference environment has a
version lock; the accelerator toolchain needs its own compatible Linux lock.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reference-windows-py311.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pytest -q
```

Prepare the immutable source and English checkpoint:

```powershell
git clone --no-checkout https://github.com/NandhaKishorM/laya.git .cache/upstream/laya
git -C .cache/upstream/laya checkout --detach 970dc8c5f63d7b886a68409493f37d569424f933
.\.venv\Scripts\python.exe scripts/download_checkpoint.py
.\.venv\Scripts\python.exe scripts/pin_checkpoint.py --repo-id convaiinnovations/laya --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
.\.venv\Scripts\python.exe scripts/build_reference.py --output artifacts/reference/cpu
```

Do not overwrite an existing evidence directory. Source and checkpoint hashes,
loaded-weight integrity, actual dtype/backend and calibration must be validated
before promoting a reference. A matching compiled graph is not sufficient when
its starting model differs from the checkpoint.

## Integration boundaries

The runtime is an internal execution component. Platform authentication, durable
reservation/replay control and authoritative billing are injected interfaces;
there is no default authentication bypass or in-process substitute for those
services. JSON schemas alone do not authenticate a tenant. Shared-service ingress
must supply a verified envelope independently of customer JSON.

See [API contract](docs/api-contract.md), [worker lifecycle](docs/runtime-lifecycle.md),
[decision service](docs/decision-service.md), [execution journal](docs/execution-journal.md), [reference capture](docs/reference.md),
[compiler spike](docs/compiler-spike.md), and [loaded-state integrity](docs/integrity.md).
The upstream model is [Convai Innovations Laya](https://github.com/NandhaKishorM/laya).
Weights remain external and are not included here.

## Acceptance

Local unit tests, CPU model execution, TT graph execution and deployed multi-tenant
acceptance are separate gates. Hardware requires a verified exclusive board
allocation; runner availability or an environment-variable device mask alone is
not proof of isolation from another runtime. No resets or firmware changes are
part of this repository's default execution flow.

Current local evidence: two fresh-process CPU captures passed the exact loaded
checkpoint-state gate and matched all captured tensors and answers. Strict
PyTorch export passed all five model calls against that reference. These results
validate the CPU baseline and export experiment. Four of five FP32-source model calls
compiled offline for the pinned P150 descriptor; the fifth failed loaded-state
integrity before compilation. Lowered IR includes BF16. Physical TT execution and
the deployed shared-service path remain outstanding.


CPU contract CI covers Python 3.11 and 3.12, validates the committed reference
index and every captured tensor, and checks schema resources from an isolated
wheel installation. It does not download model weights or run the accelerator.
The workflow is defined in `.github/workflows/cpu-contracts.yml`; local checks
and a GitHub Actions run remain distinct evidence.
