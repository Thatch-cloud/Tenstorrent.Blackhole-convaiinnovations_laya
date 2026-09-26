# Full graph compiler spike

This experiment separates four outcomes:

- `preflight`: standard-library-only artifact and host readiness inspection. It never loads a model or initializes a device. Missing Linux device nodes, compiler integration or a verified device lease yields BLOCKED and exit 2.
- `torch-export`: real strict `torch.export.export` of each captured full tensor graph on CPU FP32, followed by execution and numeric comparison. Success is not TT compilation.
- `tt-compile-only`: compile against an audited saved P150 descriptor with no accelerator nodes exposed. Writes compiler artifacts only; dummy outputs are never read or compared.
- `tt-xla`: actual compilation with the Tenstorrent backend, device execution and comparison. No automatic CPU retry is permitted. Even success keeps `physical_acceptance: false`: this spike does not certify probability semantics, isolation or performance.

All modes require the trusted SHA256 of `reference.json`. Before loading arrays, the harness checks its digest, exact lock bytes and content, fixture hash/content/order, every answer hash and all seven tensor hashes. NumPy loading disables pickle and checks shape/dtype metadata. References must record successful loaded-state integrity verification. Model modes call `validate_manifest` and independently run `verify_loaded_state` before export or device transfer.

```powershell
$reference = 'artifacts/reference/cpu-integrity-checked/reference.json'
$digest = (Get-FileHash $reference -Algorithm SHA256).Hash.ToLowerInvariant()
python scripts/compiler_spike.py --mode preflight --reference $reference --reference-sha256 $digest --output artifacts/compiler-preflight
python scripts/compiler_spike.py --mode torch-export --reference $reference --reference-sha256 $digest --output artifacts/torch-export
```

Use a fresh output directory for every run. Preserve the reference digest externally when transferring evidence; computing a digest from an already tampered index does not authenticate it.

The CPU mode writes the actual exported operator inventory, graph text and quantitative comparisons (max absolute/relative error, RMSE, PCC, tolerance and pass status). Valid option logits are compared separately so padded -10000 values cannot dominate correlation. The default absolute and relative tolerances are 1e-4 and are experiment parameters, not approved hardware promotion tolerances. Optional `--save-export` writes .pt2 graphs including weights and may use substantial disk space. Empty-question cases have no forward graph and are skipped.

## Hardware authorization and execution

Obtain an externally verified exclusive scheduler/device lease before running hardware mode. Supply a trusted out-of-band hash with `--lease-sha256`. The JSON evidence schema is:

```json
{
  "schema_version": 1,
  "verified": true,
  "exclusive": true,
  "lease_id": "scheduler-issued-identity",
  "host": "exact-worker-hostname",
  "device_id": "0000:03:00.0",
  "device_serial": "verified-physical-board-serial",
  "expires_at": "2026-10-01T00:00:00Z",
  "visibility_proof": {"file": "visibility-proof.json", "sha256": "TRUSTED_SHA256_OF_PROOF"}
}
```

These are trusted external evidence fields, not a mechanism for self-authorizing hardware access. The scheduler must establish the serial-to-device mapping and retain lease ownership for the entire process; this harness does not contact the scheduler or cryptographically verify its issuer. No lease creation, device reset, firmware change or process termination occurs here.

Set `TT_VISIBLE_DEVICES` to exactly the externally verified PCI BDF in the lease. The harness checks host, expiry and visibility before initialization and each execution, then requires exactly one live TT runtime device. Run:

```bash
TT_VISIBLE_DEVICES=0000:03:00.0 python scripts/compiler_spike.py --mode tt-xla \
  --reference-sha256 "$REFERENCE_SHA256" \
  --lease /trusted/lease.json --lease-sha256 "$LEASE_SHA256" \
  --output artifacts/tt-xla-spike
```

The implementation follows the [official TT-XLA MNIST example](https://github.com/tenstorrent/tt-xla/blob/main/examples/pytorch/mnist.py): select TT using `torch_xla.runtime.set_device_type("TT")`, compile the model with backend `tt`, obtain `xm.xla_device()`, transfer model and inputs, execute and synchronize outputs to CPU. It additionally requests a full graph and rejects detected `aten::` CPU fallback counters. Use the matched Tenstorrent Torch/XLA package set described in the [getting-started guide](https://docs.tenstorrent.com/tt-xla/getting_started.html); the CPU reference environment is not assumed compiler-compatible.

The baseline stays float32. Blindly converting the complete model to BF16 is unsafe: upstream explicitly converts marker logits and CLS pooling to float32 before the action path, so its activation/linear-weight dtype contract must be handled deliberately. No BF16 or lower-precision claim is made.

Exit codes: 0 means this requested experiment passed (or preflight readiness), 1 means validation/export/compile/numeric failure, 2 means required external runtime/lease conditions are unavailable. Failure reports retain the exception type and message; graph files and per-call results already completed remain available. Compiler success alone cannot certify semantic correctness or absence of every possible backend implementation fallback.


## Pinned plugin and visibility evidence

Preflight inspects metadata without importing accelerator packages. It requires `torch_xla`, `torch_plugin_tt`, `pjrt_plugin_tt`, and `tt_torch`, plus the `pjrt-plugin-tt` distribution's `torch_xla.plugins` entry point `tt = torch_plugin_tt:TTPlugin`. The distribution Summary must identify audited TT-XLA commit `3bf6e4201f008fdf5a0ce5d244bc83ad155ee328`; missing metadata or a different build is blocked pending review. Hardware mode explicitly imports the registration modules and verifies that backend `tt` is registered.

The lease references an externally verified, hash-pinned proof file in its own directory:

```json
{
  "verified": true,
  "initialization_isolated": true,
  "other_devices_untouched": true,
  "host": "exact-worker-hostname",
  "device_serial": "verified-physical-board-serial",
  "device_bdf": "0000:03:00.0",
  "tt_xla_commit": "3bf6e4201f008fdf5a0ce5d244bc83ad155ee328",
  "plugin_version": "EXACT_INSTALLED_DISTRIBUTION_VERSION"
}
```

The proof must cover the same physical board and installed compiler build. These fields refer to existing trusted validation; filling them in does not establish isolation. A scheduler reservation plus an environment variable alone is insufficient.

The [pinned container documentation](https://github.com/tenstorrent/tt-xla/blob/3bf6e4201f008fdf5a0ce5d244bc83ad155ee328/docs/source/getting_started.md) requires exposing every TT device node and warns that passing one node can fail fatally. Do not launch an all-device container on a shared host with other active cards based on this script. First obtain exclusive board allocation and version-specific evidence that runtime initialization leaves other devices untouched; otherwise hardware execution remains blocked. This repository supplies no broad device-passthrough launch command.

The [pinned plugin](https://github.com/tenstorrent/tt-xla/blob/3bf6e4201f008fdf5a0ce5d244bc83ad155ee328/python_package/torch_plugin_tt/__init__.py) imports `tt_torch` and normalizes integer device selectors to BDFs through a workaround. Requiring a verified BDF avoids treating mutable ordinal selection as persistent identity. [Packaging](https://github.com/tenstorrent/tt-xla/blob/3bf6e4201f008fdf5a0ce5d244bc83ad155ee328/python_package/setup.py) defines the plugin entry point and build provenance.


## Offline P150 compilation

The exact audited TT-XLA revisions support `TT_COMPILE_ONLY_SYSTEM_DESC`. The plugin loads a saved system descriptor instead of querying the host and skips opening a physical mesh. Its execution branch produces dummy buffers; copying them to CPU would not yield meaningful model results. These are not inference results.

Supported offline toolchains are explicit:

| TT-XLA commit | Package version policy |
| --- | --- |
| `3bf6e4201f008fdf5a0ce5d244bc83ad155ee328` | Build metadata must identify this audited source |
| `873c53c4ff84bd91112a1f43e788cfed75aaa914` | Exactly `pjrt-plugin-tt==1.5.0.dev20260831000501`; offline only |

The nightly's declared dependency revisions are TT-MLIR `e2b21a721955b2b68540d9e6879a215ce3dcfae5` and TT-Metal `d5f9bc4a43b1287523458f1af6a470c6b2adc180`. It is not silently treated as the newer physical-runtime audit.

Both revisions' checked-in `tests/cpu_compile_only/system_descs/p150_system_desc.ttsys` have SHA256:

```text
655fabe5445624b0c9a6a312b90fb5e9a30a78614e43d331e9d622561e09b5ac
```

Download this file from the chosen immutable source revision before running. The harness checks both the supplied digest and the audited digest. Run on Linux with no `/dev/tenstorrent/*` nodes exposed and `TT_VISIBLE_DEVICES` unset. No hardware lease is needed because no hardware access is permitted.

```bash
python scripts/compiler_spike.py --mode tt-compile-only \
  --toolchain-commit 873c53c4ff84bd91112a1f43e788cfed75aaa914 \
  --reference artifacts/reference/cpu-integrity-checked/reference.json \
  --reference-sha256 "$REFERENCE_SHA256" \
  --system-desc /trusted/p150_system_desc.ttsys \
  --system-desc-sha256 655fabe5445624b0c9a6a312b90fb5e9a30a78614e43d331e9d622561e09b5ac \
  --case-id mixed-question-widths --call-index 0 \
  --output artifacts/compile-only-mixed
```

Use a fresh process/output directory for each case/shape. The upstream persistent cache is a singleton, and the artifact parser requires a single compiled entry. The harness triggers lazy compilation with `torch_xla.sync(wait=True)`, retains the dummy outputs only until synchronization, and never copies or reads their values. It requires nonempty `.ttnn`, `_ttnn.mlir`, and `_ttir.mlir` artifacts and records hashes/sizes.

Successful status is `COMPILED`, with `compile_only: true`, `numerical_comparison_performed: false` in the compilation record, and `physical_acceptance: false`. It does not set execution flags. Failure remains a compiler feasibility result, not evidence of model quality. Hardware mode rejects `TT_COMPILE_ONLY_SYSTEM_DESC` even when empty, preventing dummy outputs from being mislabeled as physical inference.

Primary-source audit: [nightly compile-only test](https://github.com/tenstorrent/tt-xla/blob/873c53c4ff84bd91112a1f43e788cfed75aaa914/tests/cpu_compile_only/test_cpu_compile_only.py), [client descriptor/mesh branches](https://github.com/tenstorrent/tt-xla/blob/873c53c4ff84bd91112a1f43e788cfed75aaa914/pjrt_implementation/src/api/client_instance.cc), [no-execution branch](https://github.com/tenstorrent/tt-xla/blob/873c53c4ff84bd91112a1f43e788cfed75aaa914/pjrt_implementation/src/api/loaded_executable_instance.cc), [dummy host-buffer behavior](https://github.com/tenstorrent/tt-xla/blob/873c53c4ff84bd91112a1f43e788cfed75aaa914/pjrt_implementation/src/api/buffer_instance.cc).

## Bounded UMD visibility audit

The audited physical stack resolves TT-XLA `3bf6e4201f008fdf5a0ce5d244bc83ad155ee328` to TT-MLIR `c282803f8f628881e805b2bbf6dc5bf53a0d230d`, TT-Metal `d04395ed862b4c65eb6877000c40200f456cb74e`, and UMD `8f3ffb71150b4ebf9691b01fd24b5b6f8fb8d404` (gitlink confirmed separately). Metal's `tt_metal/llrt/tt_cluster.cpp::open_driver` constructs a Silicon UMD cluster without explicitly passing target devices. UMD's Silicon cluster constructor calls `TopologyDiscovery::discover`; its PCIe discovery enumerates devices, constructs TTDevice objects, then calls `init_device` for discovered nodes.

The completed [device visibility audit](device-visibility-audit.md) locates filtering before TTDevice creation and initialization in this newer UMD revision. However, constructing the BDF map first opens all numeric device nodes for metadata queries. This source result does not establish physical isolation or support for a one-node container on the installed runtime. A verified device lease and installed-stack isolation evidence remain absent. Offline compilation avoids physical discovery by loading the descriptor and skipping physical mesh creation, with no device nodes exposed.

Sources: [Metal cluster](https://github.com/tenstorrent/tt-metal/blob/d04395ed862b4c65eb6877000c40200f456cb74e/tt_metal/llrt/tt_cluster.cpp), [UMD cluster](https://github.com/tenstorrent/tt-umd/blob/8f3ffb71150b4ebf9691b01fd24b5b6f8fb8d404/device/cluster.cpp), [UMD topology discovery](https://github.com/tenstorrent/tt-umd/blob/8f3ffb71150b4ebf9691b01fd24b5b6f8fb8d404/device/topology/topology_discovery.cpp).

## Use the migrated offline descriptor

The nightly requires the reviewed schema migration in
`configs/compiler/migrated-p150/p150-migrated-nightly.ttsys`.
Use its SHA256 `6028485af0f8c233185b87277f36061c2460db606f14a47bcb8284f5af5f56f3`
in place of the original descriptor and digest above. The harness accepts this
specific derived descriptor for offline compilation only. See
[descriptor migration](descriptor-migration.md) for provenance and limitations.

For the repeatable offline setup and profile sweep, use the manual GitHub workflow
or `scripts/setup_offline_compiler_ci.py` followed by
`python scripts/run_offline_matrix.py --help`. The setup requires Linux/Python 3.12
and no exposed accelerator device nodes. Use a fresh virtual environment.
