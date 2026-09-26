# Finite shape CPU experiment

`scripts/probe_shape_profiles.py` explores a single-question-row execution layout
for a future accelerator backend. It does not change the serving backend or
approve a precision policy. No TT runtime is imported or device opened.

Each captured batch is split in its original row order. Sequence width is rounded
up to 32, 64, 128, 256 or 512, and marker width is padded to 64. Token and marker
masks retain their original values; added positions are masked. Existing batch
padding is retained. Outputs are trimmed to the original marker width and joined
before the unchanged upstream decoder. Encoded-token usage excludes padding.
Oversized inputs fail rather than truncate. Native serving's strict admission
policy is unchanged; the historical truncation fixture tests upstream compatibility.

The model, checkpoint and fixture inputs are verified against the pinned FP32
reference. Every persistent model tensor is checked before and after inference.
The experiment bypasses upstream AMP/retry and uses deterministic CPU FP32.
`OBSERVED` means execution completed, not that comparisons passed; individual
comparison results and decoded differences remain explicit in the report.

Run from the repository with the reference dependencies and pinned assets:

```powershell
.venv/Scripts/python.exe scripts/probe_shape_profiles.py --output artifacts/shape-profile-new
.venv/Scripts/python.exe -m pytest tests/test_shape_profiles.py -q
```

Use a new output directory for each run.

## Observation on 2026-09-27

[Retained report](../evidence/shape-profiles/cpu-fp32.json) records all six fixtures:
five original forward calls became 17 single-row executions; empty questions
performed no forward. All decoded answers, their numeric fields, and their order
matched exactly. All raw logits passed the existing `atol=rtol=1e-4` comparison.
Maximum option-logit absolute error was `5.4836273193359375e-6`; maximum action-logit
absolute error was `0.001953125` (within relative tolerance). Both loaded-state
checks passed. The existing upstream temperature clamp remains in effect.

The original fixture pass exercised sequence widths 32, 64, 128 and 512.
Across these fixtures 1,187 encoded tokens used
1,696 padded token slots. This is computational overhead, not billable usage or
a measured performance result. Fourteen adapter tests passed, including preservation
of masks, order, input ownership, all bucket boundaries, and rejection beyond
the token/marker bounds.

The retained report also includes a five-width padding sweep using the pinned
24-token single-option input. The full model ran at every proposed width,
including 256, with 64 marker slots. All five comparisons passed; option-logit
error was `1.0728836059570313e-6` and action-logit error was `0.00048828125`.
This establishes a padding check at all shapes, not quality coverage for 256
actual content tokens or 64 valid options. The complete repeated fixture pass
still decoded identically, and post-sweep checkpoint integrity passed.

Remaining gates are content-length and marker-limit quality coverage, offline
compilation of these exact shapes, explicit precision acceptance, and physical
parity under verified exclusive allocation. These CPU results establish neither
BF16 quality nor accelerator capacity, latency, isolation or serving readiness.

## Offline compiler sweep

The existing `offline-option-compile.yml` workflow accepts `profile_sweep=true`.
It runs the audited compiler with no TT device nodes and a fresh process/cache
for each of the five shapes, using the pinned single-option fixture. Each graph
has a 1,200-second limit; the job has a 110-minute total limit. The default
dispatch still compiles the original option-temperature batch.

`compiler_spike.py --profile-bucket` is restricted to `tt-compile-only` and the
single-option call. It shares the CPU experiment's layout function. The matrix
verifies the requested bucket, all five tensor shapes, compiler/descriptor
identity and artifact hashes. Compilation does not read dummy outputs and
cannot establish numerical parity. Source weights remain FP32; the compiler's
lowering precision must be assessed separately.

The retained CPU report predates extraction of the unchanged layout function
into `scripts/shape_profiles.py`; its script hash identifies the tested version
at commit `e8add32d20e92a3ddf8b3e59e706b320452c4ec2`.

The first offline dispatch, [36270462408](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-convaiinnovations_laya/actions/runs/36270462408),
failed during environment setup, before any model compilation. The explicit
compiler lock had not acquired the runtime's newer `cryptography` dependency;
mandatory `pip check` rejected the environment. The lock now includes
`cryptography==48.0.1`, `cffi==2.1.1` and `pycparser==3.0`, matching the installed
CPU development versions. The compiler versions and mandatory dependency check
are unchanged. This setup failure is not a graph failure.

New profile reports also retain the layout-helper hash, padding-token identity,
and exact input tensor hashes. The compiler checks that the pinned encoder and
tokenizer agree on the padding token before using the shared adapter.

## Verified offline result

[Run 36270614489](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-convaiinnovations_laya/actions/runs/36270614489)
completed successfully at `0a6a8184ca185b217ed132ec29f9372f8ea1a1c2`.
All five profiles compiled. Each process verified all 206 persistent checkpoint
tensors before compilation and emitted TTIR, TTNN IR and a binary. Compilation
took approximately 29–39 seconds per shape; these are host compilation times,
not inference latency. Binary sizes range from 2.52 to 5.69 MB and do not measure
model residency or peak device memory.

[Verification summary](../evidence/shape-profiles/offline-36270614489/summary.json)
records the downloaded artifact/report/log hash checks. All 25 input tensor
hashes were independently reconstructed from the pinned fixture and the local
CPU profile adapter and matched the compiler reports. Compiler harness, matrix
runner and layout-helper hashes also matched the tested checkout. The complete
matrix, reports, logs and 15 generated artifacts are retained in the hashed
`profile-sweep.zip` beside the summary; the archive is approximately 1.92 MB.

Every generated TTNN IR contains BF16 tile layouts despite FP32 source weights.
No dummy output was read, no numerical TT comparison was performed and no device
was exposed. This closes offline compilation for the proposed profile shapes.
Precision quality, actual device memory fit, isolated initialization, physical
parity and performance remain open.
