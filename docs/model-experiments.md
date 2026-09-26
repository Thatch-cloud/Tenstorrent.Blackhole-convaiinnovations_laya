# Model shape and precision experiments

Run probes only after the pinned source, checkpoint and CPU reference are prepared.
Use `--help` on each script for output directory and environment options. Keep each
run in a fresh ignored artifact directory.

- `probe_shape_profiles.py` compares FP32 CPU execution using single-row token
  buckets of 32, 64, 128, 256 and 512, with 64 padded marker slots. Padding and row
  splitting must preserve masks, decoded answers and actual token counts.
- `probe_profile_boundaries.py` checks the native maximum of 64 options and
  64 question rows. These boundary cases do not replace a diverse quality corpus.
- `probe_bf16_cpu.py` checks an experimental BF16 encoder/scorer with an FP32 action
  head. It verifies converted checkpoint state separately. CPU probability changes
  mean this policy is not a promoted replacement for the FP32 baseline.
- `run_offline_matrix.py --profile-buckets 32 64 128 256 512 --cases single-option`
  compiles these shapes without a device. Add `--precision mixed-bf16-fp32` only
  for the explicit precision experiment. Default source precision is float32.
- `diagnose_checkpoint_conversion.py` investigates loaded-weight conversion
  independently of inference. It does not repair a failed reference.

Native complete-input limits are 48 tokens per option, 192 head tokens and
512 total tokens per question. `cpu_backend.py` rejects input that upstream would
silently truncate. Checkpoint calibration for 11 or more options is clamped by
upstream; parity does not establish calibrated confidence in that range.

FP32-source lowering can contain BF16 intermediates. A declared source dtype,
compiled binary size, or compile duration is not evidence of physical precision,
device memory usage, inference speed or numerical parity. Physical validation
must compare valid logits, probabilities, decisions and threshold flips.
