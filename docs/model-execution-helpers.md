# Reusable model execution helpers

The installed `laya-tt` package exposes the experimental model operations used
by the recipe probes. Importing these modules does not import a TT runtime or
initialize an accelerator:

- `laya_tt.precision.apply_candidate_policy(model)` applies the existing mixed
  BF16/FP32 policy to a verified CPU FP32 model. The action head and buffers stay
  FP32. Check loaded-state integrity after conversion as the BF16 probe does.
- `laya_tt.shape_profiles.profile_rows(inputs, pad_token_id)` prepares single-row
  32/64/128/256/512-token profiles with 64 marker slots. Inputs are CPU tensors in
  `(input_ids, attention_mask, marker_pos, marker_mask, qtype)` order. It preserves
  row order and existing masks, and rejects input outside the finite bounds.
- `laya_tt.profiled_forward.profiled_forward(model, inputs, device=..., pad_token_id=...)`
  invokes an already prepared model on the explicit device. It returns owned CPU
  FP32 decision and action logits, with the original marker width and row order.
  Output buffers are copied before another row can reuse them. Wrong output shape,
  dtype, placement, or non-finite valid logits fail; backend exceptions propagate.

Use `scripts/probe_shape_profiles.py --output artifacts/profile-check` for the
pinned CPU reference comparison through this helper. Provision the pinned source
and checkpoint first as described in the README. Probe reports hash the package
modules containing the implementation, rather than only compatibility scripts.
The old script imports remain available for existing diagnostic recipes.

These are model execution primitives, not a qualified TT loader. The caller owns
weight verification, compiler setup and lowered-IR checks, device ownership,
serialization, native decoding and calibration, and release-quality evaluation.
No compiler revision is promoted by this extraction. Explicit CPU execution is
available for reference comparison; accelerator failure never triggers a CPU
retry. Checking output device placement does not detect every compiler fallback
operation. Physical parity and action-head qualification remain unproven.
