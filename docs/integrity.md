# Loaded-state integrity gate

Checkpoint file hashes establish artifact identity but do not prove that tensors loaded into process memory match those bytes. Run `verify_loaded_state(model, checkpoint, expected_sha256=locked_digest)` after CPU model construction and dtype conversion, before reference capture, export or transfer to an accelerator.

The verifier reads the archive one tensor at a time, promotes its values independently with NumPy, and compares every parameter and persistent buffer exactly. It verifies checkpoint keys, tensor shapes, numerical values, and the raw artifact digest when supplied. Unsupported dtypes, nonfinite source weights, non-CPU model state, changed artifacts and mismatches fail closed. It does not repair model memory or apply tolerances.

Float16 checkpoints can be promoted to float32 without Torch's float conversion path. BF16 comparisons use bit conversion and round-to-nearest-even rather than Torch numerical conversion. A successful result records tensor/element counts and the artifact hash. Nonpersistent derived buffers, such as rotary frequencies, are explicitly listed as unchecked because no checkpoint value exists for comparison; validate these against configuration separately.

`LoadedStateIntegrityError.report` contains compact diagnostics: offending key, shape/dtype, mismatch count, first index and maximum error. It never dumps tensor values. Handle this error as a hard readiness failure and retain the report.

## Observed Windows baseline failure

During initial capture with Torch 2.14.0, Transformers 5.17.0, NumPy 2.4.6 and safetensors 0.8.0, fresh processes produced different outputs despite identical inputs, artifact hashes, deterministic settings, seeds and one CPU thread. Repeated forwards within a process matched exactly. Loaded state hashes differed across processes.

A diagnostic found one differing element in `encoder.layers.6.mlp.Wo.weight` at index [379,1778]. The F16 checkpoint stored -0.7265625; the loaded model contained -0.002838134765625. Another process showed a different element in the adjacent Wi tensor. Explicit loading into the same model storage did not remove one observed mismatch. Independent scalar reads and twelve standalone Torch/NumPy dtype-conversion comparisons agreed with checkpoint bytes.

These observations establish an invalid loaded baseline. They do not identify whether the cause is hardware, allocation, copying or another runtime fault. They do not justify widening tolerances, automatically reloading until success, or claiming a hardware diagnosis.

Initial Windows captures must not serve as acceptance evidence. Regenerate and independently repeat the reference in a clean environment after this gate passes. The gate detects checkpoint-to-memory disagreement; separate repeatability tests still establish stable computation and derived-buffer behavior.


## Mismatch diagnostics

Value mismatches now retain the first differing index, observed/expected values
as strings, raw observed/expected storage bytes in hexadecimal, and host byte
order. BF16 storage is reported as its original two bytes with decoded FP32
values alongside it. Non-finite values remain JSON-safe diagnostic strings and
still fail verification. These fields do not relax exact comparison or repair
loaded parameters.

The option/temperature compiler-matrix attempt also failed this check on local
WSL/Linux before compiler invocation. The old failure only retained the parameter
name; its element values cannot be reconstructed from that report. Fresh diagnostic
runs must preserve their own outcomes rather than overwrite the failed attempt.
A mismatch does not by itself identify faulty hardware, a conversion defect, or
an incorrect verifier. Independent source-byte and scalar-conversion checks are
needed to distinguish those explanations.
