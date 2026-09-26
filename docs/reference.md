# CPU reference evidence

The baseline executes audited upstream Laya commit `970dc8c5f63d7b886a68409493f37d569424f933` on CPU with float32 model weights, no autocast, one Torch thread and deterministic algorithms. It is an upstream semantic reference, not evidence of Tenstorrent execution. Pin package versions through the repository dependency lock; their installed versions are recorded with each capture.

Download the checkpoint at a full Hugging Face commit SHA into `.cache/checkpoints/laya-english` and check out upstream into `.cache/upstream/laya`. The downloader must verify that repository/revision identity; the local pin script hashes bytes but cannot authenticate a user-supplied revision assertion.

```powershell
python scripts/pin_checkpoint.py --repo-id convaiinnovations/laya --revision FULL_40_CHARACTER_SHA
python scripts/build_reference.py --output artifacts/reference/cpu
python -m pytest tests/test_reference.py
```

The scripts require the repository's reference dependencies: torch, transformers, numpy, safetensors and huggingface_hub, plus upstream Laya's declared dependencies. Tests require pytest and do not load weights.

The lock records source commit/path, checkpoint repository/revision/path, every artifact SHA256 and explicit CPU float32 policy. Relative paths resolve against repository root. Missing/extra/changed files, dirty tracked upstream source, wrong source HEAD, nonimmutable revision and dtype policy mismatches fail closed. Hugging Face local download metadata is excluded. A lock is not a cryptographic attestation of download origin; retain the downloader receipt.

Upstream may normalize tokenizer compatibility files during loading. Capture rejects any such change. Normalize deliberately before pinning and document source/artifact hashes; never silently repin a mutation while treating it as the original upstream artifact.

Fixtures cover all typed answers and action probability, standalone single-option behavior, mixed question option widths and padding masks, differently sized states in a batch, long chronological conversation truncation, and empty questions. Outputs are authentic model results, never hand-authored golden values.

Each forward saves five inputs and both raw logit tensors as deterministic little-endian NumPy .npy files (pickle disabled). Tensor metadata records shape, NumPy dtype, Torch dtype and file hash. CPU JSON answers, fixture hash, lock hash, package versions and numerical policy are stored alongside. Tensor paths are relative to their case directory; answer paths are relative to the capture root. Floating-point outputs may differ across dependency/CPU versions despite deterministic execution settings; compare only with explicit provenance.

The output directory must be empty. A failure can leave incomplete evidence; only a capture with final `reference.json` is complete. Preserve failed evidence separately and use a fresh directory when retrying. Performance numbers from this CPU capture are not accelerator acceptance.

Hardware parity should inspect logits, calibrated probabilities, score expectation, boolean probabilities, action probability and threshold flips, including padding/option-count boundaries. High encoder correlation alone is insufficient. Set tolerances using measured baseline variability before promotion.

## Current reference evidence

The checked fixture under `tests/fixtures/cpu-reference` passed exact loaded-state
verification for 206 tensors (421,293,830 elements). A second fresh process passed
the same gate and matched all 35 captured input/output tensors and every answer
JSON byte exactly in the reference experiment.
Four nonpersistent rotary buffers are not checkpoint-backed; identical repeated
outputs do not constitute an independent derivation check for those buffers.

Earlier unverified captures are quarantined under ignored local artifacts and
must not be used as golden outputs. See `integrity.md` for the observed failure.
The checkpoint's temperature for 11 or more options is clamped by upstream;
affected confidence remains uncalibrated, even when implementation parity passes.


## Independent CPU host check

The manual `Independent CPU reference check` GitHub workflow downloads the pinned
public checkpoint and source on a fresh hosted CPU runner, then calls
`scripts/verify_reference_host.py --output artifacts/host-reference/capture` once.
It retains dependency versions, the new capture and `host-reference-check.json`
even when the check fails. There is no automatic retry or baseline replacement.

Every input tensor must match the committed capture exactly. Both logits outputs
must satisfy the predeclared CPU experiment tolerance `atol=rtol=1e-4`, and decoded
answer values and question order must match exactly. These are separate reported
conditions; tolerant tensor agreement cannot hide a changed decision. These
thresholds are not approved accelerator promotion tolerances. The full model must
first pass exact checkpoint-loaded-state verification. An integrity exception is
recorded with its structured mismatch details and fails the job before inference.

This lane tests a fresh full model load on an independent CPU host. Ordinary CPU
recipe CI only validates existing fixture files and model helpers.
Neither lane executes a Tenstorrent device or establishes inference performance.
