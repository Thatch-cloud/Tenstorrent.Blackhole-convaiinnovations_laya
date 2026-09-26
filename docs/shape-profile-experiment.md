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
