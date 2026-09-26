# Native admission boundary layout experiment

`scripts/probe_profile_boundaries.py` compares normal CPU batch execution with
the experimental single-row padded layout through `CpuReferenceBackend`.
The serving backend is unchanged. Both paths use the same admitted payload,
upstream decoder, verified checkpoint and FP32 parameters, without AMP or retry.
These are paired CPU observations, not an independent accuracy benchmark.

Run with the pinned local assets and reference dependencies:

```powershell
.venv/Scripts/python.exe scripts/probe_profile_boundaries.py --output artifacts/profile-boundaries-new
```

The output directory must be new. `OBSERVED` means the experiment completed;
individual comparisons remain explicit and do not approve serving or TT use.

## Observation on 2026-09-27

Both requests passed native admission without truncation. All decoded fields
and their order matched exactly, and encoded-token counts were unchanged.
All logits passed the existing `atol=rtol=1e-4` comparison.

| Case | Original token shape | Profile layout | Encoded tokens | Maximum option-logit error | Maximum action-logit error |
| --- | --- | --- | ---: | ---: | ---: |
| One choice with 64 valid options | `[1,142]` | one `[1,256]` row, 64 markers | 142 | 0.0000054836273193359375 | 0.0029296875 |
| 64 question rows | `[64,38]` | 64 `[1,64]` rows, 64 marker slots each | 2,432 | 0.0000075101852416992188 | 0 |

The action-logit difference is within relative tolerance. Exact persistent
checkpoint-state verification passed before and after the paired executions.
The 64-row request repeats one noul question under distinct IDs; it exercises
row-count capacity rather than a diverse 64-question quality corpus. The choice
case uses short numeric labels. Longer criteria remain subject to the separate
192-token head budget, and no larger or truncated request is admitted here.
The checkpoint's existing option-temperature clamp remains in effect.

[Retained report](../evidence/shape-profiles/boundaries-20260927/report.json)
contains exact requests, shapes, comparisons, calibration warnings and source
hashes. The accompanying `tensors-and-answers.zip` preserves paired outputs.
Hardware parity, BF16 quality, full-content length coverage and accelerator
capacity/performance are still unverified.
