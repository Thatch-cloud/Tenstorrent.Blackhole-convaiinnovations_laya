# CPU checkpoint conversion diagnostic

`scripts/diagnose_checkpoint_conversion.py` isolates tensor reading, FP16-to-FP32 conversion and copy from model construction. It never loads Laya, imports a TT runtime, repairs model state, or changes comparison tolerances. Its outputs are diagnostic evidence only.

The script pins the original safetensors SHA256, validates the raw header and byte ranges, and requires the exact 206 tensors / 421,293,830 elements. It handles the checkpoint's 205 FP16 tensors and one FP32 tensor individually, so it does not retain another complete model in memory. For each tensor it records source bytes and comparison hashes for:

1. Raw little-endian bytes versus the safetensors NumPy reader.
2. Raw bytes versus the safetensors Torch reader.
3. Independent NumPy promotion versus `Torch.to(float32, copy=True)`.
4. Independent NumPy promotion versus `copy_` into a separately allocated FP32 tensor.
5. Source bytes versus the Torch source after both operations.

Comparisons use exact storage bits, including signed zero. A mismatch records its count, first flat/multidimensional index, source and observed bit patterns, and checkpoint byte offset. The script rereads that scalar from the original file and uses Python `struct.unpack('<e')` (or `'<f'` for the FP32 tensor), then `struct.pack('<f')`, as a third conversion oracle. This distinguishes a wrong NumPy expectation from wrong observed Torch bits without choosing either implementation as automatically correct. Checkpoint digests are recorded before and after a completed sweep.

Example in the isolated Linux environment:

```bash
python scripts/diagnose_checkpoint_conversion.py \
  --checkpoint .cache/checkpoints/laya-english/model.safetensors \
  --output artifacts/diagnostics/cpu-conversion-v1/process-1
```

Output must be new. `MATCHED` exits zero; a mismatch or invalid source exits nonzero. Partial reports are retained after each tensor. Keep each independent process's report and log, including failures. This investigation is capped at three fresh processes; it is not a retry-until-success policy.

The per-tensor experiment differs from a full model load: allocator layout, peak resident memory, model construction, retained tensors, and exact surrounding operations differ. Passing these checks cannot certify `load_state_dict`, rule out intermittent software or host behavior, establish a hardware fault, or validate a failed model capture. Full-model integrity gates remain mandatory.

## Observed bounded result

All three fresh WSL processes completed with `MATCHED`: each checked all 206 tensors and 421,293,830 elements through the five paths above, with no mismatches and unchanged checkpoint digests. Their complete reports are byte-identical, SHA256 `cc664e21ae2b581abb21f29754d3bb6b369f07b173e73235f3e6a86afcb189b0`. Sampling stopped after the third process. No compiler or full-model load was performed by this three-process diagnostic.

The captured host context reports an Intel Core i7-1195G7 and WSL2 kernel `6.6.87.2-microsoft-standard-WSL2`; the filtered CPU flags and exact library versions are retained with the evidence. This is context, not a fault diagnosis. The isolated checks did not reproduce or explain the earlier full-model integrity failure.

All three reports, logs, process outcomes and host context are indexed in [diagnostic evidence](../evidence/diagnostics/cpu-conversion-v1/index.json).
