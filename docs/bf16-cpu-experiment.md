# Explicit mixed-precision CPU candidate

`scripts/probe_bf16_cpu.py` is an isolated experiment. It does not modify the default backend, accepted FP32 fixtures, upstream source, checkpoint, or compiler harness. It never attempts TT compilation or hardware execution. The candidate is not equivalent to upstream autocast and has no approved promotion threshold.

## Precision policy

Start with the pinned Laya model on CPU in FP32 and verify every persistent checkpoint tensor exactly. Convert parameters under `encoder`, `type_emb`, both decision-head transformer layers, and `scorer` to BF16. Keep `act_head` parameters in FP32. Do not convert any buffers: the temperature and all RoPE inverse-frequency buffers remain FP32 and retain their exact contents. Reject parameters outside the named policy. Verify the converted state against independent checkpoint promotion before any inference, and verify state again after all cases.

This follows the concrete boundary in audited [Laya `common.py`](https://github.com/NandhaKishorM/laya/blob/970dc8c5f63d7b886a68409493f37d569424f933/laya/common.py#L147): scorer output is explicitly converted to FP32 at line 163; probability/action features are derived from those logits; pooled hidden state is converted to FP32 at line 181; the action head consumes the concatenated FP32 values. A whole-model BF16 conversion would violate that action-head input/weight contract without a separate autocast policy.

## Runtime verification and comparison

Real forward hooks record input/output dtype, shape and device for the encoder, RoPE module, question-type embedding, each decision transformer layer, scorer and action head. Required hooks must execute for each captured forward. Scorer and decision-head outputs must be BF16; action-head inputs/outputs and public logits remain FP32. The experiment keeps native preprocessing and answer decoding but invokes the model directly instead of upstream's AMP/fallback wrapper. Autocast is disabled.

Every fixture input is compared exactly against the accepted hash-validated FP32 capture. Raw valid-marker logits, action logits, and action probabilities receive numeric error metrics. Native decoded output comparison records all probability/confidence/score/action numeric differences separately from discrete choices, field structure and key order. Actual logits and decoded answers are retained as new artifacts. The empty-question case remains a no-forward behavior check.

The existing FP32 `atol=rtol=1e-4` comparison is reported only as a diagnostic reference. It is not an approved BF16 quality gate. Successful experiment completion is `OBSERVED`, with `bf16_promotion_approved: false` and `physical_acceptance: false`; numerical drift is not silently converted into a pass. State-integrity, dtype or input-identity failure stops the experiment and preserves its report.

```bash
python scripts/probe_bf16_cpu.py --root "$PWD" \
  --output artifacts/precision/bf16-cpu-v1
```

Use a new output directory. The bounded local observation is one process with a 600-second timeout; there is no retry-until-success behavior. Further compiler work requires reviewing the observed semantics and defining the precision policy and acceptance thresholds explicitly.

## Observed CPU result

The single run completed in 235 seconds with status `OBSERVED`. All three exact state checks passed (initial FP32, converted candidate, and post-forward candidate). The parameter inventory contains 201 BF16 tensors and four FP32 action-head tensors. All buffers were bitwise preserved, and every required dtype hook passed for all five tensor graphs. The sixth fixture has no questions and no tensor forward.

All discrete answers, structure and question order matched the accepted fixtures. Numerical outputs were not equivalent:

| Quantity | Largest observed absolute change |
| --- | ---: |
| Valid marker logits | 0.06393694877624512 |
| Action logits | 18.6572265625 |
| Published calibrated option probabilities | 0.01040000000000002 |
| Published score | 0.01639999999999997 |
| Published confidence fields | 0.007800000000000029 |
| Published noul probability | 0.0023999999999999577 |
| Action probabilities | 0 |

**The action probabilities are saturated in these examples. Their zero observed change does not validate action quality or compensate for the action-logit drift.** Only the single-option and empty-question decoded results were fully exact; the other four fixtures had numeric differences. The inherited checkpoint temperature-clamping warning also remains applicable.

This run does not meet FP32 equivalence and does not promote BF16. No further model execution or compiler attempt was made. The full outputs, dtype events, state guards, numeric comparisons and process log are preserved in [the evidence index](../evidence/precision/bf16-cpu-v1/index.json); [summary.json](../evidence/precision/bf16-cpu-v1/summary.json) separates score, probability and confidence changes. Raw report SHA256: `ff6a38dc74d8d9b0503d85bb699708234a55846f9911d60a16653a40f4d40b67`.
# Offline compilation follow-up

The compiler harness now accepts `--precision mixed-bf16-fp32` only with
`--mode tt-compile-only`. It reuses this experiment's conversion policy after
verifying the original FP32 state, verifies the converted state against the
checkpoint, and records the policy source hash and parameter/buffer inventory.
The matrix verifier checks both state proofs and the expected parameter dtypes.
Default runs remain FP32-source experiments.

To attempt the five finite profiles on Linux without device nodes:

```sh
python scripts/run_offline_matrix.py --root "$PWD" --cases single-option \
  --profile-buckets 32 64 128 256 512 --precision mixed-bf16-fp32 \
  --timeout 1200 --output artifacts/offline-mixed-profiles
```

This option is also available in the independent offline compilation workflow.
Compilation does not read dummy outputs or establish numerical parity, final
lowered operator precision, physical memory fit or inference performance. The
candidate remains experimental; the existing CPU differences and calibration
limitations still apply. Actual compilation evidence is pending.
