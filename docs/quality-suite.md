# Standard English quality suite

Use AG News test (7,600 rows), DAIR Emotion test (2,000), SST-5 test
(2,210) and BoolQ validation (3,270). Together these provide topic/emotion
choice, ordinal sentiment score and boolean reading-comprehension tasks.
The [pinned upstream benchmark](https://github.com/NandhaKishorM/laya/blob/970dc8c5f63d7b886a68409493f37d569424f933/research/scripts/build_benchmark_nb.py)
uses these datasets and distinguishes AG News/BoolQ retention from held-out
Emotion/SST-5 evaluation. This suite uses its own fixed prompt templates;
published upstream scores are not directly comparable.

Dataset revisions and file hashes are frozen in `configs/quality-datasets.json`.
Dataset text stays in ignored `artifacts/`; do not commit or redistribute the
downloaded corpus. Source cards: [AG News](https://huggingface.co/datasets/fancyzhx/ag_news),
[Emotion](https://huggingface.co/datasets/dair-ai/emotion),
[SST-5](https://huggingface.co/datasets/SetFit/sst5),
[BoolQ](https://huggingface.co/datasets/google/boolq).

Install `pyarrow==20.0.0` in a separate evaluation environment, then run:

```sh
python scripts/prepare_quality_suite.py --download
python scripts/prepare_quality_suite.py --per-dataset 64 --output artifacts/quality/pilot.jsonl
```

The full suite has 15,080 requests. The 256-request pilot is for bring-up only.
Both have deterministic selection and content hashes. The preparer verifies
every file hash and source row count before converting labels. It does not run
a model, truncate inputs or establish numerical acceptance.

After preparing the pinned checkpoint and upstream source using the normal CPU
reference instructions, capture the pilot with:

```sh
python scripts/build_quality_reference.py --cases artifacts/quality/pilot.jsonl
```

The manual CPU-reference workflow accepts `quality_pilot=true` to run this on an
independent CPU host after checking the original six-case baseline. The capture
uses native upstream inference, retains serialized inputs, logits, action logits
and decoded answers, and refuses to replace an existing output directory. Native
512-token compatibility truncation applies; this pilot is not the native API's
oversize-rejection test or the final full-suite quality gate.

Score a native capture, optionally comparing another capture with identical
case identities, inputs and labels:

```sh
python scripts/score_quality_capture.py artifacts/quality-reference/capture --output artifacts/quality-metrics.json
python scripts/score_quality_capture.py artifacts/candidate/capture --baseline artifacts/quality-reference/capture --output artifacts/quality-drift.json
```

The scorer verifies answer-file hashes and native answer structure. It reports
per-dataset metrics and disagreements without declaring acceptance. Probability
quantiles use nearest rank; native four-decimal probabilities remain unchanged.
ECE is reported for choice/noul only, because an ordinal expected score is not
a categorical predicted answer. Paired bootstrap intervals are described below. The scorer does not enforce
the proposed limits below.

## Proposed engineering acceptance criteria

There is no universal BF16 probability-error standard. Freeze these proposed
limits before evaluating this suite on hardware; do not loosen them to match
observed errors. They measure port fidelity, not suitability for a particular
application. Keep the original six-case semantic matrix and shape/boundary
tests as additional requirements.

Compare TT with both pinned FP32 CPU and the identical mixed-precision CPU
candidate. Report conversion error and device error separately. All requests
must complete with exact row/question attribution, finite outputs, normalized
distributions and no silent fallback. Record token lengths, truncation and
shape buckets; report truncated rows separately, never drop failures from totals.

For each dataset and each CPU comparison:

- Choice/noul agreement at least 99.5%; accuracy loss at most 0.5 percentage
  points against ground truth. Report all disagreements and CPU decision margins.
- Absolute option-probability drift: mean at most 0.005, p99 at most 0.02,
  maximum at most 0.05. Report full distributions, not only the winning option.
- Increase in multiclass Brier score at most 0.01 and negative log likelihood
  at most 0.02. Define Brier as the unnormalized sum over classes and clip
  probabilities only for NLL at 1e-12. Report 15-bin equal-width ECE as a
  diagnostic, including sample counts; it is sensitive to bin boundaries.
- SST-5 score drift: mean absolute difference at most 0.02 and maximum 0.10
  on the native 0–4 scale; ground-truth MAE increase at most 0.02.

Report paired bootstrap confidence intervals with a fixed seed alongside point
estimates; full-suite results are required for qualification. These tasks do not
label Laya's auxiliary action head. Retain action-logit/probability checks and
add non-saturated action-boundary examples before claiming action qualification.
No new temperature fitting or checkpoint changes are allowed in this comparison.

The [pinned upstream limitations](https://github.com/NandhaKishorM/laya/blob/970dc8c5f63d7b886a68409493f37d569424f933/README.md#known-limitations)
already identify near-universal saturation of `action.act_probability` and poor
correctness discrimination. Treat that as a checkpoint limitation to investigate,
not evidence that an accelerator port introduced the saturation. Preserve native
outputs and raw action logits in the parity comparison. Matching values near 1.0
establishes neither useful action gating nor accuracy near a decision boundary.
Synthetic decoder boundary tests can verify arithmetic but cannot substitute for
checkpoint-level examples. If representative non-saturated examples cannot be
established, report action qualification as unproven; do not silently recalibrate
the head, substitute confidence, or relax the release gate.

The paired scorer now reports 95% percentile intervals from 2,000 paired-row
resamples (NumPy default RNG, seed 20260928), sorted by case identity so capture
order cannot change the result. It resamples candidate and baseline together
within each dataset for accuracy loss, agreement, Brier/NLL increases and ordinal
MAE increase/mean score drift. These are diagnostic intervals, not automatic
acceptance gates or simultaneous bounds across metrics. A degenerate interval
from identical observations does not establish a population error bound. The
pilot remains too small for release qualification.

The [independent CPU pilot](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-convaiinnovations_laya/actions/runs/36357873844)
completed all 256 requests. All 1,792 retained tensor hashes were independently
checked; loaded persistent state matched all 206 tensors. Input lengths were
37–462 tokens, so this sample adds no 512-token boundary coverage.
`configs/quality-pilot-cpu-baseline.json` retains the reviewed aggregate only;
raw dataset text and host captures remain untracked.

With 64 examples per dataset, observed accuracy was 90.625% AG News, 67.1875%
Emotion and 73.4375% BoolQ; SST-5 MAE was 0.7474 on the 0–4 scale. These are
small-sample results for these prompts, not reproductions of upstream headline
scores or release quality claims. The associated calibration metrics are in the
aggregate file. TT evaluation, confidence-interval gates and release
qualification are not yet completed.


## Complete CPU reference

`configs/quality-full-cpu-baseline.json` records the reviewed full-suite aggregate
for the same pinned checkpoint, dataset revisions and prompt recipe. It includes
15,080 requests; all 105,560 tensor hashes and 15,080 answer hashes were verified.
Reconstructed input records match the prepared-case digest, and rerunning the
scorer reproduced the recorded metrics exactly. Loaded persistent model state
matched all 206 checkpoint tensors; nonpersistent rotary buffers are outside that
checkpoint comparison. Raw captures are not included in this public aggregate.

| Dataset | Cases | CPU result |
| --- | ---: | ---: |
| AG News | 7,600 | Accuracy 92.3158% |
| BoolQ | 3,270 | Accuracy 83.2722% |
| Emotion | 2,000 | Accuracy 59.25% |
| SST-5 | 2,210 | Score MAE 0.83786 |

These results describe this recipe, not a byte-identical reproduction of upstream
benchmark prompts. Calibration metrics and the relevant library versions are in
the aggregate. A complete CPU reference does not establish accelerator fidelity,
action-head qualification or application suitability.

Recorded lengths range from 36 to 512 tokens. Thirteen BoolQ cases reach exactly
512 tokens; none in the pilot did. A recorded length equal to the limit alone does
not prove truncation of the original text. Keep all cases and the explicit semantic
boundary fixtures in accelerator acceptance. The aggregate also reports counts
for the 64/128/256/512 padded profiles.

To reproduce the complete reference instead of the pilot, use the pinned setup
above, then run:

```sh
python scripts/prepare_quality_suite.py --download --output artifacts/quality/full.jsonl
python scripts/build_quality_reference.py --cases artifacts/quality/full.jsonl --output artifacts/quality-full-reference
python scripts/score_quality_capture.py artifacts/quality-full-reference/capture --output artifacts/quality-full-reference/metrics.json
```

Use a new capture output directory. The expected prepared-case SHA256 is
`4cccca3d98b3a0116ea2438133e755fd6d324a9097edb9517be6603191baa025`.
