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
a categorical predicted answer. It does not yet calculate bootstrap intervals
or enforce the proposed limits below.

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

Status: preparation, capture and descriptive metric tools are implemented.
Independent CPU pilot execution is pending. TT evaluation, confidence-interval
gates and release qualification are not yet completed.
