"""Score native quality captures; these metrics do not certify hardware acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from laya_tt.reference import validate_answers


def load_capture(directory):
    directory = directory.resolve()
    report = json.loads((directory / "reference.json").read_text(encoding="utf-8"))
    records = {}
    for item in report["cases"]:
        case = item["input"]
        identity = case["quality_id"]
        if identity in records:
            raise ValueError("Duplicate quality identity")
        path = (directory / item["answers"]["file"]).resolve()
        if directory not in path.parents:
            raise ValueError("Answer path escapes capture")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != item["answers"]["sha256"]:
            raise ValueError("Answer hash mismatch")
        values = json.loads(payload)
        if len(values) != 1:
            raise ValueError("Quality case must have one native result")
        validate_answers(values[0], case["questions"])
        if set(case["questions"]) != {"decision"}:
            raise ValueError("Unexpected quality question schema")
        records[identity] = {"input": case, "answer": values[0]["answers"]["decision"]}
    if not records:
        raise ValueError("Empty quality capture")
    return records


def observation(record):
    case, answer = record["input"], record["answer"]
    kind = case["questions"]["decision"]["type"]
    expected = case["expected"]["decision"]
    if kind == "noul":
        if not isinstance(expected, bool):
            raise ValueError("Boolean ground truth required")
        probabilities = [1 - answer["noul"], answer["noul"]]
        gold, predicted = int(expected), int(answer["noul"] >= 0.5)
    else:
        probabilities = list(answer["probabilities"].values())
        if kind == "choice":
            labels = list(case["questions"]["decision"]["criteria"])
            gold, predicted = labels.index(expected), labels.index(answer["choice"])
        else:
            if type(expected) is not int or not 0 <= expected < len(probabilities):
                raise ValueError("Ordinal ground truth outside rubric")
            gold = expected
            predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {"kind": kind, "probabilities": probabilities, "gold": gold,
            "predicted": predicted, "correct": int(gold == predicted),
            "confidence": answer["answer_confidence"],
            "brier": sum((p - int(i == gold)) ** 2 for i, p in enumerate(probabilities)),
            "nll": -math.log(max(probabilities[gold], 1e-12)),
            "score": answer.get("score"),
            "score_error": abs(answer["score"] - expected) if kind == "score" else None,
            "action_probability": answer["action"]["act_probability"]}


def ece(rows):
    # Scores are ordinal expectations, not categorical predicted answers.
    if rows[0]["kind"] == "score":
        return None
    total = 0.0
    for index in range(15):
        bucket = [r for r in rows if min(int(r["confidence"] * 15), 14) == index]
        if bucket:
            total += len(bucket) / len(rows) * abs(
                statistics.mean(r["confidence"] for r in bucket)
                - statistics.mean(r["correct"] for r in bucket))
    return total


def summarize(rows):
    kind = rows[0]["kind"]
    if any(row["kind"] != kind for row in rows):
        raise ValueError("Mixed decision kinds within a dataset")
    return {"rows": len(rows), "type": kind,
            "accuracy": statistics.mean(r["correct"] for r in rows) if kind != "score" else None,
            "brier": statistics.mean(r["brier"] for r in rows),
            "nll": statistics.mean(r["nll"] for r in rows), "ece_15": ece(rows),
            "score_mae": statistics.mean(r["score_error"] for r in rows) if kind == "score" else None}


def score(records, baseline=None):
    if baseline is not None and records.keys() != baseline.keys():
        raise ValueError("Candidate and baseline case identities differ")
    grouped = {}
    for identity, record in records.items():
        if baseline is not None and record["input"] != baseline[identity]["input"]:
            raise ValueError("Candidate and baseline inputs or labels differ")
        dataset = identity.split(":", 1)[0]
        if dataset not in {"ag_news", "emotion", "sst5", "boolq"}:
            raise ValueError("Unknown quality dataset")
        grouped.setdefault(dataset, []).append(identity)
    result = {"scope": "native_decoded_quality_metrics", "physical_acceptance": False,
              "quality_qualification": False, "datasets": {}}
    for dataset, identities in sorted(grouped.items()):
        rows = [observation(records[i]) for i in identities]
        summary = summarize(rows)
        result["datasets"][dataset] = summary
        if baseline is None:
            continue
        prior = [observation(baseline[i]) for i in identities]
        prior_summary = summarize(prior)
        differences = []
        for actual, expected in zip(rows, prior):
            if len(actual["probabilities"]) != len(expected["probabilities"]):
                raise ValueError("Probability vector width mismatch")
            differences.extend(abs(a - b) for a, b in zip(actual["probabilities"], expected["probabilities"]))
        ordered = sorted(differences)
        paired = {"probability_abs_mean": statistics.mean(differences),
                  "probability_abs_p99": ordered[math.ceil(0.99 * len(ordered)) - 1],
                  "probability_abs_max": max(differences),
                  "brier_increase": summary["brier"] - prior_summary["brier"],
                  "nll_increase": summary["nll"] - prior_summary["nll"],
                  "action_probability_abs_max": max(abs(a["action_probability"] - b["action_probability"]) for a, b in zip(rows, prior))}
        if summary["type"] == "score":
            score_delta = [abs(a["score"] - b["score"]) for a, b in zip(rows, prior)]
            paired.update(score_abs_mean=statistics.mean(score_delta), score_abs_max=max(score_delta),
                          score_mae_increase=summary["score_mae"] - prior_summary["score_mae"])
        else:
            paired.update(agreement=statistics.mean(a["predicted"] == b["predicted"] for a, b in zip(rows, prior)),
                          accuracy_loss=prior_summary["accuracy"] - summary["accuracy"],
                          disagreement_ids=[i for i, a, b in zip(identities, rows, prior) if a["predicted"] != b["predicted"]])
        summary["paired"] = paired
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="Directory containing reference.json and hashed answer files")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(load_capture(args.capture), load_capture(args.baseline) if args.baseline else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
