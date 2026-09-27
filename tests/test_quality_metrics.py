import copy
import importlib.util
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "quality_metrics", Path(__file__).resolve().parents[1] / "scripts/score_quality_capture.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def example():
    return {"ag_news:0": {
        "input": {"questions": {"decision": {"type": "choice", "criteria": {"yes": None, "no": None}}}, "expected": {"decision": "yes"}},
        "answer": {"choice": "yes", "probabilities": {"yes": 0.8, "no": 0.2}, "answer_confidence": 0.8, "action": {"act_probability": 0.6}},
    }}


def test_hand_calculated_metrics_and_identical_control():
    data = example()
    result = module.score(data, data)["datasets"]["ag_news"]
    assert result["accuracy"] == 1
    assert result["brier"] == pytest.approx(0.08)
    assert result["nll"] == pytest.approx(-math.log(0.8))
    assert result["ece_15"] == pytest.approx(0.2)
    assert result["paired"]["agreement"] == 1
    assert result["paired"]["probability_abs_max"] == 0
    intervals = result["paired"]["bootstrap"]["intervals"]
    assert intervals["accuracy_loss"] == {"lower": 0, "upper": 0}
    assert intervals["agreement"] == {"lower": 1, "upper": 1}


def test_relabelled_or_missing_baseline_cannot_compare():
    baseline = example()
    changed = copy.deepcopy(baseline)
    changed["ag_news:0"]["input"]["expected"]["decision"] = "no"
    with pytest.raises(ValueError, match="inputs or labels"):
        module.score(changed, baseline)
    with pytest.raises(ValueError, match="identities"):
        module.score({}, baseline)


def test_decision_flip_and_probability_drift_are_reported():
    baseline = example()
    candidate = copy.deepcopy(baseline)
    candidate["ag_news:0"]["answer"].update(choice="no", probabilities={"yes": 0.3, "no": 0.7}, answer_confidence=0.7)
    paired = module.score(candidate, baseline)["datasets"]["ag_news"]["paired"]
    assert paired["agreement"] == 0
    assert paired["accuracy_loss"] == 1
    assert paired["probability_abs_max"] == pytest.approx(0.5)
    assert paired["disagreement_ids"] == ["ag_news:0"]


def test_boolean_and_ordinal_metrics_use_native_semantics():
    records = {
        "boolq:0": {"input": {"questions": {"decision": {"type": "noul"}}, "expected": {"decision": True}},
                    "answer": {"noul": 0.6, "answer_confidence": 0.6, "action": {"act_probability": 0.2}}},
        "sst5:0": {"input": {"questions": {"decision": {"type": "score"}}, "expected": {"decision": 2}},
                   "answer": {"score": 2.25, "probabilities": {"0": 0.05, "1": 0.15, "2": 0.4, "3": 0.3, "4": 0.1},
                              "answer_confidence": 0.4, "action": {"act_probability": 0.9}}},
    }
    result = module.score(records, records)["datasets"]
    assert result["boolq"]["accuracy"] == 1
    assert result["boolq"]["brier"] == pytest.approx(0.32)
    assert result["sst5"]["score_mae"] == 0.25
    assert result["sst5"]["ece_15"] is None
    assert result["sst5"]["paired"]["score_abs_max"] == 0


def test_bootstrap_keeps_pairs_and_is_independent_of_capture_order():
    baseline = example()
    baseline["ag_news:1"] = copy.deepcopy(baseline["ag_news:0"])
    candidate = copy.deepcopy(baseline)
    candidate["ag_news:1"]["answer"].update(choice="no", probabilities={"yes": 0.3, "no": 0.7}, answer_confidence=0.7)
    first = module.score(candidate, baseline)
    second = module.score(dict(reversed(list(candidate.items()))), baseline)
    assert first == second
    intervals = first["datasets"]["ag_news"]["paired"]["bootstrap"]["intervals"]
    assert intervals["accuracy_loss"] == {"lower": 0, "upper": 1}
    assert intervals["agreement"] == {"lower": 0, "upper": 1}
    # Shared per-row correctness cancels exactly, despite different predictions.
    rows = [module.observation(v) for v in candidate.values()]
    identical = module.paired_intervals(rows, rows)
    assert identical["intervals"]["accuracy_loss"] == {"lower": 0, "upper": 0}
