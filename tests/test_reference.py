import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from laya_tt.reference import UPSTREAM_COMMIT, checkpoint_files, sha256_file, validate_answers, validate_manifest

def locked(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    for name in ("rl_agent_config.json", "model.safetensors", "encoder/config.json", "tokenizer/tokenizer.json"):
        path = checkpoint / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture only")
    return {"schema_version": 1, "upstream": {"commit": UPSTREAM_COMMIT, "path": "source"},
            "checkpoint": {"repo_id": "test/model", "revision": "a"*40, "path": "checkpoint", "files": checkpoint_files(checkpoint)},
            "runtime": {"device": "cpu", "dtype": "float32"}}

def test_digest_is_content_based(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"known")
    assert sha256_file(path) == hashlib.sha256(b"known").hexdigest()

def test_valid_lock_and_metadata_exclusion(tmp_path):
    lock = locked(tmp_path)
    cache = tmp_path / "checkpoint/.cache/download"
    cache.parent.mkdir()
    cache.write_text("download metadata")
    with patch("subprocess.check_output", side_effect=[UPSTREAM_COMMIT+"\n", ""]):
        _, checkpoint = validate_manifest(lock, tmp_path)
    assert checkpoint == tmp_path / "checkpoint"

@pytest.mark.parametrize("mutation", ["content", "added", "missing", "revision", "runtime"])
def test_rejects_artifact_or_provenance_changes(tmp_path, mutation):
    lock = locked(tmp_path)
    if mutation == "content":
        (tmp_path / "checkpoint/model.safetensors").write_bytes(b"modified")
    elif mutation == "added":
        (tmp_path / "checkpoint/new").write_bytes(b"extra")
    elif mutation == "missing":
        (tmp_path / "checkpoint/model.safetensors").unlink()
    elif mutation == "revision":
        lock["checkpoint"]["revision"] = "main"
    else:
        lock["runtime"]["dtype"] = "bfloat16"
    with patch("subprocess.check_output", side_effect=[UPSTREAM_COMMIT, ""]):
        with pytest.raises(ValueError):
            validate_manifest(lock, tmp_path)

@pytest.mark.parametrize("head,status", [("b"*40, ""), (UPSTREAM_COMMIT, " M laya/agent.py")])
def test_rejects_different_or_dirty_upstream(tmp_path, head, status):
    lock = locked(tmp_path)
    with patch("subprocess.check_output", side_effect=[head, status]):
        with pytest.raises(ValueError):
            validate_manifest(lock, tmp_path)

def answer():
    return {"answers":{"q":{"type":"choice","choice":"sole","probabilities":{"sole":1.0},
                           "answer_confidence":1.0,"action":{"act_probability":0.5}}}}

def test_single_option_semantics():
    validate_answers(answer(), {"q":{"type":"choice", "criteria":["sole"]}})

@pytest.mark.parametrize("field,value", [("answer_confidence", float("nan")), ("choice","absent"), ("probabilities",{"sole":0.2})])
def test_rejects_invalid_answer(field, value):
    result = answer()
    result["answers"]["q"][field] = value
    with pytest.raises(ValueError):
        validate_answers(result, {"q":{"type":"choice", "criteria":["sole"]}})

def test_case_inventory_covers_required_shapes():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "configs/reference-cases.json").read_text())["cases"]
    assert {q["type"] for case in cases for q in case["questions"].values()} == {"choice","score","noul"}
    assert any("states" in case for case in cases)
    assert any(len(case["questions"]) == 1 and next(iter(case["questions"].values()))["type"] == "choice" for case in cases)


def test_rejects_other_request_options_even_when_question_id_matches():
    with pytest.raises(ValueError, match="requested criteria"):
        validate_answers(answer(), {"q":{"type":"choice", "criteria":["another-tenant-option"]}})


def test_rejects_reordered_probability_options():
    result = answer()
    result["answers"]["q"].update(choice="first", probabilities={"second":0.2,"first":0.8})
    with pytest.raises(ValueError, match="order"):
        validate_answers(result, {"q":{"type":"choice", "criteria":["first","second"]}})


def test_score_legend_is_bound_to_request():
    result = {"answers":{"q":{"type":"score", "score":0.5,
        "probabilities":{"0":0.5,"1":0.5}, "legend":{"0":"Wrong","1":"High"},
        "answer_confidence":0.5, "action":{"act_probability":0.5}}}}
    with pytest.raises(ValueError, match="legend"):
        validate_answers(result, {"q":{"type":"score", "criteria":["Low","High"]}})


def test_all_recorded_answers_match_their_actual_request_criteria():
    root = Path(__file__).resolve().parents[1]/"tests/fixtures/cpu-reference"
    report=json.loads((root/"reference.json").read_text(encoding="utf-8"))
    for case in report["cases"]:
        for result in json.loads((root/case["answers"]["file"]).read_text(encoding="utf-8")):
            validate_answers(result, case["input"]["questions"])
