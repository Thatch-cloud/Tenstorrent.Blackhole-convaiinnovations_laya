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
    validate_answers(answer(), {"q":{"type":"choice"}})

@pytest.mark.parametrize("field,value", [("answer_confidence", float("nan")), ("choice","absent"), ("probabilities",{"sole":0.2})])
def test_rejects_invalid_answer(field, value):
    result = answer()
    result["answers"]["q"][field] = value
    with pytest.raises(ValueError):
        validate_answers(result, {"q":{"type":"choice"}})

def test_case_inventory_covers_required_shapes():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "configs/reference-cases.json").read_text())["cases"]
    assert {q["type"] for case in cases for q in case["questions"].values()} == {"choice","score","noul"}
    assert any("states" in case for case in cases)
    assert any(len(case["questions"]) == 1 and next(iter(case["questions"].values()))["type"] == "choice" for case in cases)
