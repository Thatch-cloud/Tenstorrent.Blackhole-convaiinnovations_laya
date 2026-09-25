import hashlib
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("numpy")
safetensors = pytest.importorskip("safetensors.torch")
from laya_tt.integrity import LoadedStateIntegrityError, verify_loaded_state

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.125, -0.7265625], [2., 3.]], dtype=torch.float32))
        self.register_buffer("persistent", torch.tensor([2., 4.]))
        self.register_buffer("derived", torch.tensor([1.]), persistent=False)

def checkpoint(tmp_path, model):
    path = tmp_path / "model.safetensors"
    safetensors.save_file({name:value.half() for name,value in model.state_dict().items()}, str(path))
    return path

def test_exact_independent_promotion_and_nonpersistent_disclosure(tmp_path):
    model = Tiny()
    path = checkpoint(tmp_path, model)
    report = verify_loaded_state(model, path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert report["verified"]
    assert report["checked_tensors"] == 2
    assert report["unchecked_nonpersistent_buffers"] == ["derived"]

@pytest.mark.parametrize("key", ["weight", "persistent"])
def test_single_changed_element_fails_and_does_not_repair(tmp_path, key):
    model = Tiny()
    path = checkpoint(tmp_path, model)
    with torch.no_grad():
        getattr(model, key).flatten()[0] += 0.01
    before = getattr(model, key).detach().clone()
    with pytest.raises(LoadedStateIntegrityError) as exc:
        verify_loaded_state(model, path)
    detail = exc.value.report["mismatches"][0]
    assert detail["key"] == key
    assert detail["differing_elements"] == 1
    assert detail["max_abs_error"] > 0
    assert torch.equal(before, getattr(model, key))

def test_shape_mismatch(tmp_path):
    model = Tiny()
    path = checkpoint(tmp_path, model)
    model.weight = torch.nn.Parameter(torch.ones(4))
    with pytest.raises(LoadedStateIntegrityError) as exc:
        verify_loaded_state(model, path)
    assert exc.value.report["mismatches"][0]["reason"] == "shape_mismatch"

def test_key_mismatch(tmp_path):
    model = Tiny()
    path = checkpoint(tmp_path, model)
    model.register_buffer("unexpected", torch.ones(1))
    with pytest.raises(LoadedStateIntegrityError) as exc:
        verify_loaded_state(model, path)
    assert any(row["reason"] == "model_key_missing_from_checkpoint" for row in exc.value.report["mismatches"])

def test_source_key_missing_from_model(tmp_path):
    model = Tiny()
    path = checkpoint(tmp_path, model)
    del model.weight
    with pytest.raises(LoadedStateIntegrityError) as exc:
        verify_loaded_state(model, path)
    assert any(row["reason"] == "checkpoint_key_missing_from_model" for row in exc.value.report["mismatches"])

def test_hash_mismatch_precedes_tensor_load(tmp_path):
    model = Tiny()
    path = checkpoint(tmp_path, model)
    with pytest.raises(LoadedStateIntegrityError) as exc:
        verify_loaded_state(model, path, expected_sha256="0"*64)
    assert exc.value.report["checked_tensors"] == 0

def test_bfloat16_conversion_is_verified_by_bits(tmp_path):
    model = Tiny()
    path = checkpoint(tmp_path, model)
    model.bfloat16()
    assert verify_loaded_state(model, path)["verified"]

def test_nan_is_not_accepted(tmp_path):
    model = Tiny()
    path = checkpoint(tmp_path, model)
    with torch.no_grad():
        model.weight[0,0] = float("nan")
    with pytest.raises(LoadedStateIntegrityError) as exc:
        verify_loaded_state(model, path)
    assert exc.value.report["mismatches"][0]["max_abs_error"] is None
