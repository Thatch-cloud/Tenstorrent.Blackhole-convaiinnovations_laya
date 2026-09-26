import importlib.util
from pathlib import Path
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_bf16_cpu.py"
SPEC = importlib.util.spec_from_file_location("bf16_candidate", PATH)
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


def toy_model():
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 2)
    model.encoder.register_buffer("inv_freq", torch.tensor([0.123456789]), persistent=False)
    model.type_emb = torch.nn.Embedding(3, 2)
    model.head = None
    model.scorer = torch.nn.Linear(2, 1)
    model.act_head = torch.nn.Linear(2, 2)
    model.register_buffer("temperature", torch.ones(3))
    return model


def test_policy_preserves_fp32_buffers_and_action_head():
    torch = pytest.importorskip("torch")
    model = toy_model()
    before = candidate.buffer_inventory(model)
    metadata = candidate.apply_candidate_policy(model)
    assert metadata["buffers"] == before == candidate.buffer_inventory(model)
    assert model.encoder.weight.dtype == model.scorer.weight.dtype == torch.bfloat16
    assert model.type_emb.weight.dtype == torch.bfloat16
    assert model.act_head.weight.dtype == model.temperature.dtype == torch.float32


def test_unknown_parameter_policy_fails_closed():
    torch = pytest.importorskip("torch")
    model = toy_model()
    model.unknown = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="Unknown parameter"):
        candidate.apply_candidate_policy(model)


def test_hooks_record_actual_dtype_and_reject_wrong_output():
    torch = pytest.importorskip("torch")
    model = toy_model()
    candidate.apply_candidate_policy(model)
    events = []
    handles, expected = candidate.install_dtype_hooks(model, events)
    try:
        model.encoder(torch.ones(1, 2, dtype=torch.bfloat16))
        assert events[-1]["outputs"][0]["dtype"] == "torch.bfloat16"
        model.act_head = model.act_head.bfloat16()
        with pytest.raises(ValueError, match="dtype mismatch"):
            model.act_head(torch.ones(1, 2, dtype=torch.bfloat16))
    finally:
        for handle in handles:
            handle.remove()


def test_decoded_metrics_separates_numeric_drift_choice_and_order():
    expected = [{"answers": {"question": {"choice": "a", "probabilities": {"a": 0.8, "b": 0.2}}}}]
    actual = [{"answers": {"question": {"choice": "a", "probabilities": {"a": 0.79, "b": 0.21}}}}]
    result = candidate.decoded_metrics(actual, expected)
    assert result["nonnumeric_and_order_exact"]
    assert result["max_numeric_absolute_error"] == pytest.approx(0.01)
    assert not result["decoded_answers_exact"]
    actual[0]["answers"]["question"]["choice"] = "b"
    assert not candidate.decoded_metrics(actual, expected)["nonnumeric_and_order_exact"]
