import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")
spec = importlib.util.spec_from_file_location(
    "rotary_probe", Path(__file__).resolve().parents[1] / "scripts/probe_rotary_cpu.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def fixture():
    model = torch.nn.Module()
    model.encoder = torch.nn.Module()
    model.encoder.config = SimpleNamespace(
        hidden_size=4, num_attention_heads=1,
        rope_parameters={kind: {"rope_theta": 100.0, "rope_type": "default"}
                         for kind in ("full_attention", "sliding_attention")})
    model.encoder.rotary_emb = torch.nn.Module()
    for kind in model.encoder.config.rope_parameters:
        for suffix in ("inv_freq", "original_inv_freq"):
            model.encoder.rotary_emb.register_buffer(
                f"{kind}_{suffix}", torch.tensor([1.0, 0.1]), persistent=False)
    return model


def test_known_formula_and_corruption_are_observed_without_repair():
    model = fixture()
    assert all(row["max_ulp"] == 0 for row in probe.observe(model))
    target = model.encoder.rotary_emb.full_attention_inv_freq
    target[1] = 0.2
    rows = probe.observe(model)
    row = next(row for row in rows if row["buffer"].endswith("full_attention_inv_freq"))
    assert row["different_elements"] == 1
    assert row["max_abs_error"] == pytest.approx(0.1)
    assert row["max_ulp"] > 1
    assert target[1].item() == pytest.approx(0.2)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_values_fail(invalid):
    model = fixture()
    model.encoder.rotary_emb.full_attention_inv_freq[0] = invalid
    with pytest.raises(ValueError, match="Nonfinite or nonpositive"):
        probe.observe(model)


def test_extra_buffers_and_unsupported_rope_fail():
    model = fixture()
    model.register_buffer("unexpected", torch.ones(1), persistent=False)
    with pytest.raises(ValueError, match="inventory"):
        probe.observe(model)
    model = fixture()
    model.encoder.config.rope_parameters["full_attention"]["rope_type"] = "dynamic"
    with pytest.raises(ValueError, match="default RoPE"):
        probe.observe(model)
