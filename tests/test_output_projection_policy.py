import copy

import pytest

torch = pytest.importorskip("torch")
from laya_tt.precision import apply_output_projection_policy, buffer_inventory


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.Module()
        for family in ("attn", "mlp"):
            module = torch.nn.Module()
            module.Wi = torch.nn.Linear(4, 4)
            module.Wo = torch.nn.Linear(4, 4)
            setattr(self.encoder.layer, family, module)
        self.encoder.norm = torch.nn.LayerNorm(4)
        self.act_head = torch.nn.Linear(4, 2)
        self.register_buffer("temperature", torch.ones(2))

    def forward(self, value):
        for module in (self.encoder.layer.attn, self.encoder.layer.mlp):
            value = value + module.Wo(torch.relu(module.Wi(value)))
        return self.act_head(self.encoder.norm(value))


def test_exact_cast_arithmetic_and_fp32_preservation():
    torch.manual_seed(42)
    model = ToyModel().eval()
    original = copy.deepcopy(model)
    buffers = buffer_inventory(model)
    metadata = apply_output_projection_policy(model)
    assert metadata["policy"] == "bf16_wo_fp32_rest"
    assert len(metadata["input_cast_modules"]) == 2
    assert metadata["buffers"] == buffers == buffer_inventory(model)
    for name, parameter in model.named_parameters():
        source = dict(original.named_parameters())[name]
        expected = source.bfloat16() if ".Wo." in name else source
        assert parameter.dtype == expected.dtype
        assert torch.equal(parameter, expected)
    value = torch.randn(3, 4)
    expected = value.clone()
    for module in (original.encoder.layer.attn, original.encoder.layer.mlp):
        inner = torch.relu(module.Wi(expected)).bfloat16()
        projection = torch.nn.functional.linear(inner, module.Wo.weight.bfloat16(),
                                                module.Wo.bias.bfloat16())
        assert projection.dtype == torch.bfloat16
        expected = expected + projection
        assert expected.dtype == torch.float32
    expected = original.act_head(original.encoder.norm(expected))
    assert torch.equal(model(value), expected)
    cloned = copy.deepcopy(model)
    assert cloned.encoder.layer.attn.Wo.forward.__self__ is cloned.encoder.layer.attn.Wo
    assert torch.equal(cloned(value), expected)
    # Full-graph capture checks bound-method tracing without requiring TT.
    compiled = torch.compile(cloned, backend="eager", fullgraph=True)
    assert torch.equal(compiled(value), expected)


@pytest.mark.parametrize("problem", ["unknown", "dtype", "buffer", "shared", "forward", "missing"])
def test_invalid_model_rejected_before_mutation(problem):
    model = ToyModel()
    if problem == "unknown":
        model.unknown = torch.nn.Linear(4, 4)
    elif problem == "dtype":
        model.act_head.double()
    elif problem == "buffer":
        model.temperature = model.temperature.double()
    elif problem == "shared":
        model.encoder.layer.attn.Wi.weight = model.encoder.layer.attn.Wo.weight
    elif problem == "forward":
        model.encoder.layer.mlp.Wo.forward = lambda value: value
    else:
        del model.encoder.layer
    before = {name: value.clone() for name, value in model.state_dict().items()}
    forwards = {name: module.forward for name, module in model.named_modules()}
    with pytest.raises(ValueError):
        apply_output_projection_policy(model)
    for name, value in model.state_dict().items():
        assert value.dtype == before[name].dtype
        assert torch.equal(value, before[name])
    for name, module in model.named_modules():
        assert module.forward == forwards[name]
