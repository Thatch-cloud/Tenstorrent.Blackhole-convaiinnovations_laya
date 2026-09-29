"""Explicit experimental mixed BF16/FP32 model policy; not release qualification."""
import hashlib
import types


def _output_projection_forward(module, value):
    import torch.nn.functional as functional
    return functional.linear(value.to(module.weight.dtype), module.weight, module.bias)


def apply_output_projection_policy(model):
    """Use BF16 only for encoder attention/MLP Wo; validate before mutation.

    Start from a verified CPU FP32 checkpoint. Selected linear inputs are cast
    to BF16 and their outputs remain BF16; the surrounding FP32 residual path
    promotes them naturally. This is an experimental policy, not qualification.
    """
    import torch
    before = buffer_inventory(model)
    parameters = dict(model.named_parameters())
    selected = {
        name: module for name, module in model.named_modules()
        if name.startswith("encoder.")
        and name.endswith((".attn.Wo", ".mlp.Wo"))
        and isinstance(module, torch.nn.Linear)
    }
    if not selected:
        raise ValueError("No encoder output projections found")
    for name, parameter in parameters.items():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ValueError(f"Policy requires verified CPU FP32 parameters: {name}")
        if not name.startswith(("encoder.", "head.", "type_emb.", "scorer.", "act_head.")):
            raise ValueError(f"Unknown parameter outside audited policy: {name}")
    # Reject shared parameters across the precision boundary instead of silently
    # converting a parameter that also belongs to an FP32 operation.
    targets = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        target = name.rsplit(".", 1)[0] in selected
        previous = targets.setdefault(id(parameter), target)
        if previous != target:
            raise ValueError("Shared parameter crosses precision boundary")
    for name, module in selected.items():
        if "forward" in module.__dict__:
            raise ValueError(f"Output projection already has a custom forward: {name}")
    for name, parameter in parameters.items():
        if targets[id(parameter)]:
            parameter.data = parameter.detach().to(dtype=torch.bfloat16)
    for module in selected.values():
        module.forward = types.MethodType(_output_projection_forward, module)
    if buffer_inventory(model) != before:
        raise ValueError("Precision conversion changed a buffer")
    return {
        "policy": "bf16_wo_fp32_rest",
        "parameters": {name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                       for name, value in parameters.items()},
        "buffers": before,
        "input_cast_modules": sorted(selected),
    }


def buffer_inventory(model):
    import torch
    result = {}
    for name, value in model.named_buffers():
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(f"Expected FP32 buffer: {name}")
        array = value.detach().cpu().contiguous().numpy()
        result[name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}
    return result


def apply_candidate_policy(model):
    import torch
    before = buffer_inventory(model)
    parameters = {}
    for name, parameter in model.named_parameters():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ValueError(f"Policy requires verified CPU FP32 parameters: {name}")
        if name.startswith("act_head."):
            target = torch.float32
        elif name.startswith(("encoder.", "head.", "type_emb.", "scorer.")):
            target = torch.bfloat16
        else:
            raise ValueError(f"Unknown parameter outside audited policy: {name}")
        parameter.data = parameter.detach().to(dtype=target)
        parameters[name] = {"dtype": str(parameter.dtype), "shape": list(parameter.shape)}
    if buffer_inventory(model) != before:
        raise ValueError("Precision conversion changed a buffer")
    return {"parameters": parameters, "buffers": before}

