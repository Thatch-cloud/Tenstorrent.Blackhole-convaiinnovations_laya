"""Explicit experimental mixed BF16/FP32 model policy; not release qualification."""
import hashlib


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


