"""Fixed-profile execution for an explicitly supplied model and device.

This helper does not load weights, compile, select a device, acquire ownership,
change precision, or provide fallback. The caller must serialize model access.
"""
from .shape_profiles import MARKERS, profile_rows


def profiled_forward(model, inputs, *, device, pad_token_id):
    """Return owned CPU FP32 decision/action logits in original row order.

    The model must already be prepared for the supplied device. Compiler/device
    acceptance remains the caller's responsibility; device placement alone does
    not prove numerical parity or absence of compiler fallback operations.
    """
    import torch

    device = torch.device(device)
    logits, actions = [], []
    with torch.no_grad():
        for profile in profile_rows(inputs, pad_token_id):
            outputs = model(*(value.to(device) for value in profile))
            if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
                raise ValueError("Model must return decision and action logits")
            for value, shape in zip(outputs, ((1, MARKERS), (1, 2))):
                if not isinstance(value, torch.Tensor) or value.shape != shape:
                    raise ValueError("Unexpected model output shape")
                if value.dtype != torch.float32:
                    raise ValueError("Expected FP32 model outputs")
                if value.device.type != device.type or (device.index is not None and value.device.index != device.index):
                    raise ValueError("Model output did not stay on the requested device")
            # Own results before another row can reuse a model's output buffers.
            row_logits, row_actions = (value.detach().cpu().clone() for value in outputs)
            if not torch.isfinite(row_logits[profile[3]]).all() or not torch.isfinite(row_actions).all():
                raise ValueError("Non-finite valid decision or action logits")
            logits.append(row_logits[:, :inputs[2].shape[1]])
            actions.append(row_actions)
    return torch.cat(logits), torch.cat(actions)
