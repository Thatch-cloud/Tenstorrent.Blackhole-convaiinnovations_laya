import pytest

torch = pytest.importorskip("torch")
from laya_tt.profiled_forward import profiled_forward


def inputs():
    return (torch.tensor([[3, 4, 5], [6, 7, 0]]),
            torch.tensor([[1, 1, 1], [1, 1, 0]]),
            torch.tensor([[1, 2], [1, 0]]),
            torch.tensor([[True, True], [True, False]]), torch.tensor([0, 1]))


def test_profiled_execution_owns_reused_buffers_and_preserves_row_order():
    decisions = torch.empty(1, 64)
    actions = torch.empty(1, 2)
    calls = []

    def model(ids, attention, positions, markers, qtype):
        calls.append((ids.clone(), attention.clone(), positions.clone(), markers.clone()))
        decisions.fill_(-float("inf"))
        decisions[markers] = ids[0, 0].float()
        actions.fill_(qtype.item())
        return decisions, actions

    result, action = profiled_forward(model, inputs(), device="cpu", pad_token_id=42)
    assert result.shape == (2, 2) and action.shape == (2, 2)
    assert result[0].tolist() == [3., 3.]
    assert result[1, 0] == 6 and torch.isneginf(result[1, 1])
    assert action.tolist() == [[0., 0.], [1., 1.]]
    assert len(calls) == 2
    assert calls[0][0].shape == (1, 32) and calls[0][2].shape == (1, 64)
    assert calls[0][0][0, 3:].tolist() == [42] * 29
    assert calls[1][1][0, :3].tolist() == [1, 1, 0]
    decisions.zero_(); actions.zero_()
    assert result[0].tolist() == [3., 3.] and action[1].tolist() == [1., 1.]


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "action_inf"])
def test_invalid_outputs_fail_without_fallback(fault):
    calls = []
    def model(*args):
        calls.append(1)
        result = torch.zeros(1, 64)
        action = torch.zeros(1, 2)
        if fault == "shape": result = torch.zeros(1, 1)
        if fault == "dtype": result = result.bfloat16()
        if fault == "nan": result[0, 0] = float("nan")
        if fault == "action_inf": action[0, 1] = float("inf")
        return result, action
    with pytest.raises(ValueError):
        profiled_forward(model, inputs(), device="cpu", pad_token_id=0)
    assert len(calls) == 1


def test_requested_device_mismatch_is_rejected():
    def model(*args):
        return torch.zeros(1, 64), torch.zeros(1, 2)
    with pytest.raises(ValueError, match="requested device"):
        profiled_forward(model, inputs(), device="meta", pad_token_id=0)


def test_backend_failure_propagates_without_retry():
    calls = []
    def model(*args):
        calls.append(1)
        raise RuntimeError("compiler execution failed")
    with pytest.raises(RuntimeError, match="compiler execution failed"):
        profiled_forward(model, inputs(), device="cpu", pad_token_id=0)
    assert len(calls) == 1
