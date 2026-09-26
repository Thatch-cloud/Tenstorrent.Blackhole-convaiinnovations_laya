import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_shape_profiles import profile_rows


@pytest.mark.parametrize("width,bucket", [(1, 32), (32, 32), (33, 64), (65, 128), (129, 256), (257, 512), (512, 512)])
def test_padding_preserves_rows_masks_and_token_usage(width, bucket):
    torch = pytest.importorskip("torch")
    ids = torch.arange(2 * width).reshape(2, width)
    mask = torch.ones_like(ids)
    mask[1, -1] = 0
    inputs = (ids, mask, torch.zeros((2, 1), dtype=torch.long), torch.ones((2, 1), dtype=torch.bool), torch.tensor([0, 2]))
    before = [value.clone() for value in inputs]
    rows = list(profile_rows(inputs, 42))
    assert sum(int(row[1].sum()) for row in rows) == int(mask.sum())
    for index, row in enumerate(rows):
        assert row[0].shape == (1, bucket)
        assert row[2].shape == row[3].shape == (1, 64)
        assert torch.equal(row[0][0, :width], ids[index])
        assert torch.all(row[0][0, width:] == 42)
        assert not row[1][0, width:].any()
        assert not row[3][0, 1:].any()
        assert row[4].item() == inputs[4][index].item()
    rows[0][0].zero_()
    assert all(torch.equal(value, original) for value, original in zip(inputs, before))


@pytest.mark.parametrize("width,markers", [(513, 1), (32, 65), (0, 1)])
def test_no_silent_truncation(width, markers):
    torch = pytest.importorskip("torch")
    inputs = (torch.zeros((1, width), dtype=torch.long), torch.ones((1, width), dtype=torch.long),
              torch.zeros((1, markers), dtype=torch.long), torch.ones((1, markers), dtype=torch.bool), torch.tensor([0]))
    with pytest.raises(ValueError):
        list(profile_rows(inputs, 0))


@pytest.mark.parametrize("bucket", [True, 16, 33, 1024])
def test_explicit_bucket_rejects_unsupported_or_truncating_values(bucket):
    torch = pytest.importorskip("torch")
    inputs = (torch.zeros((1, 32), dtype=torch.long), torch.ones((1, 32), dtype=torch.long),
              torch.zeros((1, 1), dtype=torch.long), torch.ones((1, 1), dtype=torch.bool), torch.tensor([0]))
    with pytest.raises(ValueError, match="Requested bucket"):
        list(profile_rows(inputs, 0, sequence_bucket=bucket))
