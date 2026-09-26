"""Experimental CPU input layout shared by parity and offline compilation probes."""
BUCKETS = (32, 64, 128, 256, 512)
MARKERS = 64


def profile_rows(inputs, pad_token_id, *, sequence_bucket=None):
    """Preserve masked positions and row order; never truncate or change accounting."""
    import torch

    ids, attention, positions, markers, qtype = inputs
    if ids.ndim != 2 or attention.shape != ids.shape:
        raise ValueError("Invalid token shape")
    rows, width = ids.shape
    if not 1 <= rows <= 64 or not 1 <= width <= BUCKETS[-1]:
        raise ValueError("Input outside experimental profile bounds")
    if positions.ndim != 2 or positions.shape != markers.shape or positions.shape[0] != rows:
        raise ValueError("Invalid marker shape")
    if not 1 <= positions.shape[1] <= MARKERS or qtype.shape != (rows,):
        raise ValueError("Input outside experimental marker/type bounds")
    if any(value.device.type != "cpu" for value in inputs):
        raise ValueError("This experiment accepts CPU tensors only")
    if markers.dtype != torch.bool or not torch.all((attention == 0) | (attention == 1)):
        raise ValueError("Invalid masks")
    bucket = next(size for size in BUCKETS if size >= width)
    if sequence_bucket is not None:
        if type(sequence_bucket) is not int or sequence_bucket not in BUCKETS or sequence_bucket < width:
            raise ValueError("Requested bucket must be supported and cannot truncate")
        bucket = sequence_bucket
    for row in range(rows):
        padded_ids = ids.new_full((1, bucket), pad_token_id)
        padded_attention = attention.new_zeros((1, bucket))
        padded_positions = positions.new_zeros((1, MARKERS))
        padded_markers = markers.new_zeros((1, MARKERS))
        padded_ids[:, :width] = ids[row:row + 1]
        padded_attention[:, :width] = attention[row:row + 1]
        padded_positions[:, :positions.shape[1]] = positions[row:row + 1]
        padded_markers[:, :markers.shape[1]] = markers[row:row + 1]
        yield (padded_ids, padded_attention, padded_positions, padded_markers, qtype[row:row + 1].clone())

