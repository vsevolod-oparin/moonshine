import torch

_MASK_NEG = -1e4


def make_sliding_window_mask(
    seq_len: int,
    window_left: int,
    window_right: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    rows = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(1)
    cols = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    mask = torch.where(
        (cols >= rows - window_left) & (cols <= rows + window_right),
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(_MASK_NEG, device=device, dtype=dtype),
    )
    return mask.unsqueeze(0).unsqueeze(0)


def make_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    mask = torch.triu(torch.full((seq_len, seq_len), _MASK_NEG, device=device, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def make_padding_mask(
    lengths: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = arange < lengths.unsqueeze(1)
    return mask


def combine_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    result = None
    for m in masks:
        if m is None:
            continue
        if result is None:
            result = m.clone()
        else:
            result = result + m
    return result


def make_cross_window_mask(
    seq_len: int,
    window_left: int,
    window_right: int,
    cross_frames: int,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    mask = make_sliding_window_mask(seq_len, window_left, window_right, device, dtype)
    if chunk_size <= 0 or cross_frames <= 0:
        return mask
    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        left_boundary = slice(max(0, chunk_start - cross_frames), chunk_start)
        right_boundary = slice(chunk_end, min(seq_len, chunk_end + cross_frames))
        first_frames = slice(chunk_start, min(chunk_start + cross_frames, chunk_end))
        last_frames = slice(max(chunk_start, chunk_end - cross_frames), chunk_end)
        if chunk_start > 0:
            mask[0, 0, first_frames, left_boundary] = 0.0
        if chunk_end < seq_len:
            mask[0, 0, last_frames, right_boundary] = 0.0
    return mask
