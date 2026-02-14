import torch
import math 

def masked_sum(values: torch.Tensor, mask: torch.Tensor, axis: int | tuple[int, ...] | None = None) -> torch.Tensor:
    valid_values = torch.where(mask.bool(), values, 0.0)
    return (valid_values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(axis=axis) + 1e-8)