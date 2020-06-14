import torch


def tensor_equals(x: torch.Tensor, y: torch.Tensor) -> bool:
    return all(x == y)
