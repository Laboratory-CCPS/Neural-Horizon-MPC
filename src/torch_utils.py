import torch
import torch.nn as nn


def count_linear_modules(model: nn.Module) -> int:
    """
    Counts the linear modules in a torch model.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            count += 1
    return count


def count_parameters(model: nn.Module,) -> int:
    """
    Summs up all parameters of a model.
    """
    return sum(p.numel() for p in model.parameters())


def count_non_zero_parameters(model: nn.Module,) -> int:
    """
    Summs up all non-zero parameters of a model.
    """
    return sum(torch.count_nonzero(p).item() for p in model.parameters())


def count_only_trainable_parameters(model: nn.Module,) -> int:
    """
    Summs up all trainable parameters of a model that require a gradient.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)