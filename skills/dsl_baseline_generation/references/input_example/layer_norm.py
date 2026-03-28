import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Functional implementation of LayerNorm.

    Args:
        x (torch.Tensor): Input tensor of shape (*, normalized_shape).
        weight (torch.Tensor): Weight tensor of shape (normalized_shape).
        bias (torch.Tensor): Bias tensor of shape (normalized_shape).
        eps (float): Epsilon parameter for numerical stability.

    Returns:
        torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
    """
    normalized_shape = tuple(x.shape[-len(weight.shape):])
    return F.layer_norm(
        x, normalized_shape=normalized_shape, weight=weight, bias=bias, eps=eps
    )


class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self):
        """
        Initializes the LayerNorm layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).
            weight (torch.Tensor): Weight tensor of shape (normalized_shape).
            bias (torch.Tensor): Bias tensor of shape (normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return module_fn(x, weight, bias)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    weight = torch.ones(features, dim1, dim2)
    bias = torch.zeros(features, dim1, dim2)
    return [x, weight, bias]

def get_init_inputs():
    return []
