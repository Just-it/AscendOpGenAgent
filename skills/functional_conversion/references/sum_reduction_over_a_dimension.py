import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies sum reduction over the specified dimension using functional implementation.

    Args:
        x (torch.Tensor): Input tensor of shape (..., dim, ...).
        dim (int): Dimension to reduce over (should be -1 for the transposed tensor).

    Returns:
        torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
    """
    return torch.sum(x, dim=dim, keepdim=True)


class Model(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """
    def __init__(self):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int, fn=module_fn) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).
            dim (int): Dimension to reduce over.

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        # Transpose so that the reduction dimension becomes the last axis
        xt = x.transpose(dim, -1).contiguous()
        # Apply the reduction on the last axis
        res = fn(xt, -1)
        # Transpose back to the original layout
        return res.transpose(-1, dim)


batch_size = 16
dim1 = 256
dim2 = 256
reduce_dim = 1

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [reduce_dim]