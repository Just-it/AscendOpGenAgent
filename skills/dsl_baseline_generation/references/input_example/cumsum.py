import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Functional implementation of cumulative sum operation.

    Args:
        x (torch.Tensor): Input tensor with the reduction axis as the first axis.

    Returns:
        torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along the first axis.
    """
    return torch.cumsum(x, dim=0)


class Model(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.
    """

    def __init__(self):
        """
        Initialize the Scan model.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Forward pass for the Scan model, computing the cumulative sum along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape), where `*input_shape` 
                              can vary depending on the use case.
            dim (int): The dimension along which to perform the cumulative sum.

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        ndim = x.ndim
        perm = list(range(ndim))
        perm[0], perm[dim] = perm[dim], perm[0]
        
        xt = x.permute(*perm).contiguous()
        res = module_fn(xt)
        res = res.permute(*perm)
        
        return res


batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    """
    Generates random inputs for testing the Scan model.

    Returns:
        list: A list containing a single randomly generated tensor with shape 
              (batch_size, *input_shape).
    """
    return [torch.rand(batch_size, *input_shape), dim]

def get_init_inputs():
    """
    Returns the initialization parameters for the Scan model.

    Returns:
        list: A list containing the `dim` parameter for model initialization.
    """
    return []
