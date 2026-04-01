import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, num_features: int, eps: float = 1e-5) -> torch.Tensor:
    """
    Applies RMS Normalization to the input tensor using functional implementation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).
        num_features (int): Number of features in the input tensor.
        eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.

    Returns:
        torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
    """
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms


class Model(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self):
        """
        Initializes the RMSNorm layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, num_features: int, eps: float = 1e-5) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        xt = x.transpose(1, -1).contiguous()
        res = module_fn(xt, num_features, eps)
        return res.transpose(-1, 1)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x, features]

def get_init_inputs():
    return []
