import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    x = x.permute(0, 3, 1, 2).contiguous()
    x = F.avg_pool2d(x, kernel_size=kernel_size)
    x = x.permute(0, 2, 3, 1).contiguous()
    return x


class Model(nn.Module):
    """
    Simple model that performs 2D Average Pooling.
    """
    def __init__(self):
        """
        Initializes the Average Pooling layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            kernel_size (int): Size of the pooling window.

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        x = x.permute(0, 2, 3, 1).contiguous()
        y = module_fn(x, kernel_size)
        return y.permute(0, 3, 1, 2).contiguous()

batch_size = 16
channels = 64
height = 256
width = 256
kernel_size = 3

def get_inputs():
    x = torch.rand(batch_size, channels, height, width)
    return [x, kernel_size]

def get_init_inputs():
    return []
