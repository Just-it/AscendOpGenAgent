import torch
import torch.nn as nn
import torch.nn.functional as F
import custom_ops_lib


def module_fn(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return custom_ops_lib.average_pooling_2d_custom(x, kernel_size)


class ModelNew(nn.Module):
    """
    Simple model that performs 2D Average Pooling.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, kernel_size: int, stride: int = None, padding: int = None) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            kernel_size (int, optional): Size of the pooling window. Defaults to None (use initialized value).
            stride (int, optional): Stride of the pooling operation. Defaults to None (use initialized value).
            padding (int, optional): Padding applied to the input tensor. Defaults to None (use initialized value).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        x = x.permute(0, 2, 3, 1).contiguous()   # NCHW → NHWC
        y = module_fn(x, kernel_size)
        return y.permute(0, 3, 1, 2).contiguous()  # NHWC → NCHW