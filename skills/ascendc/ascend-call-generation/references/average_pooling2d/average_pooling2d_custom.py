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
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        x = x.permute(0, 2, 3, 1).contiguous()   # NCHW → NHWC
        y = fn(x, self.kernel_size)
        return y.permute(0, 3, 1, 2).contiguous()  # NHWC → NCHW