import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """
    Applies LeakyReLU activation to the input tensor using functional implementation.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        negative_slope (float): The negative slope of the activation function. Defaults to 0.01.

    Returns:
        torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
    """
    return F.leaky_relu(x, negative_slope=negative_slope)


class Model(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(Model, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return fn(x, self.negative_slope)


batch_size = 33
dim = 16333

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed