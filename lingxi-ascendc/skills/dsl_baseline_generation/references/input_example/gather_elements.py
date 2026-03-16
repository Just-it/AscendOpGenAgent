import torch
import torch.nn as nn

def module_fn(input: torch.Tensor, index: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Applies gather to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features).

    Returns:
        torch.Tensor: Output tensor with gather applied, same shape as index tensor.
    """
    return torch.gather(input, dim=dim, index=index)

class Model(nn.Module):
    """
    Simple model that performs a gather_elements operation.
    """
    def __init__(self, dim: int = -1):
        """
        Initializes the gather_elements model.

        Args:
            dim (int, optional): The axis along which to gather. Defaults to 1.
        """
        super(Model, self).__init__()
        self.dim = dim
    
    def forward(self, input: torch.Tensor, index: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Gathers values from input tensor along the specified dimension using indices.

        Args:
            input (torch.Tensor): The source tensor to gather from.
            index (torch.Tensor): The indices tensor specifying which elements to gather.

        Returns:
            torch.Tensor: Output tensor with gathered elements, same shape as index tensor.
        """
        return fn(input, index=index, dim=self.dim)

batch_size = 128
input_size = 8192
gather_size = 32

def get_inputs():
    # Create input tensor
    input_tensor = torch.randn(batch_size, input_size)
    
    # Create index tensor with valid indices for gathering along dim=1
    index_tensor = torch.randint(0, input_size, (batch_size, gather_size)).to(torch.int32)
    
    return [input_tensor, index_tensor]

def get_init_inputs():
    return [-1]  # Provide dim value for initialization