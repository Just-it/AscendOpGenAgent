import torch
import torch.nn as nn
import torch_npu
import custom_ops_lib

def module_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return custom_ops_lib.mse_loss_custom(predictions, targets)

class ModelNew(nn.Module):

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return module_fn(predictions, targets)