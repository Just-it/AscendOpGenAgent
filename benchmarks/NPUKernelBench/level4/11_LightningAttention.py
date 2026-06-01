import json
import os
import math
import torch
import torch.nn as nn

class Model(nn.Module):
    """Lightning Attention module.

    This module implements a linear attention variant that replaces the standard
    softmax attention with a head-specific exponentially decaying mask. Each
    attention head is assigned a unique decay slope (computed via ``get_slopes``),
    producing a causal lower-triangular mask of the form
    ``exp(-slope * (i - j))`` for ``i >= j``. Consequently, different heads
    attend to information at different effective distances, yielding multi-scale
    receptive fields without explicit softmax normalization.

    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply Lightning Attention to the input queries, keys, and values.

        Args:
            q: Query tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
            k: Key tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
            v: Value tensor of shape ``(batch, num_heads, seq_len, value_dim)``,
               where ``value_dim`` is fixed at 64 and may differ from ``head_dim``.

        Returns:
            Output tensor of shape ``(batch, num_heads, seq_len, value_dim)``.
        """
        b, h, n, d = q.shape

        def get_slopes(h):
            def get_slopes_power_of_2(h):
                start = 2 ** (-(2 ** -(math.log2(h) - 3)))
                ratio = start
                return [start * ratio**i for i in range(h)]

            if math.log2(h).is_integer():
                return get_slopes_power_of_2(
                    h
                )  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2 ** math.floor(
                    math.log2(h)
                )  # when the number of heads is not a power of 2, we use this workaround.
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: h - closest_power_of_2]
                )

        # h, 1, 1
        slopes = torch.tensor(get_slopes(h)).reshape(
            h, 1, 1
        ).to(q.device).to(torch.float32)


        arr = []
        for val in slopes:
            slope = val.item()
            mask = torch.triu(torch.zeros(n, n, device=q.device).float().fill_(float("-inf")), 1)
            # -n, ..., -2, -1, 0
            for i in range(n):
                x = torch.arange(i + 1, device=q.device)
                y = slope * x
                mask[i, : i + 1] = -torch.flip(y, [0])
            arr.append(torch.exp(mask))
        mask = torch.stack(arr, dim=0)

        qk = torch.matmul(q, k.transpose(2, 3))
        qk = (qk.to(torch.float32) * mask).to(q.dtype)
        o = torch.matmul(qk, v)
        return o


def get_input_groups():
    """Generate input groups from JSON test cases."""
    json_path = os.path.join(os.path.dirname(__file__), os.path.splitext(os.path.basename(__file__))[0] + '.json')
    input_groups = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            inputs = case['inputs']
            tensors = {}
            for inp in inputs:
                if inp['type'] == 'tensor':
                    name = inp['name']
                    dtype_str = inp.get('dtype', 'float32')
                    shape = inp.get('shape')
                    if shape is None:
                        tensors[name] = None
                    elif dtype_str == 'bool':
                        tensors[name] = (torch.rand(shape) > 0.5).to(torch.bool)
                    elif dtype_str in ('int32', 'int64', 'int8'):
                        max_val = {'int32': 1000, 'int64': 10000, 'int8': 127}.get(dtype_str, 100)
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}[dtype_str]
                        tensors[name] = torch.randint(0, max_val, shape, dtype=dtype)
                    else:
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}.get(dtype_str, torch.float32)
                        tensors[name] = torch.randn(shape, dtype=dtype) / 10
                elif inp['type'] == 'attr':
                    tensors[inp['name']] = inp['value']

            # Build input list in order matching forward signature
            group = []
            for inp in inputs:
                group.append(tensors[inp['name']])
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
