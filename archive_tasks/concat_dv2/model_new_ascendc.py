import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_LIB_PATTERN = str(_KERNEL_BUILD / "_ascend_concat_dv2_ext*")
if _LIB_PATTERN not in "".join(sys.path):
    import glob as _glob
    _libs = _glob.glob(_LIB_PATTERN)
    if _libs:
        torch.ops.load_library(_libs[0])


SUPPORTED_DTYPES = (
    torch.float16,
    torch.float32,
    torch.bfloat16,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _normalize_inputs(self, inputs, concat_dim: int):
        if not isinstance(inputs, (list, tuple)) or not inputs:
            raise ValueError("inputs must be a non-empty list or tuple of tensors")
        if concat_dim != 0:
            raise ValueError(f"concat_dv2 AscendC implementation only supports concat_dim=0, got {concat_dim}")
        if len(inputs) > 4:
            raise ValueError(f"up to 4 inputs are supported, got {len(inputs)}")

        reference = inputs[0]
        if not isinstance(reference, torch.Tensor):
            raise TypeError("all inputs must be torch.Tensor")
        if reference.ndim < 1:
            raise ValueError("each input tensor must have rank >= 1")
        if reference.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype: {reference.dtype}")

        tail_shape = tuple(int(dim) for dim in reference.shape[1:])
        rank = reference.ndim
        dtype = reference.dtype

        normalized = []
        dim0_sizes = []
        for tensor in inputs:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("all inputs must be torch.Tensor")
            if tensor.ndim != rank:
                raise ValueError("all inputs must have the same rank")
            if tensor.dtype != dtype:
                raise ValueError("all inputs must have the same dtype")
            if tuple(int(dim) for dim in tensor.shape[1:]) != tail_shape:
                raise ValueError("all non-concat dimensions must match")

            dim0 = int(tensor.shape[0])
            normalized.append(tensor.contiguous().reshape(dim0, -1))
            dim0_sizes.append(dim0)

        return normalized, tail_shape, dim0_sizes, dtype

    def _pad_width_if_needed(self, inputs_2d):
        original_n = int(inputs_2d[0].shape[1])
        if original_n == 0:
            return inputs_2d, original_n

        padded_n = ((original_n + 7) // 8) * 8
        if padded_n == original_n:
            return inputs_2d, original_n

        padded_inputs = []
        for tensor in inputs_2d:
            padded = torch.zeros(
                (int(tensor.shape[0]), padded_n),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            padded[:, :original_n].copy_(tensor)
            padded_inputs.append(padded)
        return padded_inputs, original_n

    def _run_kernel(self, inputs_2d):
        input_count = len(inputs_2d)
        if input_count == 1:
            return torch.ops.npu.concat_dim0_1(inputs_2d[0])
        if input_count == 2:
            return torch.ops.npu.concat_dim0_2(inputs_2d[0], inputs_2d[1])
        if input_count == 3:
            return torch.ops.npu.concat_dim0_3(inputs_2d[0], inputs_2d[1], inputs_2d[2])
        return torch.ops.npu.concat_dim0_4(inputs_2d[0], inputs_2d[1], inputs_2d[2], inputs_2d[3])

    def forward(self, inputs, concat_dim: int = 0):
        inputs_2d, tail_shape, dim0_sizes, original_dtype = self._normalize_inputs(inputs, concat_dim)
        kernel_inputs = [tensor.to(torch.float32).contiguous() for tensor in inputs_2d]
        kernel_inputs, original_n = self._pad_width_if_needed(kernel_inputs)
        y_2d = self._run_kernel(kernel_inputs)
        if y_2d.shape[1] != original_n:
            y_2d = y_2d[:, :original_n].contiguous()
        if original_dtype != torch.float32:
            y_2d = y_2d.to(original_dtype)

        output_shape = (sum(dim0_sizes), *tail_shape)
        return y_2d.reshape(output_shape)


def get_init_inputs():
    return []
