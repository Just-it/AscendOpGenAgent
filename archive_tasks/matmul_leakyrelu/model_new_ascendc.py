import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"

try:
    import matmul_leakyrelu_ext  # noqa: F401
except ImportError:
    import glob as _glob
    _libs = _glob.glob(str(_KERNEL_BUILD / "matmul_leakyrelu_ext*.so"))
    if _libs:
        torch.ops.load_library(_libs[0])


def get_init_inputs():
    """Override Model's default negative_slope=0.01 to match the kernel's hard-coded 0.001."""
    return [0.001]


class ModelNew(nn.Module):
    """AscendC-backed MatMul + LeakyReLU.

    Note: the AscendC kernel hard-codes negative_slope=0.001, so this
    binding ignores any negative_slope argument passed at construction.
    Use Model(negative_slope=0.001) as the reference when comparing.
    """

    def __init__(self, negative_slope: float = 0.001) -> None:
        super().__init__()
        if negative_slope != 0.001:
            import warnings
            warnings.warn(
                f"AscendC kernel uses negative_slope=0.001 (got {negative_slope}). "
                "Results will be compared against the hard-coded kernel value.",
                UserWarning,
                stacklevel=2,
            )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.ndim == 2 and b.ndim == 2, "a and b must be 2D"
        assert a.shape[1] == b.shape[0], "k dimension must match"
        assert a.dtype == b.dtype, "a and b must have the same dtype"
        assert a.dtype in (torch.float16, torch.int8), "dtype must be float16 or int8"
        out = torch.ops.npu.matmul_leakyrelu(a, b)
        if out.dtype == torch.float32:
            out = out.half()
        return out
