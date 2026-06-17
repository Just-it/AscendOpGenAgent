# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test__floor_log2_npu.py
# Main kernel: _floor_log2
# PT file: test__floor_log2_v2.pt

import triton
import triton.language as tl


# === _floor_log2 ===
@triton.jit
def _floor_log2(x):
    """Helper function to efficiently compute floor(log2(x))

    Args:
        x (Tensor): FP32 Input tensor to operate on.

    Returns:
        Tensor: Floor of log2(x).
    """
    # Helpful bit constants.
    FP32_EXP_MASK: tl.constexpr = 0x7F800000  # type: ignore[Incompatible variable type]
    FP32_EXP_OFFSET: tl.constexpr = 23  # type: ignore[Incompatible variable type]
    FP32_EXP_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]

    # View x as an integer and extract its exponent.
    x = x.to(tl.int32, bitcast=True) & FP32_EXP_MASK
    # Shift exponent down to bottom bits.
    x = x >> FP32_EXP_OFFSET
    # Remove FP32 exponent bias and return.
    return (x - FP32_EXP_BIAS).to(tl.float32)

