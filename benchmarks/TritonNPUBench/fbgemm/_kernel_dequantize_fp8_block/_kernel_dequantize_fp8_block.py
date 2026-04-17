# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test__kernel_dequantize_fp8_block_npu.py
# Main kernel: _kernel_dequantize_fp8_block
# PT file: test__kernel_dequantize_fp8_block_v2.pt

import triton
import triton.language as tl


# === _kernel_dequantize_fp8_block ===
@triton.jit
def _kernel_dequantize_fp8_block(
    xq_ptr,
    x_scale_ptr,
    x_dequant_ptr,
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Kernel to dequantize FP8 tensor to BF16 tensor.
    Args:
        xq_ptr (tl.constexpr): Pointer to FP8 tensor.
        x_scale_ptr (tl.constexpr): Pointer to FP8 scale tensor.
        x_dequant_ptr (tl.constexpr): Pointer to BF16 tensor.
        M (tl.constexpr): M dimension of input tensor.
        K (tl.constexpr): K dimension of input tensor.
        BLOCK_M (tl.constexpr): Block size for the M dimension.
        BLOCK_K (tl.constexpr): Block size for the K dimension.
    """
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_K)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs = offs_m[:, None] * K + offs_k[None, :]
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    xq = tl.load(xq_ptr + offs, mask=mask).to(tl.bfloat16)
    x_scale = tl.load(x_scale_ptr + pid_m * k + pid_k)
    x_dequant = xq * x_scale
    tl.store(x_dequant_ptr + offs, x_dequant, mask=mask)

