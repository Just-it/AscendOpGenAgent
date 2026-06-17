# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test__kernel_dequantize_fp8_packed_row_npu.py
# Main kernel: _kernel_dequantize_fp8_packed_row
# PT file: test__kernel_dequantize_fp8_packed_row_v2.pt

import triton
import triton.language as tl


# === _kernel_dequantize_fp8_packed_row ===
@triton.jit
def _kernel_dequantize_fp8_packed_row(
    xq_ptr,
    x_scale_ptr,
    x_dequant_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_xdqm,
    stride_xdqk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    """
    Kernel to dequantize FP8 tensor to BF16 tensor.
    Args:
        xq_ptr (tl.constexpr): Pointer to FP8 tensor.
        x_scale_ptr (tl.constexpr): Pointer to FP8 scale tensor.
        x_dequant_ptr (tl.constexpr): Pointer to BF16 tensor.
        M (tl.constexpr): M dimension of input tensor.
        K (tl.constexpr): K dimension of input tensor (along which scales are applied)
        BLOCK_SIZE (tl.constexpr): Block size for the K dimension.
    """
    pid = tl.program_id(axis=0)
    if USE_INT64:
        pid = pid.to(tl.int64)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    scales = tl.load(x_scale_ptr + offs_m)

    for _k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

        xq = tl.load(
            xq_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask,
            other=0.0,
        )
        x_dq = xq * scales[:, None]

        tl.store(
            x_dequant_ptr
            + offs_m[:, None] * stride_xdqm
            + offs_k[None, :] * stride_xdqk,
            x_dq,
            mask=mask,
        )
        offs_k += BLOCK_K

