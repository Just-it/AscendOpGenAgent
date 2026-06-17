# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test_jagged2_to_padded_dense_kernel_npu.py
# Main kernel: jagged2_to_padded_dense_kernel
# PT file: jagged2_to_padded_dense_kernel_v2.pt

import triton
import triton.language as tl


# === jagged2_to_padded_dense_kernel ===
@triton.jit
def jagged2_to_padded_dense_kernel(
    x_ptr,
    lengths_ptr,
    offsets_ptr,
    output_dense_ptr,
    stride_b,
    stride_m,
    stride_n,
    max_length,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_batch = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    begin = tl.load(offsets_ptr + pid_batch)
    seqlen = tl.load(lengths_ptr + pid_batch)

    seqlen = tl.minimum(seqlen, max_length)
    if seqlen == 0:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + begin + offs_m[:, None] * seqlen + offs_n[None, :]
    x = tl.load(x_ptrs, mask=((offs_m[:, None] < seqlen) & (offs_n[None, :] < seqlen)))

    out_ptrs = (
        output_dense_ptr
        + pid_batch * stride_b
        + offs_m[:, None] * stride_m
        + offs_n[None, :] * stride_n
    )
    tl.store(
        out_ptrs, x, mask=((offs_m[:, None] < seqlen) & (offs_n[None, :] < seqlen))
    )

