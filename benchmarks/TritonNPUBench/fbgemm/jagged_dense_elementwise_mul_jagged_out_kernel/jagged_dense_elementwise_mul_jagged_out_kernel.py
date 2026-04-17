# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test_jagged_dense_elementwise_mul_jagged_out_kernel_npu.py
# Main kernel: jagged_dense_elementwise_mul_jagged_out_kernel
# PT file: jagged_dense_elementwise_mul_jagged_out_kernel_v2.pt

import triton
import triton.language as tl


# === jagged_dense_elementwise_mul_jagged_out_kernel ===
@triton.jit
def jagged_dense_elementwise_mul_jagged_out_kernel(
    a_ptr,  # 1d jagged
    b_ptr,  # dense
    c_ptr,  # 1d jagged
    a_seq_lengths_ptr,
    a_offsets_ptr,
    stride_a,
    stride_bm,
    stride_bn,
    max_seq_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_row_block = tl.program_id(1)

    batch_offset = tl.load(a_offsets_ptr + pid_batch)
    batch_seq_len = tl.load(a_seq_lengths_ptr + pid_batch)
    truncated_seq_len = tl.minimum(batch_seq_len, max_seq_len)

    offs_row = tl.arange(0, BLOCK_M)
    offs_col = tl.arange(0, BLOCK_N)

    rows = pid_row_block * BLOCK_M + offs_row

    # a start + batch offset + row offsets + initial col offsets
    a_ptrs = (
        a_ptr
        + batch_offset * stride_a
        + rows[:, None] * truncated_seq_len
        + offs_col[None, :]
    )

    # b start + row offsets + initial col offsets
    b_ptrs = b_ptr + rows[:, None] * stride_bm + offs_col[None, :] * stride_bn

    # c start + batch offset + row offsets + initial col offsets
    c_ptrs = (
        c_ptr + batch_offset + rows[:, None] * truncated_seq_len + offs_col[None, :]
    )

    for block_start in range(0, truncated_seq_len, BLOCK_N):
        cols = block_start + offs_col
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask = (rows[:, None] < truncated_seq_len) & (cols[None, :] < truncated_seq_len)
        a = tl.load(a_ptrs, mask=mask)
        a_ptrs += BLOCK_N

        b = tl.load(b_ptrs, mask=mask)
        b_ptrs += BLOCK_N

        c = a * b
        tl.store(c_ptrs, c, mask=mask)
        c_ptrs += BLOCK_N

