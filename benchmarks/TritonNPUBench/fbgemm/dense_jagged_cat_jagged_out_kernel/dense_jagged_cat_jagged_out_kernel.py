# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test_dense_jagged_cat_jagged_out_kernel_npu.py
# Main kernel: dense_jagged_cat_jagged_out_kernel
# PT file: dense_jagged_cat_jagged_out_kernel_v2.pt

import triton
import triton.language as tl


# === dense_jagged_cat_jagged_out_kernel ===
@triton.jit
def dense_jagged_cat_jagged_out_kernel(
    a_ptr,  # dense
    b_ptr,  # jagged
    c_ptr,  # jagged
    b_offsets_ptr,
    c_offsets_ptr,
    max_seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    b_start = tl.load(b_offsets_ptr + pid_batch)
    b_end = tl.load(b_offsets_ptr + pid_batch + 1)
    c_start = b_start + pid_batch
    N = b_end - b_start
    N = tl.minimum(N, max_seq_len)

    a = tl.load(a_ptr + pid_batch)
    tl.store(c_ptr + c_start, a)

    offs_k = tl.arange(0, BLOCK_SIZE)
    for k in range(0, N, BLOCK_SIZE):
        b_offset = k + offs_k
        b_ptrs = b_ptr + b_start + b_offset
        b = tl.load(b_ptrs, mask=b_offset < N, other=0.0)
        tl.store(c_ptr + c_start + 1 + b_offset, b, mask=b_offset < N)
    tl.store(c_offsets_ptr + pid_batch, b_start + pid_batch)

