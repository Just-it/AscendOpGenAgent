# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test_jagged_self_substraction_jagged_out_kernel_npu.py
# Main kernel: jagged_self_substraction_jagged_out_kernel
# PT file: jagged_self_substraction_jagged_out_kernel_v2.pt

import triton
import triton.language as tl


# === jagged_self_substraction_jagged_out_kernel ===
@triton.jit
def jagged_self_substraction_jagged_out_kernel(
    a_ptr,  # jagged
    b_ptr,  # jagged
    a_offsets_ptr,
    b_offsets_ptr,
    max_seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_index = tl.program_id(1)

    a_offset = tl.load(a_offsets_ptr + pid_batch)
    a_length = tl.load(a_offsets_ptr + pid_batch + 1) - a_offset
    a_length = tl.minimum(a_length, max_seq_len + 1)

    if a_length <= 1:
        return

    N = a_length - 1
    if pid_index >= N:
        return

    a_cur = tl.load(a_ptr + a_offset + pid_index)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a_row = tl.load(a_ptr + a_offset + offs + 1, mask=mask)
    b = a_cur - a_row

    b_offset = tl.load(b_offsets_ptr + pid_batch)
    tl.store(b_ptr + b_offset + pid_index * N + offs, b, mask=mask)

