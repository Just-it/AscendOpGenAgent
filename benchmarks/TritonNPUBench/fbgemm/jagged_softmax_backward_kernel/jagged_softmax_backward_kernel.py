# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test_jagged_softmax_backward_kernel_npu.py
# Main kernel: jagged_softmax_backward_kernel
# PT file: jagged_softmax_backward_kernel_v2.pt

import triton
import triton.language as tl


# === jagged_softmax_backward_kernel ===
@triton.jit
def jagged_softmax_backward_kernel(
    grad_output_ptr,
    softmax_output_ptr,
    grad_input_ptr,  # return value
    input_offsets_ptr,
    grad_output_row_stride,
    grad_output_head_stride,
    softmax_output_row_stride,
    softmax_output_head_stride,
    grad_input_row_stride,
    grad_input_head_stride,
    max_seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    grad_output_ptr shpae is [SUM_B, H]
    softmax_output shape is [SUM_B, H]
    grad_input shape is [SUM_B, H]
    """

    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    row_begin = tl.load(input_offsets_ptr + pid_batch)
    row_end = tl.load(input_offsets_ptr + pid_batch + 1)
    N = tl.minimum(
        max_seq_len, row_end - row_begin
    )  # number of rows to consider softmax

    col_offsets = tl.arange(0, BLOCK_SIZE)
    grad_output_ptrs = (
        grad_output_ptr
        + row_begin * grad_output_row_stride
        + col_offsets * grad_output_row_stride
        + pid_head * grad_output_head_stride
    )
    softmax_output_ptrs = (
        softmax_output_ptr
        + row_begin * softmax_output_row_stride
        + col_offsets * softmax_output_row_stride
        + pid_head * softmax_output_head_stride
    )
    grad_output_row = tl.load(grad_output_ptrs, mask=col_offsets < N, other=0.0)
    softmax_output_row = tl.load(softmax_output_ptrs, mask=col_offsets < N, other=0.0)

    sum_value = tl.sum(grad_output_row * softmax_output_row, axis=0)
    grad_input_row = (grad_output_row - sum_value) * softmax_output_row
    grad_input_ptrs = (
        grad_input_ptr
        + row_begin * grad_input_row_stride
        + col_offsets * grad_input_row_stride
        + pid_head * grad_input_head_stride
    )
    tl.store(grad_input_ptrs, grad_input_row, mask=col_offsets < N)

