# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test_jagged_2_softmax_backward_kernel_npu.py
# Main kernel: jagged_2_softmax_backward_kernel
# PT file: test_jagged_2_softmax_backward_kernel_v2.pt

import triton
import triton.language as tl


# === jagged_2_softmax_backward_kernel ===
@triton.jit
def jagged_2_softmax_backward_kernel(
    grad_output_ptr,  # input
    softmax_output_ptr,
    grad_input_ptr,  # return value
    offsets_row_ptr,
    offsets_col_ptr,
    offsets_overall_ptr,
    grad_output_stride,
    softmax_output_stride,
    grad_input_stride,
    transpose,  # transpose
    max_seq_len_row: tl.constexpr,
    max_seq_len_col: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    begin = tl.load(offsets_overall_ptr + pid_batch)
    # end = tl.load(offsets_overall_ptr + pid_batch + 1)  # noqa F841

    # softmax on row
    if transpose:
        N = tl.load(offsets_row_ptr + pid_batch + 1) - tl.load(
            offsets_row_ptr + pid_batch
        )
        H = tl.load(offsets_col_ptr + pid_batch + 1) - tl.load(
            offsets_col_ptr + pid_batch
        )
        stride_n = H
        stride_h = H // H  # 1
        # sometimes H is larger than max_seq_len_col
        H = tl.minimum(max_seq_len_col, H)
        N = tl.minimum(max_seq_len_row, N)
    # softmax on col
    else:
        N = tl.load(offsets_col_ptr + pid_batch + 1) - tl.load(
            offsets_col_ptr + pid_batch
        )
        H = tl.load(offsets_row_ptr + pid_batch + 1) - tl.load(
            offsets_row_ptr + pid_batch
        )
        stride_h = N
        stride_n = N // N  # 1
        H = tl.minimum(max_seq_len_row, H)
        N = tl.minimum(max_seq_len_col, N)

    if pid_head >= H:
        return
    if H == 0 or N == 0:
        pass

    start_ptr = grad_output_ptr + begin * grad_output_stride
    offsets = tl.arange(0, BLOCK_SIZE)

    grad_output_ptrs = (
        start_ptr
        + offsets * grad_output_stride * stride_n
        + pid_head * grad_output_stride * stride_h
    )
    softmax_output_ptrs = (
        softmax_output_ptr
        + begin * softmax_output_stride
        + offsets * softmax_output_stride * stride_n
        + pid_head * softmax_output_stride * stride_h
    )

    grad_output_row = tl.load(grad_output_ptrs, mask=offsets < N, other=0.0)
    softmax_output_row = tl.load(softmax_output_ptrs, mask=offsets < N, other=0.0)

    sum_value = tl.sum(grad_output_row * softmax_output_row, axis=0)
    grad_input_row = (grad_output_row - sum_value) * softmax_output_row

    grad_input_row_start_ptr = grad_input_ptr + begin * grad_input_stride
    grad_input_ptrs = (
        grad_input_row_start_ptr
        + offsets * grad_input_stride * stride_n
        + pid_head * grad_input_stride * stride_h
    )
    tl.store(grad_input_ptrs, grad_input_row, mask=offsets < N)

