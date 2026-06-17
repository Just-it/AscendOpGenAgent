# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test_jagged_2_softmax_kernel_npu.py
# Main kernel: jagged_2_softmax_kernel
# PT file: test_jagged_2_softmax_kernel_v2.pt

import triton
import triton.language as tl


# === jagged_2_softmax_kernel ===
@triton.jit
def jagged_2_softmax_kernel(
    input_ptr,
    output_ptr,
    offsets_row_ptr,  # seq
    offsets_col_ptr,  # head
    offsets_overall_ptr,  # offsets for overall matrix = seq_length_i * head_i
    input_stride,
    output_stride,
    transpose,  # one if a is transpose, otherwise zero
    max_seq_len_row,  # max_seq_len for row (seq)
    max_seq_len_col,  # max_seq_len for col (head)
    BLOCK_SIZE: tl.constexpr,  # BLOCK_SIZE > seq_length
):
    """
    input shape is [sum_B(Ni * Hi)]
    output shape is [sum_B(Ni * Hi)]
    Padded version = [B, N, H]
    Calculate softmax alone N dim
    Each kernel calulates softmax for 1 sample and 1 head
    offsets_row.size == offsets_col.size == offsets_overall.size
    """

    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    # start location of current example
    begin = tl.load(offsets_overall_ptr + pid_batch)
    # end = tl.load(offsets_overall_ptr + pid_batch + 1)  # noqa F841
    # end - begin = M_i * N_i

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

    if pid_head >= H:  # TODO double check the equal here
        return
    if H == 0 or N == 0:
        return

    # start of the current example
    start_ptr = input_ptr + begin * input_stride
    # offset for n
    offsets = tl.arange(0, BLOCK_SIZE)

    # Load a softmax row
    input_ptrs = (
        start_ptr
        + offsets * input_stride * stride_n
        + pid_head * input_stride * stride_h
    )  # start + n offsets + head offset
    row = tl.load(input_ptrs, mask=offsets < N, other=-float("inf"))
    row_mins_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_mins_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # calculate output ptr, should be similar to input
    output_start_ptr = output_ptr + begin * output_stride
    output_ptrs = (
        output_start_ptr
        + offsets * output_stride * stride_n
        + pid_head * output_stride * stride_h
    )
    tl.store(output_ptrs, softmax_output, mask=offsets < N)

