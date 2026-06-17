# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test_jagged_softmax_kernel_npu.py
# Main kernel: jagged_softmax_kernel
# PT file: jagged_softmax_kernel_v2.pt

import triton
import triton.language as tl


# === jagged_softmax_kernel ===
@triton.jit
def jagged_softmax_kernel(
    input_ptr,
    output_ptr,
    input_offsets_ptr,
    input_row_stride,
    input_head_stride,
    output_row_stride,
    output_head_stride,
    max_seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # BLOCK_SIZE > N (seq len)
):
    """
    input shpae is [SUM_B, H]
    output shape is [SUM_B, H]
    """

    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    row_begin = tl.load(input_offsets_ptr + pid_batch)
    row_end = tl.load(input_offsets_ptr + pid_batch + 1)
    N = tl.minimum(
        max_seq_len, row_end - row_begin
    )  # number of rows to consider softmax
    if N == 0:
        return

    row_start_ptr = input_ptr + row_begin * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = (
        row_start_ptr + col_offsets * input_row_stride + pid_head * input_head_stride
    )
    row = tl.load(input_ptrs, mask=col_offsets < N, other=-float("inf"))
    row_mins_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_mins_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_begin * output_row_stride
    output_ptrs = (
        output_row_start_ptr
        + col_offsets * output_row_stride
        + pid_head * output_head_stride
    )

    tl.store(output_ptrs, softmax_output, mask=col_offsets < N)

