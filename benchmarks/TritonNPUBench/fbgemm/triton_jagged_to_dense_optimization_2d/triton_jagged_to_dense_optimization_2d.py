# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test_triton_jagged_to_dense_optimization_2d_npu.py
# Main kernel: triton_jagged_to_dense_optimization_2d
# PT file: test_triton_jagged_to_dense_optimization_2d_v2.pt

import triton
import triton.language as tl


# === triton_jagged_to_dense_optimization_2d ===
@triton.jit
def triton_jagged_to_dense_optimization_2d(
    # pyre-fixme[2]: Parameter must be annotated.
    input_jagged_values_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    input_jagged_offset_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    input_jagged_row_stride,
    # pyre-fixme[2]: Parameter must be annotated.
    output_dense_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    output_dense_row_stride,
    # pyre-fixme[2]: Parameter must be annotated.
    output_dense_matrix_stride,
    thread_block_row_size: tl.constexpr,
    thread_block_col_size: tl.constexpr,
    # pyre-fixme[2]: Parameter must be annotated.
    padded_value,
    operation_function: tl.constexpr,
    # pyre-fixme[2]: Parameter must be annotated.
    operation_dense,
) -> None:
    pid = tl.program_id(0)

    # Current corresponding offset indice
    offset_idx = pid

    # begin index and end index of jagged tensor Values
    begin = tl.load(input_jagged_offset_ptr + offset_idx)
    end = tl.load(input_jagged_offset_ptr + offset_idx + 1)

    # row size of current sub tensor
    cur_jagged_tensor_row_size = end - begin

    # update dense and jagged tensor Values to corresponding address
    output_dense_ptr += pid * output_dense_matrix_stride
    input_jagged_values_ptr += begin * input_jagged_row_stride

    # also need to update the operation function if exist
    # notice dense_indice of two is same because we assume
    # the two dense + dense are same size
    if operation_function is not None:
        operation_dense += pid * output_dense_matrix_stride

    # jagged tensor row block
    offset_row = tl.arange(0, thread_block_row_size)

    # dense row and col block
    # notice jagged tensor and dense share same col block since embedding dimension is same
    dense_col_size = output_dense_row_stride
    dense_row_size = output_dense_matrix_stride // output_dense_row_stride

    for _i in range(0, dense_row_size, thread_block_row_size):
        offset_col = tl.arange(0, thread_block_col_size)
        block_offset = (
            offset_row[:, None] * output_dense_row_stride + offset_col[None, :]
        )

        for _j in range(0, dense_col_size, thread_block_col_size):

            # create mask for dense and jagged tensor for boundary check
            dense_mask = (offset_row[:, None] < dense_row_size) & (
                offset_col[None, :] < dense_col_size
            )
            jagged_mask = (offset_row[:, None] < cur_jagged_tensor_row_size) & (
                offset_col[None, :] < input_jagged_row_stride
            )

            # get value from jagged tesnor
            jagged_val = tl.load(
                input_jagged_values_ptr + block_offset,
                mask=jagged_mask,
                other=padded_value,
            )

            # do fusion operation if need
            if operation_function is not None:
                operation_dense_val = tl.load(
                    operation_dense + block_offset, mask=dense_mask, other=0.0
                )
                jagged_val = operation_function(operation_dense_val, jagged_val)

            # load value into empty dense
            tl.store(output_dense_ptr + block_offset, jagged_val, mask=dense_mask)

            # update each block
            offset_col += thread_block_col_size
            block_offset += thread_block_col_size
        offset_row += thread_block_row_size

