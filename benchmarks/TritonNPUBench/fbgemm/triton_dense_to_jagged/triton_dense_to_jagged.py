# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test_triton_dense_to_jagged_npu.py
# Main kernel: triton_dense_to_jagged
# PT file: test_triton_dense_to_jagged_v2.pt

import triton
import triton.language as tl


# === triton_dense_to_jagged ===
@triton.jit
def triton_dense_to_jagged(
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_value_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_offsets_ptr,
    jagged_value_row_stride: int,
    # pyre-fixme[2]: Parameter must be annotated.
    output_dense_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_indices_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_col_stride,  # stride of output dense with dimension (z,y,x)
    dense_row_stride: int,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_matrix_stride,
    JAGGED_DIM: tl.constexpr,  # number of dimension of jagged tensor
    thread_block_row_size: tl.constexpr,
    thread_block_col_size: tl.constexpr,
    operation_function: tl.constexpr,  # fusion arithmetic opeartion function and it's input dense
    # pyre-fixme[2]: Parameter must be annotated.
    operation_jagged_value_ptr,
) -> None:
    pid = tl.program_id(0)

    begin = tl.load(jagged_offsets_ptr + pid)
    end = tl.load(jagged_offsets_ptr + (pid + 1))

    # size of the current value offset range (M , N)
    N = jagged_value_row_stride
    M = end - begin

    dense_boundary_col = dense_row_stride
    # tl.minimum will change the return type cased compile issue
    # in that case use if statement instead
    if N < dense_row_stride:
        dense_boundary_col = N

    dense_boundary_row = tl.minimum(dense_matrix_stride // dense_row_stride, M)

    jagged_value_ptr += begin * jagged_value_row_stride
    if JAGGED_DIM > 2:
        dense_indice = tl.load(dense_indices_ptr + pid)
        # if dense output range we set dense_boundary to -1
        # that mean dense values will not be use with mask
        # since we still need the calculation of fusion step
        # therefore we do not do return here
        if dense_indice == -1:
            dense_boundary_col = -1
        else:
            output_dense_ptr += dense_indice
    else:
        output_dense_ptr += pid * dense_matrix_stride

    if operation_function is not None:
        operation_jagged_value_ptr += begin * jagged_value_row_stride

    offset_row = tl.arange(0, thread_block_row_size)

    for _i in range(begin, end, thread_block_row_size):
        offset_col = tl.arange(0, thread_block_col_size)
        block_offset = (
            offset_row[:, None] * dense_row_stride
            + offset_col[None, :] * dense_col_stride
        )

        for _j in range(0, N, thread_block_col_size):
            dense_mask = (offset_row[:, None] < dense_boundary_row) & (
                offset_col[None, :] < dense_boundary_col
            )
            jagged_mask = (offset_row[:, None] < M) & (offset_col[None, :] < N)
            dense_values = tl.load(
                output_dense_ptr + block_offset, mask=dense_mask, other=0
            )
            if operation_function is not None:
                operation_jagged_value = tl.load(
                    operation_jagged_value_ptr + block_offset, mask=jagged_mask, other=0
                )
                if operation_function == "add":
                    dense_values = tensor_elementwise_add(
                        dense_values, operation_jagged_value
                    )
                else:
                    dense_values = tensor_elementwise_mul(
                        dense_values, operation_jagged_value
                    )
            tl.store(jagged_value_ptr + block_offset, dense_values, mask=jagged_mask)
            offset_col += thread_block_col_size
            block_offset += thread_block_col_size
        offset_row += thread_block_row_size

