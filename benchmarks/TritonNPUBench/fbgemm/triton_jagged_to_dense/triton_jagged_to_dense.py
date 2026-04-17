# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test_triton_jagged_to_dense_npu.py
# Main kernel: triton_jagged_to_dense
# PT file: test_triton_jagged_to_dense_v2.pt

import triton
import triton.language as tl


@triton.jit
def tensor_elementwise_add(x, y):
    return x + y

@triton.jit
def tensor_elementwise_mul(x, y):
    return x * y

# === triton_jagged_to_dense ===
@triton.jit
def triton_jagged_to_dense(
    # only constexpr annotations support in triton now
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_value_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_offsets_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_value_row_stride,
    # pyre-fixme[2]: Parameter must be annotated.
    output_dense_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_indices_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_col_stride,  # stride of output dense with dimension (z,y,x)
    # pyre-fixme[2]: Parameter must be annotated.
    dense_row_stride,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_matrix_stride,
    JAGGED_DIM: tl.constexpr,  # number of dimension of jagged tensor
    thread_block_row_size: tl.constexpr,
    thread_block_col_size: tl.constexpr,
    operation_function: tl.constexpr,  # fusion arithmetic operation function and it's input dense
    # pyre-fixme[2]: Parameter must be annotated.
    operation_dense,
) -> None:
    pid = tl.program_id(0)

    # begin index and end index of jagged tensor Values
    begin = tl.load(jagged_offsets_ptr + pid)
    end = tl.load(jagged_offsets_ptr + (pid + 1))

    # adjust the address of the jagged tensor Values to the correct address
    jagged_value_ptr += begin * jagged_value_row_stride

    # if it's 2D (or 1D) Jagged tensor we can direct use the offset in offsets ( since there is only one offset )
    # else we actually need to use the preprocess index to found the correct address of dense
    if JAGGED_DIM > 2:
        # read the index for current kernel
        dense_indice = tl.load(dense_indices_ptr + pid)

        # if the dense_indice is -1 which mean it's a truncation case
        # in that case we don't need to do anything since the dense
        # initialize with padded value
        if dense_indice == -1:
            return

        # adjust the address of output dense ptr to the correct address
        output_dense_ptr += dense_indice

        # also need to update the operation function if exist
        # notice dense_indice of two is same because we assume
        # the two dense + dense are same size
        if operation_function is not None:
            operation_dense += dense_indice
    else:
        output_dense_ptr += pid * dense_matrix_stride

        if operation_function is not None:
            operation_dense += pid * dense_matrix_stride

    offset_row = tl.arange(0, thread_block_row_size)

    # boundary need for the mask since it could be dense's size smaller than jagged tensor or revert case
    N = tl.minimum(dense_row_stride, jagged_value_row_stride)
    M = tl.minimum(dense_matrix_stride // dense_row_stride, end - begin)

    for _i in range(begin, end, thread_block_row_size):
        offset_col = tl.arange(0, thread_block_col_size)
        block_offset = (
            offset_row[:, None] * dense_row_stride
            + offset_col[None, :] * dense_col_stride
        )
        for _j in range(0, N, thread_block_col_size):
            mask = (offset_row[:, None] < M) & (offset_col[None, :] < N)
            jagged_val = tl.load(jagged_value_ptr + block_offset, mask=mask, other=0)

            # if there is some arithmetic operation we do the fusion computation
            if operation_function is not None:
                val1 = jagged_val
                val2 = tl.load(operation_dense + block_offset, mask=mask, other=0)
                # do the arithmetic operation
                if operation_function == "add":
                    jagged_val = tensor_elementwise_add(val1, val2)
                else:
                    jagged_val = tensor_elementwise_mul(val1, val2)

            # store the result
            tl.store(output_dense_ptr + block_offset, jagged_val, mask=mask)

            # update the block offset
            offset_col += thread_block_col_size
            block_offset += thread_block_col_size
        offset_row += thread_block_row_size

