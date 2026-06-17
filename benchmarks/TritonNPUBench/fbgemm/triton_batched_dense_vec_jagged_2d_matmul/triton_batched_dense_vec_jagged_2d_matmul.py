# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test_triton_batched_dense_vec_jagged_2d_matmul_npu.py
# Main kernel: triton_batched_dense_vec_jagged_2d_matmul
# PT file: test_triton_batched_dense_vec_jagged_2d_matmul_v2.pt

import triton
import triton.language as tl


# === triton_batched_dense_vec_jagged_2d_matmul ===
@triton.jit
def triton_batched_dense_vec_jagged_2d_matmul(
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_tensor_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_offset,
    thread_block_col_size: tl.constexpr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_row_stride,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_value_row_stride,
    # pyre-fixme[2]: Parameter must be annotated.
    D,
    H: tl.constexpr,
    # pyre-fixme[2]: Parameter must be annotated.
    output_ptr,
) -> None:

    pid = tl.program_id(0)

    # number of kernel need for with matrix (N,D) calculated by D // thread_block_col_size
    GRID_DIM_COL = (D + thread_block_col_size - 1) // thread_block_col_size

    # current output row index
    output_row_idx = pid // GRID_DIM_COL

    # current jagged tensor offset index
    jagged_offset_id = output_row_idx // H

    # current index with D reference since the real shape of jagged values is [B , N , H * D]
    D_refer_idx = output_row_idx % H

    # current part of [N * D] id
    group_id = pid % GRID_DIM_COL

    # size of tile
    offset = group_id * thread_block_col_size + tl.arange(0, thread_block_col_size)

    # begin index and end index of values
    begin = tl.load(jagged_offset + jagged_offset_id)
    end = tl.load(jagged_offset + (jagged_offset_id + 1))

    # update each pointer to the correct address
    dense_ptr += output_row_idx * dense_row_stride
    jagged_tensor_ptr += begin * jagged_value_row_stride + D_refer_idx * D
    output_ptr += D * output_row_idx

    # Number of row each kernel will go through
    num_row = tl.minimum(end - begin, dense_row_stride)

    # accumulation variable use for matmul
    acc = tl.zeros((thread_block_col_size,), dtype=tl.float32)
    mask = offset < D
    for i in range(num_row):
        val1 = tl.load(dense_ptr + i)
        val2 = tl.load(jagged_tensor_ptr + offset, mask=mask, other=0.0)
        result = val1 * val2
        acc += result
        jagged_tensor_ptr += jagged_value_row_stride

    tl.store(output_ptr + offset, acc, mask=mask)

