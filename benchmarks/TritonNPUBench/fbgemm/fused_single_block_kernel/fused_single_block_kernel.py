# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test_fused_single_block_kernel_npu.py
# Main kernel: fused_single_block_kernel
# PT file: test_fused_single_block_kernel_v2.pt

import triton
import triton.language as tl


# === fused_single_block_kernel ===
@triton.jit
def fused_single_block_kernel(
        m_sizes_ptr,  # [num_segments] input sizes
        size_cumulative_ptr,  # [num_segments + 1] cumulative size sum
        starting_row_after_padding_ptr,  # [num_segments + 1] output: padded cumsum
        belong_indices_ptr,  # [N] output: segment index
        row_within_tensor_ptr,  # [N] output: position within segment
        num_segments: tl.constexpr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        prefix_num: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_BLOCKS = tl.num_programs(0)

    offs = tl.arange(0, prefix_num)
    mask = offs < num_segments

    # Load m_sizes
    m_sizes = tl.load(m_sizes_ptr + offs, mask=mask, other=0)

    # Compute inclusive cumsum
    cumsum = tl.cumsum(m_sizes, axis=0)

    # Store cumsum at indices 1 through N
    tl.store(
        size_cumulative_ptr + offs + 1 + (num_segments + 1) * pid, cumsum, mask=mask
    )

    # Set first element to zero
    tl.store(
        size_cumulative_ptr + offs + (num_segments + 1) * pid,
        tl.zeros([1], dtype=cumsum.dtype),
        mask=(offs == 0),
    )

    if pid == 0:
        # Part 1: Compute padded cumsum (only first block does this)
        offs = tl.arange(0, prefix_num)
        mask = offs < num_segments

        # Load m_sizes
        m_sizes = tl.load(m_sizes_ptr + offs, mask=mask, other=0)

        # Compute padded sizes
        padded_sizes = ((m_sizes + 128 - 1) // 128) * 128

        # Compute inclusive cumsum
        cumsum = tl.cumsum(padded_sizes, axis=0)

        # Store at indices 1 through num_segments
        tl.store(starting_row_after_padding_ptr + offs + 1, cumsum, mask=mask)

        # Set first element to zero
        tl.store(
            starting_row_after_padding_ptr + offs,
            tl.zeros([1], dtype=cumsum.dtype),
            mask=(offs == 0),
        )
    tl.debug_barrier()
    # Part 2: Segmented arange - process N elements in chunks
    new_offs = tl.arange(0, BLOCK_SIZE) + BLOCK_SIZE * pid
    for start in range(0, N, BLOCK_SIZE * NUM_BLOCKS):
        row_idx = start + new_offs
        row_mask = row_idx < N

        # Binary search using the cumsum_regular we computed
        left = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        right = tl.zeros([BLOCK_SIZE], dtype=tl.int32) + num_segments

        for _ in range(64):  # log2(num_segments) iterations
            mid = (left + right) // 2

            # Get cumsum value at mid position
            # Since we need cumsum[0] = 0, cumsum[1] = m_sizes[0], etc.
            mid_val = tl.load(
                size_cumulative_ptr + mid + (num_segments + 1) * pid,
                mask=row_mask,
                other=0,
            )

            cond = mid_val <= row_idx
            left = tl.where(cond, mid + 1, left)
            right = tl.where(~cond, mid, right)

        belong_idx = left - 1
        tl.store(belong_indices_ptr + row_idx, belong_idx, mask=row_mask)

        # Compute row_within_tensor
        segment_start = tl.load(
            size_cumulative_ptr + (num_segments + 1) * pid + belong_idx,
            mask=row_mask,
            other=0,
        )
        row_within = row_idx - segment_start
        tl.store(row_within_tensor_ptr + row_idx, row_within, mask=row_mask)

