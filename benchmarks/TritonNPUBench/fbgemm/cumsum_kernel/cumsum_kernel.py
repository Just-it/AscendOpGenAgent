# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test_cumsum_kernel_gpu.py
# Main kernel: cumsum_kernel
# PT file: test_cumsum_kernel_v2.pt

import triton
import triton.language as tl


# === cumsum_kernel ===
@triton.jit
def cumsum_kernel(
    m_sizes_ptr,
    size_cumulative_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load m_sizes
    m_sizes = tl.load(m_sizes_ptr + offs, mask=mask, other=0)

    # Compute inclusive cumsum
    cumsum = tl.cumsum(m_sizes, axis=0)

    # Store cumsum at indices 1 through N
    tl.store(size_cumulative_ptr + offs + 1, cumsum, mask=mask)

    # Set first element to zero
    first_elem_mask = offs == 0
    tl.store(
        size_cumulative_ptr + offs,
        tl.zeros([BLOCK_SIZE], dtype=cumsum.dtype),
        mask=first_elem_mask,
    )

