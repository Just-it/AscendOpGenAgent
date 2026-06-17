# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_compute_problem_sizes_w4a8_kernel.py
# Main kernel: compute_problem_sizes_w4a8_kernel
# PT file: test_compute_problem_sizes_w4a8_kernel_v2.pt

import triton
import triton.language as tl


# === compute_problem_sizes_w4a8_kernel ===
@triton.jit
def compute_problem_sizes_w4a8_kernel(
    masked_m_ptr,
    problem_sizes1_ptr,
    problem_sizes2_ptr,
    n,
    k,
    num_experts,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < num_experts
    final_occurrences = tl.load(masked_m_ptr + pid, mask=mask, other=0)

    ps1_idx_0 = pid * 3
    ps1_idx_1 = ps1_idx_0 + 1
    ps1_idx_2 = ps1_idx_0 + 2

    ps2_idx_0 = pid * 3
    ps2_idx_1 = ps2_idx_0 + 1
    ps2_idx_2 = ps2_idx_0 + 2

    ps1_mask_0 = ps1_idx_0 < num_experts * 3
    ps1_mask_1 = ps1_idx_1 < num_experts * 3
    ps1_mask_2 = ps1_idx_2 < num_experts * 3
    ps2_mask_0 = ps2_idx_0 < num_experts * 3
    ps2_mask_1 = ps2_idx_1 < num_experts * 3
    ps2_mask_2 = ps2_idx_2 < num_experts * 3

    tl.store(problem_sizes1_ptr + ps1_idx_0, 2 * n, mask=ps1_mask_0)
    tl.store(problem_sizes1_ptr + ps1_idx_1, final_occurrences, mask=ps1_mask_1)
    tl.store(problem_sizes1_ptr + ps1_idx_2, k, mask=ps1_mask_2)

    tl.store(problem_sizes2_ptr + ps2_idx_0, k, mask=ps2_mask_0)
    tl.store(problem_sizes2_ptr + ps2_idx_1, final_occurrences, mask=ps2_mask_1)
    tl.store(problem_sizes2_ptr + ps2_idx_2, n, mask=ps2_mask_2)

