# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_compute_masked_m_triton_kernel.py
# Main kernel: compute_masked_m_triton_kernel
# PT file: test_compute_masked_m_triton_kernel_v2.pt

import triton
import triton.language as tl


# === compute_masked_m_triton_kernel ===
@triton.jit
def compute_masked_m_triton_kernel(seg_indptr, masked_m):
    expert_id = tl.program_id(0)
    start = tl.load(seg_indptr + expert_id)
    end = tl.load(seg_indptr + expert_id + 1)
    tl.store(masked_m + expert_id, (end - start))

