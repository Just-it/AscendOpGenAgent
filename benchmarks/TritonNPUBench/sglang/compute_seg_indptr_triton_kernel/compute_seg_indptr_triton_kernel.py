# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_compute_seg_indptr_triton_kernel.py
# Main kernel: compute_seg_indptr_triton_kernel
# PT file: test_compute_seg_indptr_triton_kernel_v2.pt

import triton
import triton.language as tl


# === compute_seg_indptr_triton_kernel ===
@triton.jit
def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_indptr, num_toks):
    expert_id_minus_1 = tl.program_id(0) - 1
    low = 0
    high = num_toks - 1
    target_location = -1
    while low <= high:
        mid = (low + high) // 2

        if tl.load(reorder_topk_ids + mid) > expert_id_minus_1:
            high = mid - 1
        else:
            low = mid + 1
            target_location = mid
    tl.store(seg_indptr + expert_id_minus_1 + 1, target_location + 1)

