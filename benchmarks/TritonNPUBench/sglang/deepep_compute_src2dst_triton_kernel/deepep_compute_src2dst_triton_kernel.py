# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_deepep_compute_src2dst_triton_kernel.py
# Main kernel: deepep_compute_src2dst_triton_kernel
# PT file: test_deepep_compute_src2dst_triton_kernel_v2.pt

import triton
import triton.language as tl


# === deepep_compute_src2dst_triton_kernel ===
@triton.jit
def deepep_compute_src2dst_triton_kernel(
    reorder_ids, src2dst, num_toks, num_minus_one, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    num_invalid = tl.load(num_minus_one)
    tl.store(src2dst + src_id, dst_id - num_invalid, mask=mask)

