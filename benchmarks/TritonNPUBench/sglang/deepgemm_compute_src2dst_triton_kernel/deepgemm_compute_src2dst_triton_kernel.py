# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_deepgemm_compute_src2dst_triton_kernel.py
# Main kernel: deepgemm_compute_src2dst_triton_kernel
# PT file: test_deepgemm_compute_src2dst_triton_kernel_v2.pt

import triton
import triton.language as tl


# === deepgemm_compute_src2dst_triton_kernel ===
@triton.jit
def deepgemm_compute_src2dst_triton_kernel(
    topk_ids,
    reorder_ids,
    seg_indptr,
    src2dst,
    m_max,
    num_toks,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    expert_id = tl.load(topk_ids + src_id, mask=(src_id < num_toks))
    expert_dst_start = tl.load(seg_indptr + expert_id, mask=(expert_id >= 0))
    expert_dst_offset = dst_id - expert_dst_start
    dst_id = expert_id * m_max + expert_dst_offset
    tl.store(src2dst + src_id, dst_id, mask=mask)

