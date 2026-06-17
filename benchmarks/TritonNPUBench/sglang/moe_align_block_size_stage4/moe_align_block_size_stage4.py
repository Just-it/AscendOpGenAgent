# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_moe_align_block_size_stage4.py
# Main kernel: moe_align_block_size_stage4
# PT file: test_moe_align_block_size_stage4_v2_new.pt

import triton
import triton.language as tl


# === moe_align_block_size_stage4 ===
@triton.jit
def moe_align_block_size_stage4(
        topk_ids_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        tokens_cnts_ptr,
        cumsum_ptr,
        num_experts: tl.constexpr,
        block_size: tl.constexpr,
        numel: tl.constexpr,
        tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)

