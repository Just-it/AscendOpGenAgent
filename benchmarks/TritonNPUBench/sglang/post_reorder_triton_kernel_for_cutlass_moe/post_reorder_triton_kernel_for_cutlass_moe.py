# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_post_reorder_triton_kernel_for_cutlass_moe.py
# Main kernel: post_reorder_triton_kernel_for_cutlass_moe
# PT file: test_post_reorder_triton_kernel_for_cutlass_moe_v2.pt

import triton
import triton.language as tl


# === post_reorder_triton_kernel_for_cutlass_moe ===
@triton.jit
def post_reorder_triton_kernel_for_cutlass_moe(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    num_local_experts,
    topk,
    num_tokens,
    hidden_size,
    routed_scaling_factor: float,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    OutDtype = output_ptr.dtype.element_ty

    offset = BLOCK_SIZE * tl.program_id(1) + tl.arange(0, BLOCK_SIZE)
    mask = offset < hidden_size

    down_output_ptr_offs = down_output_ptr + offset
    output_ptr_offs = output_ptr + offset

    start_src_idx = tl.program_id(0)
    step = tl.num_programs(0)

    for src_idx_int32 in tl.range(
        start_src_idx, num_tokens, step, num_stages=NUM_STAGES
    ):
        src_idx = src_idx_int32.to(tl.int64)
        token_src2dst_ptr = src2dst_ptr + src_idx * topk
        token_topk_ids_ptr = topk_ids_ptr + src_idx * topk
        token_topk_weights_ptr = topk_weights_ptr + src_idx * topk

        sum_vec = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for idx in range(topk):
            expert_id = tl.load(token_topk_ids_ptr + idx)
            if expert_id != num_local_experts:
                dst_idx_int32 = tl.load(token_src2dst_ptr + idx)
                dst_idx = dst_idx_int32.to(tl.int64)
                dst_idx = dst_idx
                weight_scale = tl.load(token_topk_weights_ptr + idx).to(tl.float32)
                load_ptr_offs = down_output_ptr_offs + dst_idx * hidden_size
                in_data = tl.load(load_ptr_offs, mask=mask).to(tl.float32)
                sum_vec += in_data * weight_scale
        sum_vec *= routed_scaling_factor
        store_ptr_offs = output_ptr_offs + src_idx * hidden_size
        tl.store(store_ptr_offs, sum_vec.to(OutDtype), mask=mask)

