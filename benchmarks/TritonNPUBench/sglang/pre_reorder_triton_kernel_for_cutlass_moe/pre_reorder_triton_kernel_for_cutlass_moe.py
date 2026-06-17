# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_pre_reorder_triton_kernel_for_cutlass_moe.py
# Main kernel: pre_reorder_triton_kernel_for_cutlass_moe
# PT file: test_pre_reorder_triton_kernel_for_cutlass_moe_v2.pt

import triton
import triton.language as tl


# === pre_reorder_triton_kernel_for_cutlass_moe ===
@triton.jit
def pre_reorder_triton_kernel_for_cutlass_moe(
    input_ptr,
    gateup_input_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    a1_scales_ptr,
    num_local_experts,
    topk,
    num_tokens,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty

    if a1_scales_ptr is not None:
        a1_scale = 1.0 / tl.load(a1_scales_ptr)
    else:
        a1_scale = 1.0

    offset = BLOCK_SIZE * tl.program_id(1) + tl.arange(0, BLOCK_SIZE)
    mask = offset < hidden_size

    start_src_idx = tl.program_id(0)
    step = tl.num_programs(0)

    for src_idx_int32 in tl.range(
        start_src_idx, num_tokens, step, num_stages=NUM_STAGES
    ):
        src_idx = src_idx_int32.to(tl.int64)
        token_src2dst_ptr = src2dst_ptr + src_idx * topk
        token_topk_ids_ptr = topk_ids_ptr + src_idx * topk

        src_ptr_offs = input_ptr + src_idx * hidden_size + offset
        dst_ptr_offs = gateup_input_ptr + offset
        in_data = tl.load(src_ptr_offs, mask=mask).to(tl.float32)
        out_data = (in_data * a1_scale).to(OutDtype)
        for idx in range(topk):
            expert_id = tl.load(token_topk_ids_ptr + idx)
            if expert_id != num_local_experts:
                dst_idx = tl.load(token_src2dst_ptr + idx)
                tl.store(dst_ptr_offs + dst_idx * hidden_size, out_data, mask=mask)

