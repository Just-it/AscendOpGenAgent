# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_post_reorder_triton_kernel.py
# Main kernel: post_reorder_triton_kernel
# PT file: test_post_reorder_triton_kernel_v2_unit2.pt

import triton
import triton.language as tl


# === post_reorder_triton_kernel ===
@triton.jit
def post_reorder_triton_kernel(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = down_output_ptr.dtype.element_ty

    src_idx_int32 = tl.program_id(0)
    src_idx = src_idx_int32.to(tl.int64)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    topk_weights_ptr = topk_weights_ptr + src_idx * topk

    store_ptr = output_ptr + src_idx * hidden_size

    vec = tl.arange(0, BLOCK_SIZE)

    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + vec
        mask = offset < hidden_size

        sum_vec = tl.zeros([BLOCK_SIZE], dtype=InDtype)
        for idx in range(topk):
            expert_id = tl.load(topk_ids_ptr + idx)
            if expert_id > 0:
                dst_idx_int32 = tl.load(src2dst_ptr + idx)
                dst_idx = dst_idx_int32.to(tl.int64)
                weigh_scale = tl.load(topk_weights_ptr + idx).to(InDtype)
                load_ptr = down_output_ptr + dst_idx * hidden_size
                in_data = tl.load(load_ptr + offset, mask=mask)
                sum_vec += in_data * weigh_scale
        tl.store(store_ptr + offset, sum_vec, mask=mask)

