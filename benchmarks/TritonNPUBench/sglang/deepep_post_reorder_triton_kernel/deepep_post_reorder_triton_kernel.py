# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_deepep_post_reorder_triton_kernel.py
# Main kernel: deepep_post_reorder_triton_kernel
# PT file: test_deepep_post_reorder_triton_kernel_v2.pt

import triton
import triton.language as tl


# === deepep_post_reorder_triton_kernel ===
@triton.jit
def deepep_post_reorder_triton_kernel(
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

    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    topk_weights_ptr = topk_weights_ptr + src_idx * topk

    store_ptr = output_ptr + src_idx * hidden_size
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        sum_vec = tl.zeros([BLOCK_SIZE], dtype=InDtype)
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                weigh_scale = tl.load(topk_weights_ptr + idx).to(InDtype)
                load_ptr = down_output_ptr + dst_idx * hidden_size
                in_data = tl.load(load_ptr + offset, mask=mask)
                sum_vec += in_data * weigh_scale
        tl.store(store_ptr + offset, sum_vec, mask=mask)

