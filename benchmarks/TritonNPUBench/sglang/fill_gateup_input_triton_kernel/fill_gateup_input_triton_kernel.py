# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_fill_gateup_input_triton_kernel.py
# Main kernel: fill_gateup_input_triton_kernel
# PT file: test_fill_gateup_input_triton_kernel_v2.pt

import triton
import triton.language as tl


# === fill_gateup_input_triton_kernel ===
@triton.jit
def fill_gateup_input_triton_kernel(
    input_ptr,
    scale_ptr,
    gateup_input_ptr,
    gateup_input_scale_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk,
    hidden_size,
    scale_size,
    BLOCK_SIZE: tl.constexpr,
):

    src_idx_int32 = tl.program_id(0)
    src_idx = src_idx_int32.to(tl.int64)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    src_ptr = input_ptr + src_idx * hidden_size
    scale_src_ptr = scale_ptr + src_idx * scale_size

    vec = tl.arange(0, BLOCK_SIZE)
    for idx in range(topk):
        expert_id = tl.load(topk_ids_ptr + idx)
        if expert_id >= 0:
            dst_idx_int32 = tl.load(src2dst_ptr + idx)
            dst_idx = dst_idx_int32.to(tl.int64)
            dst_ptr = gateup_input_ptr + dst_idx * hidden_size
            for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
                offset = start_offset + vec
                mask = offset < hidden_size
                in_data = tl.load(src_ptr + offset, mask=mask)
                tl.store(dst_ptr + offset, in_data, mask=mask)
            scale_dst_ptr = gateup_input_scale_ptr + dst_idx * scale_size
            for start_offset in tl.range(0, scale_size, BLOCK_SIZE):
                offset = start_offset + vec
                mask = offset < scale_size
                in_scale = tl.load(scale_src_ptr + offset, mask=mask)
                tl.store(scale_dst_ptr + offset, in_scale, mask=mask)

