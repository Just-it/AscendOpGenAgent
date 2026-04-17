# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/vllm_operator_cases/newtest_cases/test_cp_mha_gather_cache_kernel.py
# Main kernel: cp_mha_gather_cache_kernel
# PT file: cp_mha_gather_cache_kernel_v2.pt

import triton
import triton.language as tl


# === cp_mha_gather_cache_kernel ===
@triton.jit
def cp_mha_gather_cache_kernel(
    key_cache_ptr,  # [num_blocks, page_size, num_head, head_size]
    value_cache_ptr,  # [num_blocks, page_size, num_head, head_size]
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    block_table_ptr,  # [num_batches, max_block_num]
    cu_seqlens_kv_ptr,  # [num_batches + 1]
    token_to_batch_ptr,  # [max_cum_tokens]
    seq_start_ptr,  # [num_batches]
    k_scale_ptr,
    v_scale_ptr,
    num_heads,
    head_size,
    x,
    max_block_num,
    DEQUANT: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CACHE_FORMAT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    if DEQUANT:
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)

    key_ptr_offset = key_ptr + token_id * head_size * num_heads
    value_ptr_offset = value_ptr + token_id * head_size * num_heads
    batch_idx = tl.load(token_to_batch_ptr + token_id)
    batch_start = tl.load(seq_start_ptr + batch_idx)
    token_start = tl.load(cu_seqlens_kv_ptr + batch_idx)
    batch_offset = token_id - token_start + batch_start
    block_offset = batch_offset // PAGE_SIZE
    block_id = tl.load(
        block_table_ptr + max_block_num * batch_idx + block_offset
    ).to(tl.int64)
    slot_id = batch_offset % PAGE_SIZE

    if CACHE_FORMAT == "NHD":
        # for kv cache layout as
        # K: [num_blocks, page_size, num_head, head_dim]
        # V: [num_blocks, page_size, num_head, head_dim]
        key_cache_ptr_offset = (
            key_cache_ptr
            + block_id * num_heads * head_size * PAGE_SIZE
            + slot_id * num_heads * head_size
        )
        value_cache_ptr_offset = (
            value_cache_ptr
            + block_id * num_heads * head_size * PAGE_SIZE
            + slot_id * num_heads * head_size
        )

        for i in tl.range(0, head_size * num_heads, BLOCK_SIZE):
            mask = (col_offsets + i) < head_size * num_heads
            k_reg = tl.load(key_cache_ptr_offset + col_offsets + i, mask=mask)
            v_reg = tl.load(value_cache_ptr_offset + col_offsets + i, mask=mask)
            if DEQUANT:
                k_dtype = k_reg.dtype
                v_dtype = v_reg.dtype
                k_reg = (k_reg.to(tl.float32) * k_scale).to(k_dtype)
                v_reg = (v_reg.to(tl.float32) * v_scale).to(v_dtype)
            tl.store(key_ptr_offset + col_offsets + i, k_reg, mask=mask)
            tl.store(value_ptr_offset + col_offsets + i, v_reg, mask=mask)

