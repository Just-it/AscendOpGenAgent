# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_write_req_to_token_pool_triton.py
# Main kernel: write_req_to_token_pool_triton
# PT file: test_write_req_to_token_pool_triton_unit_v2.pt

import triton
import triton.language as tl


# === write_req_to_token_pool_triton ===
@triton.jit
def write_req_to_token_pool_triton(
        req_to_token_ptr,  # [max_batch, max_context_len]
        req_pool_indices,
        pre_lens,
        seq_lens,
        extend_lens,
        out_cache_loc,
        req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)

    # TODO: optimize this?
    cumsum_start = 0
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )

