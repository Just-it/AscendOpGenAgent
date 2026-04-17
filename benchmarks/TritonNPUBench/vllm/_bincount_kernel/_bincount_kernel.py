# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/vllm_operator_cases/newtest_cases/test__bincount_kernel.py
# Main kernel: _bincount_kernel
# PT file: bincount_kernel_test_v2.pt

import triton
import triton.language as tl


# === _bincount_kernel ===
@triton.jit(do_not_specialize=["prefill_len", "prompt_len"])
def _bincount_kernel(
    prefill_token_ids_ptr,
    prefill_len,
    prompt_len,
    prompt_bin_mask_ptr,
    output_bin_counts_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if block_idx * BLOCK_SIZE >= prefill_len:
        return

    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if block_idx * BLOCK_SIZE < prompt_len:
        mask = block < prompt_len
        prefill_tokens = tl.load(prefill_token_ids_ptr + block, mask=mask)
        idx = prefill_tokens // 32
        bit_idx = prefill_tokens % 32
        bit = tl.full((BLOCK_SIZE,), 1, tl.int32) << bit_idx
        tl.atomic_or(prompt_bin_mask_ptr + idx, bit, mask=mask)
    if (block_idx + 1) * BLOCK_SIZE >= prompt_len:
        mask = block < prefill_len
        mask &= block >= prompt_len
        prefill_tokens = tl.load(prefill_token_ids_ptr + block, mask=mask)
        tl.atomic_add(output_bin_counts_ptr + prefill_tokens, 1, mask=mask)

