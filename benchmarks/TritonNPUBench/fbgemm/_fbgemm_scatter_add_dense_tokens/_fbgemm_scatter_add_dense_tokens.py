# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test__fbgemm_scatter_add_dense_tokens_npu.py
# Main kernel: _fbgemm_scatter_add_dense_tokens
# PT file: test__fbgemm_scatter_add_dense_tokens_v2.pt

import triton
import triton.language as tl


# === _fbgemm_scatter_add_dense_tokens ===
@triton.jit
def _fbgemm_scatter_add_dense_tokens(
    out_tokens,
    in_tokens,
    token_indices,
    valid_token_count,
    D: tl.constexpr,
    BLOCK_D_OUTER: tl.constexpr,
    BLOCK_D_INNER: tl.constexpr,
):
    input_token_index = tl.program_id(0).to(tl.int64)
    feature_offset = tl.program_id(1) * BLOCK_D_OUTER + tl.arange(0, BLOCK_D_INNER)[:]

    if valid_token_count is not None:
        valid_token_count = tl.load(
            valid_token_count, None, eviction_policy="evict_last"
        )
        if input_token_index >= valid_token_count:
            return

    output_token_index = tl.load(
        token_indices + input_token_index, None, eviction_policy="evict_last"
    ).to(tl.int64)

    for _ in range(0, BLOCK_D_OUTER // BLOCK_D_INNER):
        input_token_value = tl.load(
            in_tokens + input_token_index * D + feature_offset,
            None,
            eviction_policy="evict_first",
        )

        tl.atomic_add(
            out_tokens + output_token_index * D + feature_offset,
            input_token_value,
            None,
            sem="relaxed",
        )
        feature_offset += BLOCK_D_INNER

