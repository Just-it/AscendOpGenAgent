# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test__fbgemm_gather_scale_dense_tokens_npu.py
# Main kernel: _fbgemm_gather_scale_dense_tokens
# PT file: test__fbgemm_gather_scale_dense_tokens_v2.pt

import triton
import triton.language as tl


# === _fbgemm_gather_scale_dense_tokens ===
@triton.jit
def _fbgemm_gather_scale_dense_tokens(
    out,
    x,
    token_indices,
    expert_indices,
    scores,
    stride_t,
    stride_e,
    valid_token_count,
    D: tl.constexpr,
    BLOCK_D_OUTER: tl.constexpr,
    BLOCK_D_INNER: tl.constexpr,
):
    output_token_index = tl.program_id(0)
    feature_offset = tl.program_id(1) * BLOCK_D_OUTER

    if valid_token_count is not None:
        valid_token_count = tl.load(
            valid_token_count, None, eviction_policy="evict_last"
        )
        if output_token_index >= valid_token_count:
            return

    input_token_index = tl.load(
        token_indices + output_token_index, None, eviction_policy="evict_last"
    )
    input_expert_index = tl.load(
        expert_indices + output_token_index, None, eviction_policy="evict_last"
    )

    input_score = tl.load(
        scores + input_token_index * stride_t + input_expert_index * stride_e,
        None,
        eviction_policy="evict_last",
    ).to(tl.float32)

    for _ in range(0, BLOCK_D_OUTER // BLOCK_D_INNER):
        input_token_value = tl.load(
            x
            + input_token_index.to(tl.int64) * D
            + feature_offset
            + tl.arange(0, BLOCK_D_INNER)[:],
            None,
        ).to(tl.float32)
        output_token_value = input_token_value * input_score

        tl.store(
            out
            + output_token_index.to(tl.int64) * D
            + feature_offset
            + tl.arange(0, BLOCK_D_INNER)[:],
            output_token_value,
            None,
        )
        feature_offset += BLOCK_D_INNER

