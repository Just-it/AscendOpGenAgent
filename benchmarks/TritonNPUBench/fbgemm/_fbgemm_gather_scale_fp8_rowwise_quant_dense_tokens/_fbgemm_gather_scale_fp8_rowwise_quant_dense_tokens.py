# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test__fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens_npu.py
# Main kernel: _fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens
# PT file: test__fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens_v2.pt

import triton
import triton.language as tl


# === _fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens ===
@triton.jit
def _fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens(
    output_ptr,
    output_scale_ptr,
    input_ptr,
    token_indices_ptr,
    expert_indices_ptr,
    scores_ptr,
    scale_ub_ptr,
    stride_t,
    stride_e,
    valid_token_count,
    D: tl.constexpr,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert(D % BLOCK_D == 0, "D must be a multiple of BLOCK_D")

    output_token_index = tl.program_id(0)

    if valid_token_count is not None:
        valid_token_count = tl.load(
            valid_token_count, None, eviction_policy="evict_last"
        )
        if output_token_index >= valid_token_count:
            return

    input_token_index = tl.load(
        token_indices_ptr + output_token_index, None, eviction_policy="evict_first"
    )
    input_expert_index = tl.load(
        expert_indices_ptr + output_token_index, None, eviction_policy="evict_first"
    )
    input_score = tl.load(
        scores_ptr + input_token_index * stride_t + input_expert_index * stride_e,
        None,
        eviction_policy="evict_first",
    ).to(tl.float32)

    row_max = 0.0
    in_2d_ptr = (
        input_ptr + input_token_index.to(tl.int64) * D + tl.arange(0, BLOCK_D)[:]
    )
    for _ in range(0, D, BLOCK_D):
        input_token_value = tl.load(
            in_2d_ptr,
            None,
            eviction_policy="evict_last",
        ).to(tl.float32)
        output_token_value = input_token_value * input_score

        tile_max = tl.max(tl.abs(output_token_value))
        row_max = tl.maximum(tile_max, row_max)
        in_2d_ptr += BLOCK_D

    # Clamp max value appropriately.
    if CLAMP_MAX:
        ub = tl.load(scale_ub_ptr, eviction_policy="evict_last")
        row_max = tl.clamp(row_max, EPS, ub)
    else:
        row_max = tl.maximum(row_max, EPS)

    # Scale and quantize.
    output_scale = MAX_FP8 / row_max
    tl.store(output_scale_ptr + output_token_index, 1.0 / output_scale)

    in_2d_ptr = (
        input_ptr + input_token_index.to(tl.int64) * D + tl.arange(0, BLOCK_D)[:]
    )
    out_2d_ptr = (
        output_ptr + output_token_index.to(tl.int64) * D + tl.arange(0, BLOCK_D)[:]
    )
    for _ in range(0, D, BLOCK_D):
        # Load from L2
        input_token_value = tl.load(
            in_2d_ptr,
            None,
            eviction_policy="evict_first",
        ).to(tl.float32)
        # Rematerilize
        output_token_value_fp8 = (input_token_value * input_score) * output_scale

        # Clamp A to fp8 range to make sure there's no overflow.
        # This is required for AMD. Nvidia's default saturation
        # handles it, but it's nice to have anyway.
        output_token_value_fp8 = tl.clamp(output_token_value_fp8, -MAX_FP8, MAX_FP8).to(
            TL_FP8_DTYPE
        )
        tl.store(
            out_2d_ptr,
            output_token_value_fp8,
            None,
            cache_modifier=".cg",
        )
        in_2d_ptr += BLOCK_D
        out_2d_ptr += BLOCK_D

