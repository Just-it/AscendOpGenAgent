# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test__fbgemm_silu_mul_quant_npu.py
# Main kernel: _fbgemm_silu_mul_quant
# PT file: test__fbgemm_silu_mul_quant_v2.pt

import triton
import triton.language as tl


# === _fbgemm_silu_mul_quant ===
@triton.jit
def _fbgemm_silu_mul_quant(
    y_ptr,
    y_inv_scale_ptr,
    x0_ptr,
    x1_ptr,
    scale_ub_ptr,
    stride_0,
    stride_1,
    valid_token_count,
    T,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
) -> None:
    PADDED_D: tl.constexpr = triton.next_power_of_2(D)  # pyre-ignore

    tidx = tl.program_id(0)
    start_idx = tidx * BLOCK_T
    end_idx = tl.minimum(start_idx + BLOCK_T, T)

    if valid_token_count is not None:
        valid_token_count = tl.load(
            valid_token_count, None, eviction_policy="evict_last"
        )
        if start_idx >= valid_token_count:
            return

    offsets = tl.arange(0, PADDED_D)[:]
    mask = offsets < D

    if CLAMP_MAX:
        ub = tl.load(scale_ub_ptr, eviction_policy="evict_last")
    else:
        ub = float("inf")

    for token_index in tl.range(start_idx, end_idx, 1, num_stages=2):
        x0 = tl.load(
            x0_ptr + token_index * stride_0 + offsets,
            mask,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x1 = tl.load(
            x1_ptr + token_index * stride_1 + offsets,
            mask,
            eviction_policy="evict_first",
        ).to(tl.float32)

        y = x0 * tl.sigmoid(x0) * x1

        # Masked values are set to 0.0.
        row_max = tl.max(tl.where(mask, tl.abs(y), 0.0))
        if CLAMP_MAX:
            row_max = tl.clamp(row_max, EPS, ub)
        else:
            row_max = tl.maximum(row_max, EPS)

        y_scale = MAX_FP8 / row_max
        tl.store(y_inv_scale_ptr + token_index, 1.0 / y_scale)

        y = y * y_scale
        # Clamp A to fp8 range to make sure there's no overflow.
        # This is required for AMD. Nvidia's default saturation
        # handles it, but it's nice to have anyway.
        y_fp8 = tl.clamp(y, -MAX_FP8, MAX_FP8).to(TL_FP8_DTYPE)

        tl.store(
            y_ptr + token_index * D + offsets,
            y_fp8,
            mask,
        )

