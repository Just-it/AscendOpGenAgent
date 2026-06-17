# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test__fbgemm_silu_mul_npu.py
# Main kernel: _fbgemm_silu_mul
# PT file: test__fbgemm_silu_mul_v2.pt

import triton
import triton.language as tl


# === _fbgemm_silu_mul ===
@triton.jit
def _fbgemm_silu_mul(
    y_ptr,
    x0_ptr,
    x1_ptr,
    stride_0,
    stride_1,
    valid_token_count,
    D: tl.constexpr,
    BLOCK_D_OUTER: tl.constexpr,
    BLOCK_D_INNER: tl.constexpr,
) -> None:
    token_index = tl.program_id(0)
    feature_offset = tl.program_id(1) * BLOCK_D_OUTER + tl.arange(0, BLOCK_D_INNER)[:]

    if valid_token_count is not None:
        valid_token_count = tl.load(
            valid_token_count, None, eviction_policy="evict_last"
        )
        if token_index >= valid_token_count:
            return

    for _ in tl.range(0, BLOCK_D_OUTER // BLOCK_D_INNER, num_stages=3):
        x0 = tl.load(
            x0_ptr + token_index * stride_0 + feature_offset,
            None,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x1 = tl.load(
            x1_ptr + token_index * stride_1 + feature_offset,
            None,
            eviction_policy="evict_first",
        ).to(tl.float32)

        y = x0 * tl.sigmoid(x0) * x1

        tl.store(
            y_ptr + token_index * D + feature_offset,
            y,
            None,
        )
        feature_offset += BLOCK_D_INNER

