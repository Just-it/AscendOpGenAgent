# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_fn_triton_kernel.py
# Main kernel: fn_triton_kernel
# PT file: test_fn_triton_kernel_v2.pt

import triton
import triton.language as tl


# === fn_triton_kernel ===
@triton.jit
def fn_triton_kernel(
    k_ptr,
    k_nope_ptr,
    k_rope_ptr,
    num_tokens,
    QK_NOPE_HEAD_DIM: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    NUM_LOCAL_HEADS: tl.constexpr,
    K_NOPE_STRIDE_0: tl.constexpr,
    K_NOPE_STRIDE_1: tl.constexpr,
    K_STRIDE_0: tl.constexpr,
    K_STRIDE_1: tl.constexpr,
    K_ROPE_STRIDE_0: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    token_id = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    token_mask = token_id < num_tokens

    head_id = tl.arange(0, NUM_LOCAL_HEADS)

    # nope
    nope_sub_id = tl.arange(0, QK_NOPE_HEAD_DIM)
    offs_nope = (
        token_id[:, None, None] * K_NOPE_STRIDE_0
        + head_id[None, :, None] * K_NOPE_STRIDE_1
        + nope_sub_id[None, None, :]
    )
    offs_k = (
        token_id[:, None, None] * K_STRIDE_0
        + head_id[None, :, None] * K_STRIDE_1
        + nope_sub_id[None, None, :]
    )
    vals_nope = tl.load(k_nope_ptr + offs_nope, mask=token_mask[:, None, None])
    tl.store(k_ptr + offs_k, vals_nope, mask=token_mask[:, None, None])

    # rope
    rope_sub_id = tl.arange(0, QK_ROPE_HEAD_DIM)
    offs_rope = token_id[:, None, None] * K_ROPE_STRIDE_0 + rope_sub_id[None, None, :]
    offs_k = (
        token_id[:, None, None] * K_STRIDE_0
        + head_id[None, :, None] * K_STRIDE_1
        + rope_sub_id[None, None, :]
        + QK_NOPE_HEAD_DIM
    )
    vals_rope = tl.load(k_rope_ptr + offs_rope, mask=token_mask[:, None, None])
    tl.store(k_ptr + offs_k, vals_rope, mask=token_mask[:, None, None])

