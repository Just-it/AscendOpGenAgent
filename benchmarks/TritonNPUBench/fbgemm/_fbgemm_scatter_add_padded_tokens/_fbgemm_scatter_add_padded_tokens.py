# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test__fbgemm_scatter_add_padded_tokens_npu.py
# Main kernel: _fbgemm_scatter_add_padded_tokens
# PT file: test__fbgemm_scatter_add_padded_tokens_v2.pt

import triton
import triton.language as tl


# === _fbgemm_scatter_add_padded_tokens ===
@triton.jit
def _fbgemm_scatter_add_padded_tokens(
    in_tokens_ptr,
    token_counts_ptr,
    token_indices_ptr,
    out_tokens_ptr,
    EP: tl.constexpr,
    E: tl.constexpr,
    T_BUCKET,
    T_K,
    D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    SPLIT_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    in_tokens: [EP, T_K, D]
    token_counts: [E]
    out_tokens: [T, D]
    """
    expert = tl.program_id(0)
    t_tile = tl.program_id(1)

    tl.static_assert(D % BLOCK_D == 0)
    NUM_D_BLOCKS: tl.constexpr = D // BLOCK_D

    num_tokens = tl.load(token_counts_ptr + expert)
    if num_tokens == 0:
        return

    num_tokens_per_cta = tl.cdiv(num_tokens, SPLIT_T)
    start_token = t_tile * num_tokens_per_cta
    end_token = min(start_token + num_tokens_per_cta, num_tokens)

    tl.static_assert(E % EP == 0)
    EXPERT_PER_RANK: tl.constexpr = E // EP
    rank = expert // EXPERT_PER_RANK

    offs_e = tl.arange(0, BLOCK_E)
    token_counts = tl.load(token_counts_ptr + offs_e, mask=(offs_e < E), other=0)
    input_local_offset = (
        tl.sum(tl.where(offs_e < expert, token_counts, 0)) + start_token
    ).to(tl.int64)

    for _t in range(start_token, end_token):
        output_local_offset = tl.load(token_indices_ptr + input_local_offset).to(
            tl.int64
        )
        output_global_offset = output_local_offset * D

        d_ptr = tl.arange(0, BLOCK_D)
        input_global_ptr = (
            in_tokens_ptr + rank * T_K * D + input_local_offset * D + d_ptr
        )
        output_global_ptr = out_tokens_ptr + output_global_offset + d_ptr

        for _d in range(NUM_D_BLOCKS):
            vec = tl.load(input_global_ptr)
            tl.atomic_add(output_global_ptr, vec, sem="relaxed")
            input_global_ptr += BLOCK_D
            output_global_ptr += BLOCK_D

        input_local_offset += 1

