# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/vllm_operator_cases/newtest_cases/test__fwd_kernel_ep_scatter_1.py
# Main kernel: _fwd_kernel_ep_scatter_1
# PT file: _fwd_kernel_ep_scatter_1_v2.pt

import triton
import triton.language as tl


# === _fwd_kernel_ep_scatter_1 ===
@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)

    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    tokens_per_expert = round_up_128(tokens_per_expert)
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)

    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)

    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    # any rows in the per-expert aligned region that do not correspond to
    # real tokens are left untouched here and should remain initialized to
    # -1 so DeepGEMM can skip them
    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        offs = start_m + off_expert
        mask = offs < cur_expert_token_num
        tl.store(
            m_indices_start_ptr + offs,
            cur_expert,
            mask=mask,
        )


# === round_up_128 ===
@triton.jit
def round_up_128(x: int) -> int:
    y = 128
    return ((x + y - 1) // y) * y

