# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_experts_combine_kernel.py
# Main kernel: experts_combine_kernel
# PT file: test_experts_combine_kernel_v2.pt

import triton
import triton.language as tl


# === experts_combine_kernel ===
@triton.jit
def experts_combine_kernel(
    out_hidden_states,
    moe_hidden_states,
    mlp_hidden_states,
    combine_k: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_index_mlp = pid * hidden_dim
    start_index_rmoe = pid * hidden_dim * combine_k
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    combine_k_offsets = tl.arange(0, combine_k)

    moe_x = tl.load(
        moe_hidden_states
        + start_index_rmoe
        + combine_k_offsets[:, None] * hidden_dim
        + offsets[None, :],
        mask=mask[None, :],
        other=0.0,
    )
    moe_x = tl.sum(moe_x, axis=0)
    mlp_x = tl.load(mlp_hidden_states + start_index_mlp + offsets, mask=mask, other=0.0)
    combined_x = (moe_x + mlp_x) / 1.4142135623730951

    tl.store(out_hidden_states + start_index_mlp + offsets, combined_x, mask=mask)

