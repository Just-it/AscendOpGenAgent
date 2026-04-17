# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test__multi_head_jagged_flash_attention_bwd_preprocess_kernel_npu.py
# Main kernel: _multi_head_jagged_flash_attention_bwd_preprocess_kernel
# PT file: test__multi_head_jagged_flash_attention_bwd_preprocess_kernel_v2.pt

import triton
import triton.language as tl


# === _multi_head_jagged_flash_attention_bwd_preprocess_kernel ===
@triton.jit
def _multi_head_jagged_flash_attention_bwd_preprocess_kernel(
    o_ptr,
    o_offset_ptr,
    do_ptr,
    delta_ptr,
    stride_oh,
    stride_om,
    stride_od,
    stride_delta_h,
    num_heads: tl.constexpr,
    max_seq_len: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    pid_batch = pid_bh // num_heads
    pid_head = pid_bh % num_heads

    begin_o = tl.load(o_offset_ptr + pid_batch)
    end_o = tl.load(o_offset_ptr + pid_batch + 1)

    M = end_o - begin_o
    M = tl.minimum(M, max_seq_len)

    if M == 0:
        return

    offs_om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_od = tl.arange(0, BLOCK_D)

    o_offsets = (
        offs_om[:, None] * stride_om
        + offs_od[None, :] * stride_od
        + pid_head * stride_oh
        + begin_o * stride_om
    )
    o_ptrs = o_ptr + o_offsets
    do_ptrs = do_ptr + o_offsets
    o_mask = (offs_om[:, None] < M) & (offs_od[None, :] < D)

    # Load o and do
    o = tl.load(o_ptrs, mask=o_mask)
    do = tl.load(do_ptrs, mask=o_mask)

    delta = tl.sum(o * do, axis=1)

    tl.store(
        delta_ptr + pid_head * stride_delta_h + begin_o + offs_om,
        delta,
        mask=offs_om < M,
    )

