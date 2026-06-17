# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test__jagged_flash_attention_bwd_preprocess_basic_kernel_npu.py
# Main kernel: _jagged_flash_attention_bwd_preprocess_basic_kernel
# PT file: _jagged_flash_attention_bwd_preprocess_basic_kernel_v2.pt

import triton
import triton.language as tl


# === _jagged_flash_attention_bwd_preprocess_basic_kernel ===
@triton.jit
def _jagged_flash_attention_bwd_preprocess_basic_kernel(
    o_ptr,
    o_offset_ptr,
    do_ptr,
    delta_ptr,
    stride_om,
    stride_od,
    max_seq_len,
    D: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)

    begin_o = tl.load(o_offset_ptr + pid_batch)
    end_o = tl.load(o_offset_ptr + pid_batch + 1)

    M = end_o - begin_o
    M = tl.minimum(M, max_seq_len)

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_od = tl.arange(0, BLOCK_SIZE_D)

    o_offsets = (
        offs_om[:, None] * stride_om
        + offs_od[None, :] * stride_od
        + begin_o * stride_om
    )
    o_ptrs = o_ptr + o_offsets
    do_ptrs = do_ptr + o_offsets
    o_mask = (offs_om[:, None] < M) & (offs_od[None, :] < D)

    # Load O
    o = tl.load(o_ptrs, mask=o_mask)
    do = tl.load(do_ptrs, mask=o_mask)

    delta = tl.sum(o * do, axis=1)

    tl.store(delta_ptr + begin_o + offs_om, delta, mask=offs_om < M)

