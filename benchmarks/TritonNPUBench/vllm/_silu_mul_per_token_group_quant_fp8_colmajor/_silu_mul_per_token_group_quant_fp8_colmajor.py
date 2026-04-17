# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/vllm_operator_cases/newtest_cases/test__silu_mul_per_token_group_quant_fp8_colmajor.py
# Main kernel: _silu_mul_per_token_group_quant_fp8_colmajor
# PT file: _silu_mul_per_token_group_quant_fp8_colmajor_v2.pt

import triton
import triton.language as tl


# === _silu_mul_per_token_group_quant_fp8_colmajor ===
@triton.jit
def _silu_mul_per_token_group_quant_fp8_colmajor(
    y_ptr,  # [M, N]
    y_q_ptr,  # [M, N // 2]
    y_s_ptr,  # [M, (N // 2) // GROUP_SIZE]
    M,  # num tokens
    N,  # intermediate size
    # Stride
    y_s_col_stride: tl.int64,
    # Information for float8
    eps,
    fp8_min,
    fp8_max,
    use_ue8m0: tl.constexpr,
    # Meta-parameters
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # TODO(varun) : Add expert_ids so we may early-exit no-op thread blocks.
    """
    Each thread block (BLOCK_N) computes [BLOCK_M, GROUP_SIZE] act-mul outputs. Then
    the thread block quantizes the [BLOCK_M, GROUP_SIZE] block of values and fills
    the outputs tensors at the right positions.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    N_2 = N // 2

    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N
    if m_offset >= M:
        return

    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)

    base_y_ptr = y_ptr + m_offset * N + n_offset

    act_in_ptrs = base_y_ptr + offs_m[:, None] * N + offs_n[None, :]

    act_in = tl.load(act_in_ptrs)
    mul_in = tl.load(act_in_ptrs + N_2)

    # silu & mul
    act_in = act_in.to(tl.float32)
    one_f32 = tl.cast(1, tl.float32)
    silu_out = (act_in / (one_f32 + tl.exp(-act_in))).to(y_ptr.dtype.element_ty)
    y = (silu_out * mul_in).to(tl.float32)

    # quant
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_s = tl.reshape(y_s, (BLOCK_M, 1))
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    # store y_q
    base_y_q_ptr = y_q_ptr + m_offset * N_2 + n_offset
    y_q_ptrs = base_y_q_ptr + offs_m[:, None] * N_2 + offs_n[None, :]
    tl.store(y_q_ptrs, y_q)

    # store y_s
    group_id = n_offset // GROUP_SIZE
    base_y_s_ptr = y_s_ptr + group_id * y_s_col_stride + m_offset
    y_s_ptrs = base_y_s_ptr + offs_m
    y_s = tl.reshape(y_s, (BLOCK_M,))
    tl.store(y_s_ptrs, y_s)

