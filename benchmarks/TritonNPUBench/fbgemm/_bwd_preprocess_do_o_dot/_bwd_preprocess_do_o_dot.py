# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test__bwd_preprocess_do_o_dot_npu.py
# Main kernel: _bwd_preprocess_do_o_dot
# PT file: _bwd_preprocess_do_o_dot_v2.pt

import triton
import triton.language as tl


# === _bwd_preprocess_do_o_dot ===
@triton.jit
def _bwd_preprocess_do_o_dot(
    o_ptr,
    do_ptr,
    delta_ptr,
    T,
    stride_ob,
    stride_ot,
    stride_od,
    stride_do_b,
    stride_do_t,
    stride_do_d,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_t = tl.program_id(0)
    offs_t = start_t * BLOCK_T + tl.arange(0, BLOCK_T)
    pid_b = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)

    o_ptrs = (
        o_ptr
        + pid_b * stride_ob
        + offs_t[:, None] * stride_ot
        + offs_d[None, :] * stride_od
    )
    do_ptrs = (
        do_ptr
        + pid_b * stride_do_b
        + offs_t[:, None] * stride_do_t
        + offs_d[None, :] * stride_do_d
    )
    o = tl.load(o_ptrs, mask=(offs_t[:, None] < T), other=0.0)
    do = tl.load(do_ptrs, mask=(offs_t[:, None] < T), other=0.0)
    delta = tl.sum(o * do, axis=1)

    delta_ptrs = delta_ptr + pid_b * T + offs_t
    tl.store(delta_ptrs, delta, mask=(offs_t < T))

