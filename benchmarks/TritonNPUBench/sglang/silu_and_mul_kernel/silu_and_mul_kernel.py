# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_silu_and_mul_kernel.py
# Main kernel: silu_and_mul_kernel
# PT file: test_silu_and_mul_kernel_v2.pt

import triton
import triton.language as tl


# === silu_and_mul_kernel ===
@triton.jit
def silu_and_mul_kernel(
    out_hidden_states_ptr,  # (bs, hidden_dim)
    out_scales_ptr,  # (bs,)
    hidden_states_ptr,  # (bs, hidden_dim * 2)
    quant_max: tl.constexpr,
    static_scale: tl.constexpr,
    hidden_dim: tl.constexpr,  # the output hidden_dim
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    input_start = pid * hidden_dim * 2
    output_start = pid * hidden_dim

    input1_offs = tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < hidden_dim  # shared for input1, input3, output
    input3_offs = hidden_dim + tl.arange(0, BLOCK_SIZE)
    output_offs = tl.arange(0, BLOCK_SIZE)

    x1 = tl.load(
        hidden_states_ptr + input_start + input1_offs, mask=mask, other=0.0
    ).to(tl.float32)
    x3 = tl.load(
        hidden_states_ptr + input_start + input3_offs, mask=mask, other=0.0
    ).to(tl.float32)

    # silu
    # cast down before mul to better match training?
    silu_x1 = x1 * tl.sigmoid(x1)
    out = x3 * silu_x1.to(hidden_states_ptr.dtype.element_ty)

    if quant_max is not None:
        raise NotImplementedError()

    tl.store(out_hidden_states_ptr + output_start + output_offs, out, mask=mask)

