# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test__silu_and_mul_post_per_tensor_quant_kernel.py
# Main kernel: _silu_and_mul_post_per_tensor_quant_kernel
# PT file: test__silu_and_mul_post_per_tensor_quant_kernel_v2.pt

import triton
import triton.language as tl


# === _silu_and_mul_post_per_tensor_quant_kernel ===
@triton.jit
def _silu_and_mul_post_per_tensor_quant_kernel(
    input_ptr,
    stride_input_expert,
    stride_input_token,
    stride_input_dim,
    output_ptr,
    stride_output_expert,
    stride_output_token,
    stride_output_dim,
    scale_ptr,
    masked_m_ptr,
    inner_dim,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    """
    Triton kernel: fused SiLU(gate) * up + per-tensor FP8 quantization.

    Shape:
        input:  [E, T_padded, 2*D]  -> gate: [:,:,D], up: [:,:,D]
        output: [E, T_padded, D], dtype=float8_e4m3fn
    """
    expert_id = tl.program_id(2)
    block_id_token = tl.program_id(1)
    block_id_dim = tl.program_id(0)

    num_token_blocks = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    scale = 1.0 / tl.load(scale_ptr).to(tl.float32)

    stride_input_expert = tl.cast(stride_input_expert, tl.int32)
    stride_output_expert = tl.cast(stride_output_expert, tl.int32)
    stride_input_token = tl.cast(stride_input_token, tl.int32)
    stride_output_token = tl.cast(stride_output_token, tl.int32)

    offset_d = block_id_dim * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_d = offset_d < inner_dim

    # base pointers for current expert and dim block
    input_base_offs = input_ptr + expert_id * stride_input_expert + offset_d
    output_base_offs = output_ptr + expert_id * stride_output_expert + offset_d

    for token_idx in tl.range(
        block_id_token, token_num_cur_expert, num_token_blocks, num_stages=NUM_STAGE
    ):
        gate_ptr = input_base_offs + token_idx * stride_input_token
        up_ptr = gate_ptr + inner_dim
        gate = tl.load(gate_ptr, mask=mask_d, other=0.0).to(tl.float32)
        up = tl.load(up_ptr, mask=mask_d, other=0.0).to(tl.float32)

        # SiLU: x * sigmoid(x)
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate

        scaled = gate_up * scale
        output_q = tl.clamp(scaled, fp8_min, fp8_max).to(output_ptr.dtype.element_ty)
        out_ptr = output_base_offs + token_idx * stride_output_token
        tl.store(out_ptr, output_q, mask=mask_d)

