# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe.py
# Main kernel: silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe
# PT file: test_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe_v2.pt

import triton
import triton.language as tl


# === silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe ===
@triton.jit
def silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe(
    input_ptr,
    output_ptr,
    scale_ptr,
    num_tokens_tensor_ptr,
    intermediate_size,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    OutDtype = output_ptr.dtype.element_ty

    num_tokens = tl.load(num_tokens_tensor_ptr)
    numel = num_tokens * intermediate_size
    gate_ptr = input_ptr
    up_ptr = input_ptr + intermediate_size
    scale = 1.0 / tl.load(scale_ptr)

    start_idx = tl.program_id(0) * BLOCK_SIZE
    step = tl.num_programs(0) * BLOCK_SIZE

    for id in tl.range(start_idx, numel, step, num_stages=NUM_STAGES):
        ids = id + tl.arange(0, BLOCK_SIZE)
        token_ids = ids // intermediate_size
        mask = ids < numel

        offs = ids + token_ids * intermediate_size
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        output = gate / (1 + tl.exp(-gate)) * up * scale
        tl.store(output_ptr + ids, output.to(OutDtype), mask=mask)

