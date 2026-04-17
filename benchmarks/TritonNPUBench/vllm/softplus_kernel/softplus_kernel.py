# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/vllm_operator_cases/newtest_cases/test_softplus.py
# Main kernel: softplus_kernel
# PT file: softplus_v2.pt

import triton
import triton.language as tl

@triton.jit
def softplus(dt):
    dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
    return dt
    

# === softplus_kernel ===
@triton.jit
def softplus_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_block = tl.load(input_ptr + offsets, mask=mask)
    output_block = softplus(input_block)
    tl.store(output_ptr + offsets, output_block, mask=mask)

