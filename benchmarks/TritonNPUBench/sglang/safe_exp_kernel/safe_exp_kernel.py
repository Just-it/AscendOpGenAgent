# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_safe_exp.py
# Main kernel: safe_exp_kernel
# PT file: test_safe_exp_kernel_v2.pt

import triton
import triton.language as tl


# === safe_exp_kernel ===
@triton.jit
def safe_exp_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.exp(tl.where(x <= 0, x, float("-inf")))
    tl.store(y_ptr + offsets, y, mask=mask)

