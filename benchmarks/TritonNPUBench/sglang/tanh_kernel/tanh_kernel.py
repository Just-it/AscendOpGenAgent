# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_tanh.py
# Main kernel: tanh_kernel
# PT file: test_tanh_kernel_v2.pt

import triton
import triton.language as tl


# === tanh_kernel ===
@triton.jit
def tanh_kernel(
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
    y = 2 * tl.sigmoid(2 * x) - 1
    tl.store(y_ptr + offsets, y, mask=mask)

