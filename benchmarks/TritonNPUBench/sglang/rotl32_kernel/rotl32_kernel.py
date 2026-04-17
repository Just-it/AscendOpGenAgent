# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test__rotl32.py
# Main kernel: rotl32_kernel
# PT file: test_rotl32_kernel_v2.pt

import triton
import triton.language as tl

@triton.jit
def _rotl32(x, r: tl.constexpr):
    return (x << r) | (x >> (32 - r))

# === rotl32_kernel ===
@triton.jit
def rotl32_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    r: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = _rotl32(x, r)
    tl.store(y_ptr + offsets, y, mask=mask)

