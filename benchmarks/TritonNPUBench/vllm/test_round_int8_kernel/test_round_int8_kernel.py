# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/vllm_operator_cases/newtest_cases/test_round_int8.py
# Main kernel: test_round_int8_kernel
# PT file: round_int8.pt

import triton
import triton.language as tl

@triton.jit
def round_int8(x):
    return tl.extra.cann.libdevice.round(x).to(tl.int8)

@triton.jit
def test_round_int8_kernel(
    x_ptr,        # 输入数据指针
    output_ptr,   # 输出数据指针
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = round_int8(x)
    tl.store(output_ptr + offsets, result, mask=mask)

