# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/vllm_operator_cases/newtest_cases/test__num_nans_kernel.py
# Main kernel: _num_nans_kernel
# PT file: None

import triton
import triton.language as tl
from triton.language.extra.cann import libdevice

# === _num_nans_kernel ===
@triton.jit
def _num_nans_kernel(
    logits_ptr,
    logits_stride,
    num_nans_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_nans = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_stride + block, mask=mask, other=0
        )
        logits = logits.to(tl.float32)
        is_nan = libdevice.isnan(logits).to(tl.int32)
        num_nans += tl.sum(is_nan)
    tl.store(num_nans_ptr + req_idx, num_nans)
    tl.device_print('num_nans = ', num_nans)

