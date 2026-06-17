# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_add_tree_reduce_u64_kernel.py
# Main kernel: add_tree_reduce_u64_kernel
# PT file: test_add_tree_reduce_u64_kernel_v2.pt

import triton
import triton.language as tl


# === add_tree_reduce_u64_kernel ===
@triton.jit
def add_tree_reduce_u64_kernel(in_ptr, out_ptr, n_elems, CHUNK: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * CHUNK
    h = tl.zeros((), dtype=tl.int64)
    for i in tl.static_range(0, CHUNK):
        idx = start + i
        m = idx < n_elems
        v = tl.load(in_ptr + idx, mask=m, other=0).to(tl.int64)
        h += v
    tl.store(out_ptr + pid, h)

