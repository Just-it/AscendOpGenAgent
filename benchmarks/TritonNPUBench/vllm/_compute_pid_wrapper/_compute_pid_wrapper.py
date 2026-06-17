# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/vllm_operator_cases/newtest_cases/test__compute_pid.py
# Main kernel: _compute_pid_wrapper
# PT file: _compute_pid_wrapper_v2.pt

import triton
import triton.language as tl

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

# === _compute_pid_wrapper ===
@triton.jit
def _compute_pid_wrapper(
    tile_id_ptr,
    num_pid_in_group_ptr,
    num_pid_m_ptr,
    GROUP_SIZE_M_ptr,
    NUM_SMS_ptr,
    output_ptr,
):
    if tl.program_id(0) == 0:
        tile_id = tl.load(tile_id_ptr)
        num_pid_in_group = tl.load(num_pid_in_group_ptr)
        num_pid_m = tl.load(num_pid_m_ptr)
        GROUP_SIZE_M = tl.load(GROUP_SIZE_M_ptr)
        NUM_SMS = tl.load(NUM_SMS_ptr)

        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        tl.store(output_ptr + 0, pid_m)
        tl.store(output_ptr + 1, pid_n)

