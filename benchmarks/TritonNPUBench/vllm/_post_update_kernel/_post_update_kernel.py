# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/vllm_operator_cases/newtest_cases/test__post_update_kernel.py
# Main kernel: _post_update_kernel
# PT file: post_update_kernel_test_data.pt

import triton
import triton.language as tl


# === _post_update_kernel ===
@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
):
    req_id = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_id)

    num_sampled = tl.load(num_sampled_ptr + req_id)
    if num_sampled > 0:
        token_id = tl.load(
            sampled_tokens_ptr + req_id * sampled_tokens_stride + num_sampled - 1
        )
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)

    for i in range(num_sampled):
        token_id = tl.load(sampled_tokens_ptr + req_id * sampled_tokens_stride + i)
        token_ptr = (
            output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + token_id
        )
        count = tl.load(token_ptr)
        count += 1
        tl.store(token_ptr, count)

    query_start = tl.load(query_start_loc_ptr + req_id)
    query_end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = query_end - query_start
    num_rejected = tl.load(num_rejected_ptr + req_id)

    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    num_computed += query_len - num_rejected
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed)

