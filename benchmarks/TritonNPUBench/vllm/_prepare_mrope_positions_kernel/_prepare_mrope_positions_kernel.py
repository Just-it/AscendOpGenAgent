# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/vllm_operator_cases/newtest_cases/test__prepare_mrope_positions_kernel.py
# Main kernel: _prepare_mrope_positions_kernel
# PT file: prepare_mrope_positions_test_v2.pt

import triton
import triton.language as tl


# === _prepare_mrope_positions_kernel ===
@triton.jit
def _prepare_mrope_positions_kernel(
    mrope_positions_ptr,
    mrope_positions_stride,
    prefill_mrope_positions_ptr,
    prefill_mrope_positions_stride0,
    prefill_mrope_positions_stride1,
    prefill_mrope_delta_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    is_prefill = num_computed < prefill_len

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    mrope_delta = tl.load(prefill_mrope_delta_ptr + req_state_idx)
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        orig_pos = num_computed + block

        for j in tl.static_range(3):
            if is_prefill:
                # Read from pre-computed M-RoPE positions.
                pos = tl.load(
                    prefill_mrope_positions_ptr
                    + req_state_idx * prefill_mrope_positions_stride0
                    + j * prefill_mrope_positions_stride1
                    + orig_pos,
                    mask=mask,
                )
            else:
                # Apply M-RoPE delta.
                pos = orig_pos + mrope_delta
            tl.store(
                mrope_positions_ptr + j * mrope_positions_stride + query_start + block,
                pos,
                mask=mask,
            )

