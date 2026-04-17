# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_unpad_draft_extend_output_kernel.py
# Main kernel: unpad_draft_extend_output_kernel
# PT file: test_unpad_draft_extend_output_kernel_v2.pt

import triton
import triton.language as tl


# === unpad_draft_extend_output_kernel ===
@triton.jit
def unpad_draft_extend_output_kernel(
    raw_out_ptr,  # Input raw output tensor (batch_size, token_per_batch, tp_q_head_num, v_head_dim)
    output_ptr,  # Output tensor (-1, tp_q_head_num, v_head_dim)
    accept_length_ptr,  # Accept lengths for each sequence [batch_size]
    cumsum_ptr,  # Cumulative sum of accept lengths [batch_size + 1]
    batch_size,
    token_per_batch,
    tp_q_head_num,
    v_head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for unpadding draft extended output tensor with parallelized head and dim processing."""
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    batch_id = batch_seq_pid // token_per_batch
    seq_pos = batch_seq_pid % token_per_batch

    if batch_id >= batch_size:
        return

    # Load accept length for this batch
    accept_len = tl.load(accept_length_ptr + batch_id)

    if seq_pos >= accept_len:
        return

    # Load cumulative sum to get start position in output tensor
    output_start = tl.load(cumsum_ptr + batch_id)
    output_pos = output_start + seq_pos

    # Calculate head and dim block ranges
    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, tp_q_head_num)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, v_head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)

    # Calculate input offset: (batch_id, seq_pos, head_id, dim_id)
    input_offset = (
        batch_id * token_per_batch * tp_q_head_num * v_head_dim
        + seq_pos * tp_q_head_num * v_head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Load data
    data = tl.load(
        raw_out_ptr + input_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    output_offset = (
        output_pos * tp_q_head_num * v_head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Store data
    tl.store(
        output_ptr + output_offset,
        data,
        mask=head_mask[:, None] & dim_mask[None, :],
    )

