# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_pad_draft_extend_query_kernel.py
# Main kernel: pad_draft_extend_query_kernel
# PT file: test_pad_draft_extend_query_kernel_v2.pt

import triton
import triton.language as tl


# === pad_draft_extend_query_kernel ===
@triton.jit
def pad_draft_extend_query_kernel(
    q_ptr,  # Input query tensor [total_seq_len, num_heads, head_dim]
    padded_q_ptr,  # Output padded query tensor [batch_size, max_seq_len, num_heads, head_dim]
    seq_lens_q_ptr,  # Sequence lengths for each sequence [batch_size]
    cumsum_ptr,  # Cumulative sum of accept lengths [batch_size + 1]
    batch_size,
    max_seq_len,
    num_heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for padding draft extended query tensor with parallelized head and dim processing."""
    # Use 3D program IDs: (batch_seq, head_block, dim_block)
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    batch_id = batch_seq_pid // max_seq_len
    seq_pos = batch_seq_pid % max_seq_len

    if batch_id >= batch_size:
        return

    # Load accept length for this batch
    seq_len = tl.load(seq_lens_q_ptr + batch_id)

    if seq_pos >= seq_len:
        return

    # Load cumulative sum to get start position in input tensor
    input_start = tl.load(cumsum_ptr + batch_id)
    input_pos = input_start + seq_pos

    # Calculate head and dim block ranges
    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, num_heads)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)

    # Calculate input offset
    input_offset = (
        input_pos * num_heads * head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Load data
    data = tl.load(
        q_ptr + input_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    # Calculate output offset
    output_offset = (
        batch_id * max_seq_len * num_heads * head_dim
        + seq_pos * num_heads * head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Store data
    tl.store(
        padded_q_ptr + output_offset,
        data,
        mask=head_mask[:, None] & dim_mask[None, :],
    )

