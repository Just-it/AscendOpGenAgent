# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test__embedding_lora_a_kernel.py
# Main kernel: _embedding_lora_a_kernel
# PT file: test__embedding_lora_a_kernel_v2.pt

import triton
import triton.language as tl


# === _embedding_lora_a_kernel ===
@triton.jit
def _embedding_lora_a_kernel(
    # Pointers to tensors
    input_ids,
    weights,
    output,
    extra_embeddings,
    # Dimensions
    vocab_size,
    rank,
    num_loras,
    # Strides
    w_stride_0,  # stride for lora index
    w_stride_1,  # stride for rank
    w_stride_2,  # stride for vocab
    output_stride_0,
    output_stride_1,
    extra_emb_stride_0,  # stride for lora index
    extra_emb_stride_1,  # stride for token
    extra_emb_stride_2,  # stride for hidden dim (= rank for extra embeddings)
    # Batch info
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    # Meta-parameters
    BLOCK_RANK: tl.constexpr,
    HAS_EXTRA_EMBEDDINGS: tl.constexpr,
):
    """
    Embedding lookup for LoRA A weights with support for extra tokens.

    Each program handles one token across a block of rank dimensions.
    Grid: (cdiv(max_len, 1), bs) - one program per token in each batch
    """
    batch_id = tl.program_id(axis=1)
    token_idx = tl.program_id(axis=0)

    w_index = tl.load(weight_indices + batch_id)
    rank_val = tl.load(lora_ranks + w_index)

    # If rank is 0, skip
    if rank_val == 0:
        return

    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)

    # Check if this token is within the segment
    if token_idx >= seg_len:
        return

    # Load the token ID
    token_id = tl.load(input_ids + seg_start + token_idx)

    # Process in chunks of BLOCK_RANK dimensions
    num_blocks = tl.cdiv(rank_val, BLOCK_RANK)

    for block_id in range(num_blocks):
        rank_offset = tl.arange(0, BLOCK_RANK) + block_id * BLOCK_RANK
        rank_mask = rank_offset < rank_val

        # Check if this is an extra token
        is_extra_token = token_id >= vocab_size

        if HAS_EXTRA_EMBEDDINGS and is_extra_token:
            # Use extra embeddings
            extra_token_id = token_id - vocab_size
            extra_emb_ptr = (
                extra_embeddings
                + w_index * extra_emb_stride_0
                + extra_token_id * extra_emb_stride_1
                + rank_offset * extra_emb_stride_2
            )
            emb_values = tl.load(extra_emb_ptr, mask=rank_mask, other=0.0)
        else:
            # Use regular LoRA A weights
            # weights shape: (num_loras, rank, vocab_size)
            # We need to load weights[w_index, rank_offset, token_id]
            token_id_clamped = tl.minimum(token_id, vocab_size - 1)
            weight_ptr = (
                weights
                + w_index * w_stride_0
                + rank_offset * w_stride_1
                + token_id_clamped * w_stride_2
            )
            emb_values = tl.load(weight_ptr, mask=rank_mask, other=0.0)

        # Write to output
        output_ptr = (
            output
            + (seg_start + token_idx) * output_stride_0
            + rank_offset * output_stride_1
        )
        tl.store(output_ptr, emb_values, mask=rank_mask)

