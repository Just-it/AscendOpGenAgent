# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/vllm_operator_cases/newtest_cases/test__fwd_diag_kernel.py
# Main kernel: _fwd_diag_kernel
# PT file: _fwd_diag_kernel_v2.pt

import triton
import triton.language as tl


# === _fwd_diag_kernel ===
@triton.jit
def _fwd_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    CBLOCK: tl.constexpr,
):
    # This kernel computes the diagonal blocks of the attention matrix
    # Each diagonal block represents attention
    # where queries attend to keys in the same block
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK  # batch-head index
    off_block = off % NUM_BLOCK  # block index within the sequence
    off_cblock = tl.program_id(1)  # sub-block index within a block

    off_h = off_bh % h  # head index

    # Calculate base offsets for the current batch and head
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    # Calculate offsets for the current block
    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    # Calculate offsets for the current sub-block
    cblock_offset = off_cblock * CBLOCK
    q_cblock_offset = cblock_offset * d
    o_cblock_offset = cblock_offset * e

    # Calculate pointers to the query, key, value, and output tensors
    Q_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + q_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    K_trans_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    # Load the decay rate for the current head
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    i = off_cblock
    q_index = tl.arange(0, CBLOCK) + i * CBLOCK

    # Load query values
    q = tl.load(Q_block_ptr, mask=block_offset + q_index[:, None] < n, other=0.0).to(
        tl.float32
    )

    # Initialize output accumulator
    qkv = tl.zeros([CBLOCK, e], dtype=tl.float32)

    # Process all sub-blocks up to and
    # including the current one (causal attention)
    for j in range(i + 1):
        kv_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = q_index[:, None] - kv_index[None, :]
        s_index = s * diff
        # Apply causal mask: only attend to positions before the current one
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        decay = tl.exp(s_index)

        # Load key and value
        k_trans = tl.load(
            K_trans_block_ptr,
            mask=block_offset + kv_index[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr,
            mask=block_offset + kv_index[:, None] < n,
            other=0.0,
        ).to(tl.float32)

        # Compute attention scores and apply decay
        qk = tl.dot(q, k_trans) * decay

        # Compute weighted values and accumulate
        qkv += tl.dot(qk, v)

        # Move to the next sub-block
        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    # Store the result
    tl.store(
        O_block_ptr,
        qkv.to(O_block_ptr.dtype.element_ty),
        mask=block_offset + q_index[:, None] < n,
    )

