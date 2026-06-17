# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_fused_moe_router_tensorcore_kernel.py
# Main kernel: fused_moe_router_tensorcore_kernel
# PT file: test_fused_moe_router_tensorcore_kernel_v2.pt

import triton
import triton.language as tl


# === fused_moe_router_tensorcore_kernel ===
@triton.jit
def fused_moe_router_tensorcore_kernel(
    a_ptr,  # input (bs, hidden_dim)
    b_ptr,  # input (num_experts, hidden_dim)
    topk_weights_ptr,  # output (bs, topk)
    topk_ids_ptr,  # output (bs, topk)
    bs,
    num_experts: tl.constexpr,
    topk: tl.constexpr,  # only support topk <= 2
    moe_softcapping: tl.constexpr,
    moe_renormalize: tl.constexpr,  # not supported
    correction_bias_ptr,
    is_correction_bias: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bn: tl.constexpr,
    dp_attn_workaround_flag: tl.constexpr,
):

    # 1. get block id
    pid = tl.program_id(axis=0)

    # 2. create pointers for the first block of A and B
    # 2.1. setup a_ptrs with offsets in m and k
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    bs_mask = offs_m < bs
    offs_k = tl.arange(0, BLOCK_SIZE_K)[None, :]
    a_ptrs = a_ptr + (offs_m * stride_am + offs_k)

    # 2.2. setup b_ptrs with offsets in k and n.
    #      Note: b matrix is k-major.
    offs_k = tl.arange(0, BLOCK_SIZE_K)[None, :]
    offs_n = tl.arange(0, BLOCK_SIZE_N)[:, None]
    expert_mask = offs_n < num_experts
    b_ptrs = b_ptr + (offs_n * stride_bn + offs_k)

    # 3. Create an accumulator of float32 of size [BLOCK_SIZE_M, BLOCK_SIZE_N]
    #    3.1. iterate in K dimension
    #    3.2. transpose tile B
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K // BLOCK_SIZE_K):  # hidden_dim % BLOCK_SIZE_K == 0
        a = tl.load(
            a_ptrs,
            mask=bs_mask,
            other=0.0,
        ).to(tl.float32)
        b = tl.load(b_ptrs, mask=expert_mask, other=0.0).to(tl.float32).T
        acc += tl.dot(a, b)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    # 4. logit softcap
    if moe_softcapping == 0:
        logits_softcapped = acc
    else:
        logits_scaled = acc / moe_softcapping
        exped = tl.exp(2 * logits_scaled)
        logits_softcapped = (exped - 1) / (exped + 1) * moe_softcapping

    # Add bias after softcapping
    if is_correction_bias:
        bias = tl.load(
            correction_bias_ptr + tl.arange(0, BLOCK_SIZE_N)[None, :],
            mask=expert_mask.T,
            other=0.0,
        )
        logits_softcapped = logits_softcapped + bias

    if dp_attn_workaround_flag:
        logits_softcapped = tl.where(
            logits_softcapped != logits_softcapped, -1e9, logits_softcapped
        )

    # 5. top1
    arange_block_size_n = tl.arange(0, BLOCK_SIZE_N)[None, :]
    cond_top1 = arange_block_size_n < num_experts
    top1 = tl.argmax(tl.where(cond_top1, logits_softcapped, float("-inf")), axis=1)
    top1_v = tl.max(
        tl.where(cond_top1, logits_softcapped, float("-inf")), axis=1, keep_dims=True
    )
    top1_invsumexp = 1.0 / tl.sum(
        tl.where(cond_top1, tl.exp(logits_softcapped - top1_v), 0.0), axis=1
    )

    # 6. store top1 to output
    offs_top1 = pid * topk * BLOCK_SIZE_M + topk * tl.arange(0, BLOCK_SIZE_M)
    top1_mask = offs_top1 < bs * topk
    tl.store(topk_ids_ptr + offs_top1, top1, mask=top1_mask)
    tl.store(
        topk_weights_ptr + offs_top1,
        top1_invsumexp,
        mask=top1_mask,
    )

    # 7. handle topk == 2
    if topk == 2:
        cond_top2 = (arange_block_size_n < num_experts) & (
            arange_block_size_n != top1[:, None]
        )
        top2 = tl.argmax(
            tl.where(cond_top2, logits_softcapped, float("-inf")),
            axis=1,
            keep_dims=True,
        )
        top2_v = tl.sum(
            logits_softcapped * (arange_block_size_n == top2), axis=1, keep_dims=True
        )
        top2_invsumexp = tl.exp(top2_v - top1_v) * top1_invsumexp[:, None]

        # store top2
        offs_top2 = (
            pid * topk * BLOCK_SIZE_M + topk * tl.arange(0, BLOCK_SIZE_M)[:, None] + 1
        )
        top2_mask = offs_top2 < bs * topk
        tl.store(topk_ids_ptr + offs_top2, top2, mask=top2_mask)
        tl.store(
            topk_weights_ptr + offs_top2,
            top2_invsumexp,
            mask=top2_mask,
        )

