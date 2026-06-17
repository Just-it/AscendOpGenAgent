# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test__fwd_grouped_kernel_stage1_rope.py
# Main kernel: _fwd_grouped_kernel_stage1_rope
# PT file: test__fwd_grouped_kernel_stage1_rope_v2.pt

import triton
import triton.language as tl


# === _fwd_grouped_kernel_stage1_rope ===
@triton.jit
def _fwd_grouped_kernel_stage1_rope(
    Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
    K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
    V_buffer,  # Holds [KV], b*s x (c)
    cos_sin_cache,  # max_seq_len x (rotary_dim * 2)
    positions,  # sequence positions
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank + 1)
    k_pe_t_out,
    stride_qb,
    stride_qh,
    stride_buf_kbs,
    stride_buf_vbs,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_kpe_tokens_out_b,
    stride_cos_sin_cache_s,
    stride_positions_b,
    rotary_dim: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    logit_cap: tl.constexpr,
    USE_ROPE: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
):

    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_c = tl.arange(0, BLOCK_C)
    offs_qk_r = tl.arange(kv_lora_rank, kv_lora_rank + BLOCK_R)  # to get the k_pe

    off_q_pe = (
        cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_qk_r[None, :]
    )
    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]

    mask_c = offs_c < kv_lora_rank
    mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx

    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_c[None, :]), other=0.0)
    q_pe = tl.load(
        Q + off_q_pe, mask=(mask_h[:, None]) & (mask_qk_r[None, :]), other=0.0
    )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # apply rotary embedding for q_pe, and k_pe (last token per batch of K_PE)
    LAST_SPLIT = split_kv_end == cur_batch_seq_len
    k_pe_last_token = tl.zeros([BLOCK_R], dtype=q.dtype)

    if USE_ROPE:
        if IS_NEOX_STYLE:
            # [BLOCK_ROTARY // 2, BLOCK_ROTARY // 2 + 1, BLOCK_ROTARY // 2 + 2, ..., 0, 1, 2, ..., BLOCK_ROTARY // 2 - 1, pass:]
            offs_qk_rot_r = kv_lora_rank + (
                (tl.arange(0, BLOCK_R) + (rotary_dim // 2)) % rotary_dim
            )
            # Which elements to flip
            mask_rotate = tl.arange(0, BLOCK_R) < (rotary_dim // 2)
            # [0 , 1, 2, ..., rotary_dim // 2 - 1, 0 , 1, 2, ..., rotary_dim // 2 - 1]
            offs_rotary = tl.arange(0, BLOCK_R) % (rotary_dim // 2)
        else:
            # [1, 0, 3, 2, 5, 4, ..., BLOCK_R, BLOCK_R - 1]
            offs_qk_rot_r = (
                kv_lora_rank
                + (((tl.arange(0, BLOCK_R) + 1) % 2) * 2)
                - 1
                + tl.arange(0, BLOCK_R)
            )
            mask_rotate = tl.arange(0, BLOCK_R) % 2 < 1
            # [0, 0, 1, 1, ..., rotary_dim // 2 - 1, rotary_dim // 2 - 1]
            offs_rotary = tl.arange(0, BLOCK_R) // 2

        if qk_rope_head_dim > rotary_dim:
            offs_qk_rot_r = tl.where(
                tl.arange(0, BLOCK_R) < rotary_dim, offs_qk_rot_r, tl.arange(0, BLOCK_R)
            )
            offs_rotary = tl.where(
                tl.arange(0, BLOCK_R) < rotary_dim, offs_rotary, tl.arange(0, BLOCK_R)
            )

        mask_rotary = tl.arange(0, BLOCK_R) < rotary_dim

        pos = tl.load(positions + cur_batch * stride_positions_b)
        cos = tl.load(
            cos_sin_cache + pos * stride_cos_sin_cache_s + offs_rotary,
            mask=mask_rotary,
            other=1.0,
        )
        sin = tl.load(
            cos_sin_cache
            + pos * stride_cos_sin_cache_s
            + offs_rotary
            + rotary_dim // 2,
            mask_rotary,
            other=0.0,
        )

        off_q_pe_rot = (
            cur_batch * stride_qb
            + cur_head[:, None] * stride_qh
            + offs_qk_rot_r[None, :]
        )
        mask_qk_rot_r = offs_qk_rot_r < (kv_lora_rank + qk_rope_head_dim)

        # 0, 2, 4,.... 1, 3, 5...
        q_pe_rot = tl.load(
            Q + off_q_pe_rot,
            mask=(mask_h[:, None]) & (mask_qk_rot_r[None, :]),
            other=0.0,
        )
        q_pe_rot = tl.where(mask_rotate[None, :], -q_pe_rot, q_pe_rot)

        q_pe = q_pe * cos + q_pe_rot * sin

        # we only apply to the last token in the K_PE
        if LAST_SPLIT:
            # debug assert
            # if (cur_batch == 0 and cur_head == 0) and split_kv_id < NUM_KV_SPLITS - 1:
            #     tl.device_assert(False, "Only last split should compute k_pe")

            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + cur_batch_seq_len - 1
            )
            offs_buf_k_pe_last_token = kv_loc * stride_buf_kbs + offs_qk_r
            offs_buf_k_pe_rot_last_token = kv_loc * stride_buf_kbs + offs_qk_rot_r
            k_pe_last_token = tl.load(K_Buffer + offs_buf_k_pe_last_token)

            k_pe_rot_last_token = tl.load(K_Buffer + offs_buf_k_pe_rot_last_token)
            k_pe_rot_last_token = tl.where(
                mask_rotate, -k_pe_rot_last_token, k_pe_rot_last_token
            )

            k_pe_last_token = k_pe_last_token * cos + k_pe_rot_last_token * sin

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_C], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            offs_buf_kv = kv_loc[None, :] * stride_buf_kbs + offs_c[:, None]
            offs_buf_k_pe = kv_loc[None, :] * stride_buf_kbs + offs_qk_r[:, None]

            k_pe = tl.load(
                K_Buffer + offs_buf_k_pe,
                mask=(offs_n[None, :] < split_kv_end) & (mask_qk_r[:, None]),
                other=0.0,
            )  # positional embedding part of keys

            if (USE_ROPE and LAST_SPLIT) and start_n >= cur_batch_seq_len - BLOCK_N:
                k_pe = tl.where(
                    offs_n[None, :] != (split_kv_end - 1),
                    k_pe,
                    k_pe_last_token[:, None],
                )

            # (16, 64) x (64, 32)
            # dot product of rope parts
            qk = tl.dot(q_pe, k_pe.to(q_pe.dtype))

            kv = tl.load(
                K_Buffer + offs_buf_kv,
                mask=(offs_n[None, :] < split_kv_end) & (mask_c[:, None]),
                other=0.0,
            )  # the shared latent tensor for keys and values

            # (16, 512) x (512, 32)
            # dot product of nope parts
            qk += tl.dot(q, kv)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = kv_loc[:, None] * stride_buf_vbs + offs_c[None, :]
            v = tl.load(
                V_buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_c[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            # (16, 32) x (32, 512)
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_c[None, :]
        )

        if USE_ROPE:
            if LAST_SPLIT:
                k_pe_last_token_ptrs = (
                    k_pe_t_out
                    + cur_batch * stride_kpe_tokens_out_b
                    + tl.arange(0, BLOCK_R)
                )
                tl.store(k_pe_last_token_ptrs, k_pe_last_token, mask=mask_qk_r)

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_c[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + kv_lora_rank
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


# === tanh ===
@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

