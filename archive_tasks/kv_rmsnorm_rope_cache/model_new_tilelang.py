import torch
from design.tile_level.kv_rmsnorm_rope import kv_rmsnorm_rope


def _update_cache_norm(k_cache, ckv_cache, k_embed, v, index):
    """Norm mode cache update."""
    k_cache_out = k_cache.clone()
    ckv_cache_out = ckv_cache.clone()
    B, N, S = k_embed.shape[0], k_embed.shape[1], k_embed.shape[2]
    if index.dim() == 2:
        for b in range(B):
            for s in range(S):
                idx = index[b, s].item()
                if idx < 0:
                    continue
                if idx < k_cache_out.shape[2]:
                    k_cache_out[b, :, idx, :] = k_embed[b, :, s, :]
                    ckv_cache_out[b, :, idx, :] = v[b, :, s, :]
    else:
        for i in range(index.numel()):
            idx = index[i].item()
            if idx < 0:
                continue
            b_idx = i // S
            s_idx = i % S
            if b_idx < B and s_idx < S and idx < k_cache_out.shape[2]:
                k_cache_out[b_idx, :, idx, :] = k_embed[b_idx, :, s_idx, :]
                ckv_cache_out[b_idx, :, idx, :] = v[b_idx, :, s_idx, :]
    return k_cache_out, ckv_cache_out


def _update_cache_pa(k_cache, ckv_cache, k_embed, v, index):
    """PA / PA_BNSD mode cache update."""
    k_cache_out = k_cache.clone()
    ckv_cache_out = ckv_cache.clone()
    B, N, S = k_embed.shape[0], k_embed.shape[1], k_embed.shape[2]
    k_flat = k_embed.reshape(B * S, N, -1)
    v_flat = v.reshape(B * S, N, -1)
    cache_shape_k = k_cache_out.shape
    cache_shape_v = ckv_cache_out.shape
    k_cache_flat = k_cache_out.reshape(-1, N, cache_shape_k[-1])
    v_cache_flat = ckv_cache_out.reshape(-1, N, cache_shape_v[-1])
    for i in range(min(len(index), B * S)):
        idx = index[i].item()
        if idx < 0:
            continue
        if idx < k_cache_flat.shape[0]:
            k_cache_flat[idx, :, :] = k_flat[i, :, :]
            v_cache_flat[idx, :, :] = v_flat[i, :, :]
    return k_cache_flat.reshape(cache_shape_k), v_cache_flat.reshape(cache_shape_v)


def _update_cache_pa_nz(k_cache, ckv_cache, k_embed, v, index):
    """PA_NZ mode cache update."""
    k_cache_out = k_cache.clone()
    ckv_cache_out = ckv_cache.clone()
    B, N, S = k_embed.shape[0], k_embed.shape[1], k_embed.shape[2]
    block_size = k_cache_out.shape[1]
    dk = k_cache_out.shape[-1]
    dv = ckv_cache_out.shape[-1]
    dk0 = 32 if k_cache_out.dtype == torch.int8 else 16
    dv0 = 32 if ckv_cache_out.dtype == torch.int8 else 16
    dk1 = dk // dk0
    dv1 = dv // dv0
    bn = k_cache_out.shape[0]
    num_head = k_cache_out.shape[2]
    k_cache_nz = k_cache_out.reshape(bn, num_head, dk1, block_size, dk0)
    v_cache_nz = ckv_cache_out.reshape(bn, num_head, dv1, block_size, dv0)
    k_flat = k_embed.reshape(B * S, N, -1)
    v_flat = v.reshape(B * S, N, -1)
    for i in range(min(len(index), B * S)):
        idx = index[i].item()
        if idx < 0:
            continue
        bn_id = idx // block_size
        block_offset = idx % block_size
        if bn_id < bn:
            for d in range(dk1):
                k_cache_nz[bn_id, :, d, block_offset, :] = k_flat[i, :, d * dk0:(d + 1) * dk0]
            for d in range(dv1):
                v_cache_nz[bn_id, :, d, block_offset, :] = v_flat[i, :, d * dv0:(d + 1) * dv0]
    return k_cache_nz.reshape(k_cache_out.shape), v_cache_nz.reshape(ckv_cache_out.shape)


def _update_cache_pa_blk_bnsd(k_cache, ckv_cache, k_embed, v, index):
    """PA_BLK_BNSD mode cache update."""
    k_cache_out = k_cache.clone()
    ckv_cache_out = ckv_cache.clone()
    B, N, S = k_embed.shape[0], k_embed.shape[1], k_embed.shape[2]
    block_size = k_cache_out.shape[1]
    ceil_div_s = (S + block_size - 1) // block_size
    for batch in range(B):
        for seq_id in range(ceil_div_s):
            seq_start = seq_id * block_size
            seq_end = S if seq_id == (ceil_div_s - 1) else (seq_id + 1) * block_size
            copy_len = seq_end - seq_start
            idx_pos = batch * ceil_div_s + seq_id
            if idx_pos >= len(index):
                continue
            idx_val = index[idx_pos].item()
            if idx_val < 0:
                continue
            cache_b = idx_val // block_size
            if cache_b < k_cache_out.shape[0]:
                k_cache_out[cache_b, :copy_len, :, :] = k_embed[batch, seq_start:seq_end, :, :]
                ckv_cache_out[cache_b, :copy_len, :, :] = v[batch, seq_start:seq_end, :, :]
    return k_cache_out, ckv_cache_out


def _update_cache_pa_blk_nz(k_cache, ckv_cache, k_embed, v, index):
    """PA_BLK_NZ mode cache update."""
    k_cache_out = k_cache.clone()
    ckv_cache_out = ckv_cache.clone()
    B, N, S = k_embed.shape[0], k_embed.shape[1], k_embed.shape[2]
    block_size = k_cache_out.shape[1]
    dk = k_cache_out.shape[-1]
    dv = ckv_cache_out.shape[-1]
    dk0 = 32 if k_cache_out.dtype == torch.int8 else 16
    dv0 = 32 if ckv_cache_out.dtype == torch.int8 else 16
    dk1 = dk // dk0
    dv1 = dv // dv0
    bn = k_cache_out.shape[0]
    num_head = k_cache_out.shape[2]
    ceil_div_s = (S + block_size - 1) // block_size
    k_cache_nz = k_cache_out.reshape(bn, num_head, dk1, block_size, dk0)
    v_cache_nz = ckv_cache_out.reshape(bn, num_head, dv1, block_size, dv0)
    for batch in range(B):
        for seq_id in range(ceil_div_s):
            seq_start = seq_id * block_size
            seq_end = S if seq_id == (ceil_div_s - 1) else (seq_id + 1) * block_size
            copy_len = seq_end - seq_start
            idx_pos = batch * ceil_div_s + seq_id
            if idx_pos >= len(index):
                continue
            idx_val = index[idx_pos].item()
            if idx_val < 0:
                continue
            cache_b = idx_val // block_size
            if cache_b < bn:
                for n_idx in range(num_head):
                    for d in range(dk1):
                        k_cache_nz[cache_b, n_idx, d, :copy_len, :] = k_embed[batch, seq_start:seq_end, n_idx, d * dk0:(d + 1) * dk0]
                    for d in range(dv1):
                        v_cache_nz[cache_b, n_idx, d, :copy_len, :] = v[batch, seq_start:seq_end, n_idx, d * dv0:(d + 1) * dv0]
    return k_cache_nz.reshape(k_cache_out.shape), v_cache_nz.reshape(ckv_cache_out.shape)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def postprocess_output(self, output, inputs):
        if len(inputs) >= 14:
            cache_mode = inputs[12]
            is_output_kv = inputs[13]
            if cache_mode == 'Norm' or not is_output_kv:
                return output[:2]
        return output

    def forward(self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                k_rope_scale=None, c_kv_scale=None,
                k_rope_offset=None, c_kv_offset=None,
                epsilon=1e-5, cache_mode='Norm', is_output_kv=False):
        B, N, S, hidden_size = kv.shape
        rms_size = gamma.shape[0]
        rope_size = hidden_size - rms_size

        # Rearrange BNSD -> BSND
        kv_bsnd = kv.permute(0, 2, 1, 3)
        cos_bsnd = cos.permute(0, 2, 1, 3)
        sin_bsnd = sin.permute(0, 2, 1, 3)

        rms_in = kv_bsnd[..., :rms_size]
        rope_in = kv_bsnd[..., rms_size:]

        # Preprocess rope_in: interleave pairs (same as reshape+transpose+reshape in reference)
        k_input = rope_in.reshape(B, S, N, rope_size // 2, 2).permute(0, 1, 2, 4, 3).reshape(B, S, N, rope_size)

        # Flatten position dimension
        total = B * S * N
        rms_in_flat = rms_in.reshape(total, rms_size)
        k_input_flat = k_input.reshape(total, rope_size)
        cos_flat = cos_bsnd.reshape(total, rope_size)
        sin_flat = sin_bsnd.reshape(total, rope_size)

        # Build and call TileLang kernel
        dtype_str = str(kv.dtype).split('.')[-1]
        kernel = kv_rmsnorm_rope(total, rms_size, rope_size, eps=epsilon, dtype=dtype_str)
        v_flat, k_embed_flat = kernel(rms_in_flat, gamma, k_input_flat, cos_flat, sin_flat)

        # Reshape outputs back to BNSD
        v = v_flat.reshape(B, S, N, rms_size).permute(0, 2, 1, 3)
        k_embed = k_embed_flat.reshape(B, S, N, rope_size).permute(0, 2, 1, 3)

        # Cache update (dispatch by cache_mode)
        if cache_mode == 'Norm':
            k_cache_out, ckv_cache_out = _update_cache_norm(k_cache, ckv_cache, k_embed, v, index)
        elif cache_mode in ('PA', 'PA_BNSD'):
            k_cache_out, ckv_cache_out = _update_cache_pa(k_cache, ckv_cache, k_embed, v, index)
        elif cache_mode == 'PA_NZ':
            k_cache_out, ckv_cache_out = _update_cache_pa_nz(k_cache, ckv_cache, k_embed, v, index)
        elif cache_mode == 'PA_BLK_BNSD':
            k_cache_out, ckv_cache_out = _update_cache_pa_blk_bnsd(k_cache, ckv_cache, k_embed, v, index)
        elif cache_mode == 'PA_BLK_NZ':
            k_cache_out, ckv_cache_out = _update_cache_pa_blk_nz(k_cache, ckv_cache, k_embed, v, index)
        else:
            raise ValueError(f"Unsupported cache_mode: {cache_mode}")

        if is_output_kv:
            k_embed_ret = k_embed
            y_ret = v
        else:
            k_embed_ret = None
            y_ret = None

        return k_cache_out, ckv_cache_out, k_embed_ret, y_ret
