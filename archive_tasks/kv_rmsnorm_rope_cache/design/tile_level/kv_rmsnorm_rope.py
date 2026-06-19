"""Tile-level TileLang design for KvRmsnormRopeCache.

Completes the block-level skeleton with full tile-level compute details.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[4, 5], pass_configs=pass_configs)
def kv_rmsnorm_rope(total, rms_size, rope_size, eps=1e-5, dtype="float16"):
    block_size = 64
    num_physical_cores = 20
    total_blocks = (total + block_size - 1) // block_size
    used_core_num = min(num_physical_cores, total_blocks)
    tasks_per_core = (total_blocks + used_core_num - 1) // used_core_num
    vec_num = 2
    sub_block_size = block_size // vec_num

    need_cast = dtype != "float32"
    out_cast_mode = "CAST_ROUND" if dtype == "bfloat16" else "CAST_NONE"
    eps_const = T.float32(eps)
    inv_rms_size = T.float32(1.0 / rms_size)

    @T.prim_func
    def main(
        rms_in: T.Tensor((total, rms_size), dtype),
        gamma: T.Tensor((rms_size,), dtype),
        k_input: T.Tensor((total, rope_size), dtype),
        cos: T.Tensor((total, rope_size), dtype),
        sin: T.Tensor((total, rope_size), dtype),
        v_out: T.Tensor((total, rms_size), dtype),
        k_embed_out: T.Tensor((total, rope_size), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            core_idx = cid

            with T.Scope("V"):
                gamma_in_ub = T.alloc_ub((rms_size,), dtype)
                gamma_ub = T.alloc_ub((rms_size,), "float32")
                if need_cast:
                    T.copy(gamma[0], gamma_in_ub)
                    T.tile.cast(gamma_ub, gamma_in_ub, mode="CAST_NONE", count=rms_size)
                else:
                    T.copy(gamma[0], gamma_ub)

                eps_ub = T.alloc_ub((1,), "float32")
                inv_n_ub = T.alloc_ub((1,), "float32")
                T.tile.fill(eps_ub, eps_const)
                T.tile.fill(inv_n_ub, inv_rms_size)

                x_in_ub = T.alloc_ub((rms_size,), dtype)
                x_ub = T.alloc_ub((rms_size,), "float32")
                x_sq_ub = T.alloc_ub((rms_size,), "float32")
                sum_sq_ub = T.alloc_ub((1,), "float32")
                inv_rms_ub = T.alloc_ub((1,), "float32")
                out_ub = T.alloc_ub((rms_size,), "float32")
                out_cast_ub = T.alloc_ub((rms_size,), dtype)
                reduce_tmp = T.alloc_ub((2 * rms_size,), "uint8")

                k_in_ub = T.alloc_ub((rope_size,), dtype)
                k_ub = T.alloc_ub((rope_size,), "float32")
                cos_ub = T.alloc_ub((rope_size,), "float32")
                sin_ub = T.alloc_ub((rope_size,), "float32")
                rotate_half_ub = T.alloc_ub((rope_size,), "float32")
                tmp1_ub = T.alloc_ub((rope_size,), "float32")
                tmp2_ub = T.alloc_ub((rope_size,), "float32")
                k_embed_ub = T.alloc_ub((rope_size,), "float32")
                k_embed_cast_ub = T.alloc_ub((rope_size,), dtype)

                for local_idx in T.serial(tasks_per_core):
                    bx = core_idx * tasks_per_core + local_idx
                    if bx < total_blocks:
                        for row in T.serial(sub_block_size):
                            pos = bx * block_size + vid * sub_block_size + row
                            if pos < total:
                                # RMSNorm
                                if need_cast:
                                    T.copy(rms_in[pos, :], x_in_ub)
                                    T.tile.cast(x_ub, x_in_ub, mode="CAST_NONE", count=rms_size)
                                else:
                                    T.copy(rms_in[pos, :], x_ub)

                                T.tile.mul(x_sq_ub, x_ub, x_ub)
                                T.reduce_sum(x_sq_ub, sum_sq_ub, reduce_tmp, dim=-1)
                                T.tile.mul(sum_sq_ub, sum_sq_ub, inv_n_ub[0])
                                T.tile.add(sum_sq_ub, sum_sq_ub, eps_ub[0])
                                T.tile.rsqrt(inv_rms_ub, sum_sq_ub)

                                inv_rms = inv_rms_ub[0]
                                T.tile.mul(out_ub, x_ub, inv_rms)
                                T.tile.mul(out_ub, out_ub, gamma_ub)

                                if need_cast:
                                    T.tile.cast(out_cast_ub, out_ub, mode=out_cast_mode, count=rms_size)
                                    T.copy(out_cast_ub, v_out[pos, :])
                                else:
                                    T.copy(out_ub, v_out[pos, :])

                                # RoPE
                                if need_cast:
                                    T.copy(k_input[pos, :], k_in_ub)
                                    T.tile.cast(k_ub, k_in_ub, mode="CAST_NONE", count=rope_size)
                                    T.copy(cos[pos, :], k_in_ub)
                                    T.tile.cast(cos_ub, k_in_ub, mode="CAST_NONE", count=rope_size)
                                    T.copy(sin[pos, :], k_in_ub)
                                    T.tile.cast(sin_ub, k_in_ub, mode="CAST_NONE", count=rope_size)
                                else:
                                    T.copy(k_input[pos, :], k_ub)
                                    T.copy(cos[pos, :], cos_ub)
                                    T.copy(sin[pos, :], sin_ub)

                                half = rope_size // 2
                                T.tile.neg(rotate_half_ub[:half], k_ub[half:rope_size])
                                T.copy(k_ub[:half], rotate_half_ub[half:rope_size])

                                T.tile.mul(tmp1_ub, k_ub, cos_ub)
                                T.tile.mul(tmp2_ub, rotate_half_ub, sin_ub)
                                T.tile.add(k_embed_ub, tmp1_ub, tmp2_ub)

                                if need_cast:
                                    T.tile.cast(k_embed_cast_ub, k_embed_ub, mode=out_cast_mode, count=rope_size)
                                    T.copy(k_embed_cast_ub, k_embed_out[pos, :])
                                else:
                                    T.copy(k_embed_ub, k_embed_out[pos, :])

    return main
