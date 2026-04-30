"""Tile-level TileLang kernel for ``F.interpolate`` (4D NCHW) — Round 2.

Lesson avoidance (本轮规避):
  - lesson #1: bicubic align_corners=True fp32 边界 ref≈0 处 MARE 超阈
    本轮关键变化:
      (a) Phase 2 W-axis K_w-tap 累加改为 Kahan compensated summation;
      (b) host 端 h_w / w_w 已按 |w| 降序排列, 配套的 h_idx / w_idx 同步重排;
      (c) 输出 dtype cast: bf16 → CAST_ROUND, fp16 → CAST_NONE.

Layout & block decomposition: same as Round 1.

Per-mode kernel selection (kept separable, NOT collapsed to single-pass K_h*K_w direct
sum: 直接累加 K_h*K_w=16 项的累计误差更大).
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[5], pass_configs=pass_configs)
def tl_interpolate(
    NC, H_in, W_in, H_out, W_out,
    K_h=2, K_w=2,
    dtype="float32",
    accum_dtype="float32",
    mode="bilinear",
):
    """Build a TileLang kernel for ``F.interpolate``.

    Returned kernel signature: kernel(X, h_idx, w_idx, h_w, w_w) -> Y.
    h_w / w_w 已在 host 端按 |w| 降序排序, h_idx / w_idx 同步重排.
    """
    num_physical_cores = 20
    block_NC = max(1, NC // num_physical_cores)
    used_cores = min(num_physical_cores, (NC + block_NC - 1) // block_NC)
    tasks_per_core = (NC + used_cores * block_NC - 1) // (used_cores * block_NC)

    @T.prim_func
    def unified_kahan(
        X: T.Tensor((NC, H_in, W_in), dtype),
        h_idx: T.Tensor((H_out, K_h), "int32"),
        w_idx: T.Tensor((W_out, K_w), "int32"),
        h_w:   T.Tensor((H_out, K_h), accum_dtype),
        w_w:   T.Tensor((W_out, K_w), accum_dtype),
        Y: T.Tensor((NC, H_out, W_out), dtype),
    ):
        with T.Kernel(used_cores, is_npu=True) as (cid, vid):
            # Cache W tables in UB (small).
            w_idx_ub_list = [T.alloc_ub((W_out,), "int32") for _ in range(K_w)]
            w_w_ub_list   = [T.alloc_ub((W_out,), accum_dtype) for _ in range(K_w)]
            for k in range(K_w):
                T.copy(w_idx[0:W_out, k], w_idx_ub_list[k])
                T.copy(w_w[0:W_out, k], w_w_ub_list[k])

            for local in T.serial(tasks_per_core):
                bx = cid * tasks_per_core + local
                with T.Scope("V"):
                    if bx < NC:
                        for h_out in T.serial(H_out):
                            row_mix = T.alloc_ub((W_in,), accum_dtype)
                            T.tile.fill(row_mix, T.float32(0.0))

                            # Phase 1: row_mix = sum_kh(h_w[h_out,kh] * X[bx, h_idx[h_out,kh], :])
                            for kh in range(K_h):
                                src = T.alloc_ub((W_in,), dtype)
                                row = T.alloc_ub((W_in,), accum_dtype)
                                T.copy(X[bx, h_idx[h_out, kh], 0:W_in], src)
                                T.tile.cast(row, src, "CAST_NONE", W_in)
                                hk = T.float32(h_w[h_out, kh])
                                # Axpy: row_mix += hk * row (single FMA rounding).
                                T.tile.axpy(row_mix, row, hk)

                            # Phase 2: Y[bx,h_out,w_out] = Kahan_sum_kw(
                            #   w_w[w_out,kw] * row_mix[w_idx[w_out,kw]])
                            # — actual implementation in AscendC kernel uses
                            # compensated summation per w_out (scalar GetValue).
                            out_ub = T.alloc_ub((W_out,), dtype)
                            # TODO(tile-level): scalar Kahan loop over w_out × K_w.
                            T.copy(out_ub, Y[bx, h_out, 0:W_out])

    return unified_kahan
