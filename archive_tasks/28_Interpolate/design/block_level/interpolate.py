"""Block-level design for ``F.interpolate`` (4D NCHW resampling) — Round 2.

Lesson avoidance (本轮规避):
  - lesson #1: bicubic + align_corners=True 在 fp32 + 边界 ref≈0 处的 MARE 超阈
    避免方式: kernel 内 W 方向 K-tap 累加改用 Kahan compensated summation;
              host 端对每个输出位置的 K_h / K_w 权重按 |w| 降序排列, 减小累加震荡;
              不再依赖单一 fp32 累加路径

Operator semantics: same as Round 1.
    Input:  x in [N, C, H_in, W_in]
    Output: y in [N, C, H_out, W_out]
    Modes:  "nearest" / "bilinear" / "bicubic" / "area"
    Aux:    align_corners ∈ {True, False, None}

Block-level decomposition: same as Round 1 (NC = N*C 作为外层并行轴, kernel 内
serial 遍历每个 (nc, h_out) 行). 这一层无需改动 —— 改动集中在 tile-level
的 W 方向累加方式.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[6], pass_configs=pass_configs)
def block_design(NC, H_in, W_in, H_out, W_out, K_h, K_w, dtype="float32",
                 mode="bilinear"):
    """Block-level skeleton — picks an internal ``T.prim_func`` by ``mode``.

    See tile_level/interpolate.py for the actual kernel realisation.
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
        h_w:   T.Tensor((H_out, K_h), dtype),  # already |w| desc sorted on host
        w_w:   T.Tensor((W_out, K_w), dtype),  # already |w| desc sorted on host
        Y: T.Tensor((NC, H_out, W_out), dtype),
    ):
        with T.Kernel(used_cores, is_npu=True) as (cid, vid):
            for local in T.serial(tasks_per_core):
                bx = cid * tasks_per_core + local
                with T.Scope("V"):
                    if bx < NC:
                        # TODO(tile-level):
                        #  Phase 1: row_mix = sum_kh(h_w[h_out,kh]*X[bx, h_idx[h_out,kh], :])
                        #           via Axpy (vector FMA).  K_h ≤ 4.
                        #  Phase 2: per w_out, K_w-tap *Kahan compensated* scalar sum:
                        #             acc=0; c=0
                        #             for kw in 0..K_w:
                        #               y = w_w[w_out,kw] * row_mix[w_idx[w_out,kw]] - c
                        #               t = acc + y
                        #               c = (t - acc) - y
                        #               acc = t
                        #           This bounds W-axis fp32 accumulation error to O(eps^2)
                        #           instead of O(K*eps); critical for bicubic boundary cases.
                        T.evaluate(0)

    return unified_kahan
