"""AscendC wrapper for ``F.interpolate`` (4D NCHW) — Round 2.

Lesson avoidance (本轮规避):
  - lesson #1: bicubic align_corners=True fp32 边界 ref≈0 处 MARE 超阈
    本轮: host 端 idx/weight 表按 |w| 降序排序; kernel 内 W-axis K_w-tap 求和
          改为 Kahan compensated summation (见 kernel/interpolate_unified_kernel.h).
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _interpolate_ext as _ext


def _resolve_output_size(H_in, W_in, size, scale_factor):
    if size is not None:
        if isinstance(size, (list, tuple)):
            return int(size[0]), int(size[1])
        return int(size), int(size)
    if scale_factor is None:
        return H_in, W_in
    if isinstance(scale_factor, (list, tuple)):
        sf_h, sf_w = float(scale_factor[0]), float(scale_factor[1])
    else:
        sf_h = sf_w = float(scale_factor)
    return int(math.floor(H_in * sf_h)), int(math.floor(W_in * sf_w))


def _src_coord(out_idx, in_size, out_size, align_corners, mode):
    if out_size <= 1:
        return 0.0
    if mode == "nearest":
        return out_idx * in_size / out_size
    if align_corners:
        return out_idx * (in_size - 1) / (out_size - 1)
    return (out_idx + 0.5) * in_size / out_size - 0.5


def _bicubic_kernel(t, a=-0.75):
    t = abs(t)
    if t <= 1.0:
        return ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0
    if t < 2.0:
        return ((a * t - 5.0 * a) * t + 8.0 * a) * t - 4.0 * a
    return 0.0


def _sort_pairs_by_abs_weight(idx_list, w_list):
    paired = sorted(zip(idx_list, w_list), key=lambda p: -abs(p[1]))
    return [p[0] for p in paired], [p[1] for p in paired]


def _build_nearest(in_size, out_size, K=1):
    idx, w = [], []
    for o in range(out_size):
        s = _src_coord(o, in_size, out_size, False, "nearest")
        i = int(math.floor(s))
        if i < 0:
            i = 0
        if i > in_size - 1:
            i = in_size - 1
        idx.append([i] + [0] * (K - 1))
        w.append([1.0] + [0.0] * (K - 1))
    return idx, w


def _build_bilinear(in_size, out_size, align_corners, K=2):
    idx, w = [], []
    for o in range(out_size):
        s = _src_coord(o, in_size, out_size, bool(align_corners), "bilinear")
        if s < 0.0:
            s = 0.0
        if s > in_size - 1:
            s = float(in_size - 1)
        i0 = int(math.floor(s))
        i1 = i0 + 1
        if i1 > in_size - 1:
            i1 = in_size - 1
        frac = s - i0
        row_idx = [i0, i1] + [0] * (K - 2)
        row_w = [1.0 - frac, frac] + [0.0] * (K - 2)
        ri, rw = _sort_pairs_by_abs_weight(row_idx, row_w)
        idx.append(ri)
        w.append(rw)
    return idx, w


def _build_bicubic(in_size, out_size, align_corners, K=4):
    idx, w = [], []
    for o in range(out_size):
        s = _src_coord(o, in_size, out_size, bool(align_corners), "bilinear")
        i_floor = int(math.floor(s))
        frac = s - i_floor
        ks = [i_floor - 1, i_floor, i_floor + 1, i_floor + 2]
        offs = [-1.0 - frac, -frac, 1.0 - frac, 2.0 - frac]
        ws = [_bicubic_kernel(d) for d in offs]
        clamped = []
        for k in ks:
            if k < 0:
                k = 0
            if k > in_size - 1:
                k = in_size - 1
            clamped.append(k)
        while len(clamped) < K:
            clamped.append(0)
            ws.append(0.0)
        ri, rw = _sort_pairs_by_abs_weight(clamped, ws)
        idx.append(ri)
        w.append(rw)
    return idx, w


def _build_area(in_size, out_size, K_max):
    idx, w = [], []
    for o in range(out_size):
        start = int(math.floor(o * in_size / out_size))
        end = int(math.ceil((o + 1) * in_size / out_size))
        if end <= start:
            end = start + 1
        if end > in_size:
            end = in_size
        if start > in_size - 1:
            start = in_size - 1
        span = end - start
        inv = 1.0 / float(span)
        row_idx = []
        row_w = []
        for k in range(K_max):
            if k < span:
                row_idx.append(start + k)
                row_w.append(inv)
            else:
                row_idx.append(0)
                row_w.append(0.0)
        ri, rw = _sort_pairs_by_abs_weight(row_idx, row_w)
        idx.append(ri)
        w.append(rw)
    return idx, w


def _select_mode(mode, H_in, W_in, H_out, W_out):
    if mode in ("linear", "bilinear"):
        return "bilinear", 2, 2
    if mode == "bicubic":
        return "bicubic", 4, 4
    if mode == "nearest":
        return "nearest", 1, 1
    if mode == "area":
        if H_out >= H_in and W_out >= W_in:
            return "nearest", 1, 1
        K_h = max(1, int(math.ceil(H_in / max(H_out, 1))))
        K_w = max(1, int(math.ceil(W_in / max(W_out, 1))))
        return "area", K_h, K_w
    return "nearest", 1, 1


def _build_tables(eff_mode, in_size, out_size, K, align_corners):
    if eff_mode == "nearest":
        return _build_nearest(in_size, out_size, K=K)
    if eff_mode == "bilinear":
        return _build_bilinear(in_size, out_size, align_corners, K=K)
    if eff_mode == "bicubic":
        return _build_bicubic(in_size, out_size, align_corners, K=K)
    if eff_mode == "area":
        return _build_area(in_size, out_size, K)
    return _build_nearest(in_size, out_size, K=K)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, size=None, scale_factor=None,
                mode="nearest", align_corners=None,
                recompute_scale_factor=None, antialias=False):
        N = int(x.shape[0])
        C = int(x.shape[1])
        H_in = int(x.shape[2])
        W_in = int(x.shape[3])
        H_out, W_out = _resolve_output_size(H_in, W_in, size, scale_factor)
        NC = N * C

        eff_mode, K_h, K_w = _select_mode(mode, H_in, W_in, H_out, W_out)

        h_idx_list, h_w_list = _build_tables(eff_mode, H_in, H_out, K_h, align_corners)
        w_idx_list, w_w_list = _build_tables(eff_mode, W_in, W_out, K_w, align_corners)

        device = x.device
        h_idx = torch.tensor(h_idx_list, dtype=torch.int32, device=device).contiguous()
        w_idx = torch.tensor(w_idx_list, dtype=torch.int32, device=device).contiguous()
        h_w = torch.tensor(h_w_list, dtype=torch.float32, device=device).contiguous()
        w_w = torch.tensor(w_w_list, dtype=torch.float32, device=device).contiguous()

        x_flat = x.reshape(NC, H_in, W_in).contiguous()
        y_flat = _ext.run_interpolate(
            x_flat, h_idx, w_idx, h_w, w_w,
            NC, H_in, W_in, H_out, W_out, K_h, K_w)

        return y_flat.reshape(N, C, H_out, W_out)
