#!/usr/bin/env python3
"""
precision_forensics.py — 精度取证数值分析

职责边界:
  ✅ 做: diff 统计、pattern hint 分类(含算子语义加权)、worst 定位、
        尾块分析、维度分析、误差分布分析、内存布局分析、历史轮次对比
  ❌ 不做: 代码分析、根因诊断、修复建议（这些是 Agent 的事）

信息层级:
  L0: PASS/FAIL           — 直接从 mismatch 判定
  L1: 统计值              — basic_stats, error_distribution
  L2: 位置特征            — tail, dimension, boundary concentration
  L3: 数值特征            — magnitude correlation, NaN/Inf, sign analysis
  L4: 张量切片            — per-index 分布, worst elements
  L5: 中间结果            — TODO: 需要 intermediate_probe (重编译)
  L6: 内存布局            — tensor stride/layout 分析 (已实现)
  L7: 代码位置映射         — TODO: 需要理解 tiling 逻辑 (Agent 做)
  L8: 算子语义            — 算子类型检测 + 语义感知 pattern 权重 (部分实现)

用法:
    python3 precision_forensics.py <op_name> --output-path <path> [--attempt <N>]

输出:
    {output_path}/precision_tuning/forensics_report_{attempt}.json

依赖: torch, torch_npu, numpy
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def setup_ascend_environment(op_name: str, output_path: str) -> str | None:
    """
    设置AscendC算子运行所需的全部环境变量。

    包含三个部分:
    1. ASCEND_CUSTOM_OPP_PATH - 指向算子编译产物目录
    2. LD_LIBRARY_PATH - 包含torch、torch_npu和op_api的库路径

    目录查找顺序(优先从短路径开始):
      1. {output_path}/vendors/customize          (安装后目录)
      2. {OpName}Custom/vendors/customize          (编译输出目录)
      3. {OpName}Custom/build_out/.../customize    (打包后的run包)

    环境路径配置说明:
    - torch_lib: torch库的lib目录, 包含libc10.so等核心库
    - torch_npu_lib: torch_npu库的lib目录, 包含libtorch_npu.so
    - op_api_lib: AscendC OP API库目录, 包含libcust_opapi.so
    """
    base = Path(output_path).resolve()
    op_name_lower = op_name.lower()

    # 查找顺序：优先短路径 (output_path/vendors/customize)
    # 再试 *Custom/vendors/customize
    # 最后试 build_out 中的 run 包
    search_patterns = [
        # 1. output_path/vendors/customize (最优先，路径最短)
        (base / "vendors" / "customize", "installed_vendors"),
        # 2. *Custom/vendors/customize (编译输出目录)
        None,  # 动态查找，见下
        # 3. *Custom/build_out/.../customize.run/packages/vendors/customize
        None,  # 动态查找，见下
    ]

    custom_opp = None
    found_in = None

    # 查找模式1: output_path/vendors/customize
    if (search_patterns[0][0]).exists():
        custom_opp = search_patterns[0][0]
        found_in = search_patterns[0][1]

    # 查找模式2: *Custom/vendors/customize
    if not custom_opp:
        for item in base.iterdir():
            if item.is_dir() and item.name.lower().endswith('custom'):
                candidate = item / "vendors" / "customize"
                if candidate.exists():
                    custom_opp = candidate
                    found_in = "build_custom_vendors"
                    break

    # 查找模式3: build_out 中的 run 包
    if not custom_opp:
        for item in base.iterdir():
            if item.is_dir() and item.name.lower().endswith('custom'):
                run_pkg = (
                    item / "build_out" / "_CPack_Packages" / "Linux" /
                    "External" / "custom_opp_ubuntu_aarch64.run" /
                    "packages" / "vendors" / "customize"
                )
                if run_pkg.exists():
                    custom_opp = run_pkg
                    found_in = "cpack_run_package"
                    break

    # 设置 ASCEND_CUSTOM_OPP_PATH
    if custom_opp and custom_opp.exists():
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp)

        # 设置 LD_LIBRARY_PATH - 按优先级追加路径
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        paths_to_add = []

        # 1. torch/lib (必须, 否则 ImportError: libc10.so)
        try:
            import torch
            torch_lib = Path(torch.__file__).parent / "lib"
            if torch_lib.exists():
                paths_to_add.append(str(torch_lib))
        except Exception:
            pass

        # 2. torch_npu/lib (必须, 否则 ImportError: libtorch_npu.so)
        try:
            import torch_npu
            torch_npu_lib = Path(torch_npu.__file__).parent / "lib"
            if torch_npu_lib.exists():
                paths_to_add.append(str(torch_npu_lib))
        except Exception:
            pass

        # 3. op_api/lib (AscendC算子API库)
        op_api_lib = custom_opp / "op_api" / "lib"
        if op_api_lib.exists():
            paths_to_add.append(str(op_api_lib))

        # 合并到 LD_LIBRARY_PATH (避免重复)
        new_ld = ":".join(paths_to_add)
        if existing_ld:
            new_ld = f"{new_ld}:{existing_ld}"
        os.environ["LD_LIBRARY_PATH"] = new_ld

        return found_in
    return None


import numpy as np
import torch
import torch_npu  # noqa: F401


# ============================================================
# 算子类型检测 (L8 基础)
# ============================================================

class OperatorTypeDetector:
    """从 op_desc.json 或算子名推断算子类型, 用于语义加权 pattern hint"""

    OP_TYPE_PATTERN_PRIORITY = {
        "reduction": ["magnitude_correlated", "tail_spike", "uniform_offset", "scattered"],
        "pooling": ["tail_spike", "boundary_concentration", "uniform_offset"],
        "loss": ["magnitude_correlated", "uniform_offset", "nan_inf_contamination"],
        "matmul": ["dimension_concentration", "scattered", "magnitude_correlated"],
        "activation": ["nan_inf_contamination", "uniform_offset", "boundary_concentration"],
        "normalization": ["nan_inf_contamination", "magnitude_correlated", "tail_spike"],
        "convolution": ["dimension_concentration", "boundary_concentration", "tail_spike"],
    }

    def detect(self, op_name: str, output_path: str) -> dict:
        op_type, source = "unknown", "none"
        attributes = {}
        shape_info = {}

        # 1. 尝试从 op_desc.json 读取 (优先级最高)
        desc_path = os.path.join(output_path, f"{op_name}_op_desc.json")
        if os.path.exists(desc_path):
            try:
                with open(desc_path) as f:
                    desc = json.load(f)
                # 兼容多种字段名: op_type / type / category
                op_type = desc.get("op_type", desc.get("type", desc.get("category", "unknown"))).lower()
                source = "op_desc.json"
                attributes = desc.get("attributes", {})
                shape_info = desc.get("shape_info", {})
            except (json.JSONDecodeError, KeyError):
                pass

        # 2. fallback: 从 project.json 读取
        if op_type == "unknown":
            proj_path = os.path.join(output_path, f"{op_name}_project.json")
            if os.path.exists(proj_path):
                try:
                    with open(proj_path) as f:
                        proj = json.load(f)
                    if isinstance(proj, list) and len(proj) > 0:
                        op_type = proj[0].get("op", "unknown").lower()
                    elif isinstance(proj, dict):
                        op_type = proj.get("op_type", "unknown").lower()
                    else:
                        op_type = "unknown"
                    source = "project.json"
                except (json.JSONDecodeError, KeyError):
                    pass

        # 3. fallback: 从算子名推断
        if op_type == "unknown":
            op_type = self._infer_from_name(op_name)
            if op_type != "unknown":
                source = "name_inference"

        result = {
            "op_type": op_type,
            "source": source,
            "pattern_priority": self.OP_TYPE_PATTERN_PRIORITY.get(op_type, []),
            "attributes": attributes,
        }

        # 推断归约轴信息 (对有 dim 属性的算子)
        reduction_info = self._infer_reduction_axis(attributes, shape_info)
        if reduction_info:
            result["reduction_axis"] = reduction_info

        return result

    def _infer_reduction_axis(self, attributes: dict, shape_info: dict) -> dict | None:
        """从 attributes.dim 和 shape_info 推断归约轴信息"""
        dim = attributes.get("dim")
        if dim is None:
            # 检查 normalized_dim (LayerNorm) 或 normalized_shape
            norm_shape = attributes.get("normalized_shape")
            norm_dim = attributes.get("normalized_dim")
            if norm_dim is not None:
                dim = norm_dim
            elif norm_shape is not None:
                # normalized_shape=[768] 意味着归约最后 len(normalized_shape) 个维度
                dim = -1  # 简化: 取最后一维
            else:
                return None

        input_shapes = shape_info.get("input_shapes", [])
        if not input_shapes:
            return {"dim": dim, "axis": None, "axis_length": None}

        shape = input_shapes[0].get("shape", [])
        ndim = len(shape)
        axis = dim if dim >= 0 else ndim + dim
        axis_length = shape[axis] if 0 <= axis < ndim else None

        return {
            "dim": dim,
            "axis": axis,
            "axis_length": axis_length,
            "input_shape": shape,
            "tiling_constraint": "reduction_axis_must_be_intact",
        }

    def _infer_from_name(self, op_name: str) -> str:
        name = op_name.lower()
        rules = [
            (["pool", "avg_pool", "max_pool"], "pooling"),
            (["reduce", "sum", "mean", "prod", "cumsum"], "reduction"),
            (["loss", "mse", "cross_entropy", "nll"], "loss"),
            (["matmul", "bmm", "linear", "gemm"], "matmul"),
            (["relu", "gelu", "silu", "sigmoid", "tanh", "leaky", "softmax", "log_softmax"], "activation"),
            (["norm", "layer_norm", "batch_norm", "group_norm"], "normalization"),
            (["conv", "conv2d", "conv1d"], "convolution"),
        ]
        for keywords, t in rules:
            if any(kw in name for kw in keywords):
                return t
        return "unknown"


# ============================================================
# 执行引擎
# ============================================================

class OperatorExecutor:
    """复用 evaluate.py 的 exec + Model/ModelNew 模式, 获取 raw output tensor"""

    def __init__(self, op_name: str, output_path: str):
        self.op_name = op_name
        self.output_path = output_path
        self.device = torch.device("npu:0")
        self.context = {}

    def setup_environment(self):
        """
        设置AscendC算子运行所需的全部环境变量。

        委托给模块级 setup_ascend_environment() 函数执行实际逻辑。
        此方法保留用于保持向后兼容。
        """
        setup_ascend_environment(self.op_name, self.output_path)

    def load_and_execute(self) -> dict:

        ref_path = os.path.join(self.output_path, f"{self.op_name}_reference.py")
        custom_path = os.path.join(self.output_path, f"{self.op_name}_custom.py")
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"参考代码不存在: {ref_path}")
        if not os.path.exists(custom_path):
            raise FileNotFoundError(f"自定义算子代码不存在: {custom_path}")

        with open(custom_path) as f:
            exec(f.read(), self.context)
        with open(ref_path) as f:
            exec(f.read(), self.context)

        torch.manual_seed(1024)
        torch_npu.npu.manual_seed_all(1024)

        inputs = self._to_device(self.context["get_inputs"]())
        init_inputs = self._to_device(self.context["get_init_inputs"]())

        ref_model = self.context["Model"](*init_inputs).to(self.device)
        new_model = self.context["ModelNew"](*init_inputs).to(self.device)

        with torch.no_grad():
            ref_out = ref_model(*inputs)
            new_out = new_model(*inputs)
        torch_npu.npu.synchronize()

        ref_list = ref_out if isinstance(ref_out, (list, tuple)) else [ref_out]
        new_list = new_out if isinstance(new_out, (list, tuple)) else [new_out]

        raw_inputs = self.context["get_inputs"]()
        if not isinstance(raw_inputs, list):
            raw_inputs = [raw_inputs]

        return {
            "ref_outputs": [t.cpu() for t in ref_list],
            "new_outputs": [t.cpu() for t in new_list],
            "input_tensors": raw_inputs,
        }

    def _to_device(self, data):
        if isinstance(data, list):
            return [self._to_device(x) for x in data]
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return data


# ============================================================
# Diff 数值分析引擎 (L0-L4 + L8 语义加权)
# ============================================================

class DiffAnalyzer:
    """OpenOps→OpGen 适配说明:
    OpenOps 原版使用硬编码 ATOL = RTOL = 1e-2 做粗筛 mismatch 阈值。
    在 AscendOpGenAgent 中,**最终精度判定的权威源**是
    `utils/verification_ascendc.py` 的 MERE/MARE
    (`_check_precision_mere_mare`,准则为 MERE < threshold 且
    MARE < 10*threshold)。本类的内部阈值仅用于内部排序
    (哪些元素值得列为 worst、哪些尾块 mismatch 率值得标注),
    不参与最终 PASS/FAIL 判断 —— 那由 ascend-kernel-developer 的
    Phase 4 / Phase 6 流程承担。
    阈值改为按 golden.dtype 从 PRECISION_THRESHOLDS 取值,
    与 OpGen 体系对齐;import 失败时退化为 1e-2 兼容旧行为。
    """

    def __init__(self, op_type_info: dict = None):
        self.op_type_info = op_type_info or {"op_type": "unknown", "pattern_priority": []}
        # 阈值来源:OpGen utils/verification_ascendc.py PRECISION_THRESHOLDS
        try:
            import os as _os
            import sys as _sys
            _opgen_root = _os.path.abspath(_os.path.join(
                _os.path.dirname(__file__), "..", "..", "..", ".."))
            if _opgen_root not in _sys.path:
                _sys.path.insert(0, _opgen_root)
            from utils.verification_ascendc import PRECISION_THRESHOLDS as _PT
            self._dtype_thresholds = dict(_PT)
        except Exception:
            self._dtype_thresholds = {}

    def _threshold_for(self, dtype) -> float:
        # 用 OpGen MERE 阈值表;未命中时回退到 1e-2(仅用于内部排序,不影响最终判定)
        return float(self._dtype_thresholds.get(str(dtype), 1e-2))

    def analyze(self, golden: np.ndarray, actual: np.ndarray) -> dict:
        thr = self._threshold_for(golden.dtype)
        abs_diff = np.abs(golden - actual)
        threshold = thr + thr * np.abs(golden)
        mismatch_mask = abs_diff > threshold

        return {
            "pass_fail": bool(np.sum(mismatch_mask) == 0),
            "basic_stats": self._basic_stats(golden, actual, abs_diff, mismatch_mask),
            "error_distribution": self._error_distribution(golden, actual, abs_diff),
            "value_range": self._value_range(golden, actual),
            "pattern_hint": self._classify_pattern(golden, actual, abs_diff, mismatch_mask),
            "worst_elements": self._worst_elements(abs_diff, golden, actual, top_k=10),
            "tail_analysis": self._tail_analysis(abs_diff, mismatch_mask, golden.shape),
            "dimension_analysis": self._dimension_analysis(abs_diff, mismatch_mask, golden.shape),
            "L5_intermediate": None,
            "L7_code_mapping": None,
            "L8_op_type": self.op_type_info.get("op_type", "unknown"),
        }

    # ---- L1 ----

    def _basic_stats(self, golden, actual, abs_diff, mismatch_mask) -> dict:
        total = max(golden.size, 1)
        n = int(np.sum(mismatch_mask))
        return {
            "max_abs_diff": float(np.max(abs_diff)),
            "mean_abs_diff": float(np.mean(abs_diff)),
            "median_abs_diff": float(np.median(abs_diff)),
            "p99_abs_diff": float(np.percentile(abs_diff, 99)),
            "num_mismatched": n,
            "total_elements": int(golden.size),
            "mismatch_ratio": n / total,
            "match_rate": 1.0 - n / total,
        }

    def _error_distribution(self, golden, actual, abs_diff) -> dict:
        diff_signed = (actual - golden).flatten()
        abs_flat = abs_diff.flatten()
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        quartile = {f"p{p}": float(np.percentile(abs_flat, p)) for p in percentiles}

        golden_flat = golden.flatten()
        safe = np.abs(golden_flat) > 1e-7
        rel = np.zeros_like(abs_flat)
        if np.any(safe):
            rel[safe] = abs_flat[safe] / np.abs(golden_flat[safe])

        n_pos = int(np.sum(diff_signed > 0))
        n_neg = int(np.sum(diff_signed < 0))
        return {
            "abs_diff_percentiles": quartile,
            "rel_error_mean": float(np.mean(rel)),
            "rel_error_max": float(min(np.max(rel), 1e6)),
            "sign_analysis": {
                "positive_count": n_pos,
                "negative_count": n_neg,
                "zero_count": int(np.sum(diff_signed == 0)),
                "bias_direction": "positive" if n_pos > n_neg * 1.5
                                  else "negative" if n_neg > n_pos * 1.5
                                  else "balanced",
                "mean_signed_diff": float(np.mean(diff_signed)),
            },
        }

    def _value_range(self, golden, actual) -> dict:
        def _s(arr, name):
            return {
                f"{name}_min": float(np.min(arr)), f"{name}_max": float(np.max(arr)),
                f"{name}_mean": float(np.mean(arr)), f"{name}_std": float(np.std(arr)),
                f"{name}_has_nan": bool(np.any(np.isnan(arr))),
                f"{name}_has_inf": bool(np.any(np.isinf(arr))),
                f"{name}_nan_count": int(np.sum(np.isnan(arr))),
                f"{name}_inf_count": int(np.sum(np.isinf(arr))),
            }
        r = {}
        r.update(_s(golden, "golden"))
        r.update(_s(actual, "actual"))
        return r

    # ---- L2/L3: Pattern Hint (含语义加权) ----

    def _classify_pattern(self, golden, actual, abs_diff, mismatch_mask) -> dict:
        shape = golden.shape
        total = max(golden.size, 1)
        mismatch_ratio = np.sum(mismatch_mask) / total
        hints = []

        nan_n = int(np.sum(np.isnan(actual)))
        inf_n = int(np.sum(np.isinf(actual)))
        if nan_n > 0 or inf_n > 0:
            hints.append({"pattern": "nan_inf_contamination", "confidence": 0.95,
                          "evidence": f"NPU 输出含 NaN={nan_n}, Inf={inf_n}"})

        if mismatch_ratio > 0.9:
            dv = (actual - golden).flatten()
            dm, ds = float(np.mean(dv)), float(np.std(dv))
            if ds < 0.1 * abs(dm) and abs(dm) > 1e-3:
                hints.append({"pattern": "uniform_offset", "confidence": 0.85,
                              "evidence": f"全局偏移 mean={dm:.6f}, std={ds:.6f}"})
            else:
                hints.append({"pattern": "all_wrong", "confidence": 0.9,
                              "evidence": f"mismatch={mismatch_ratio:.1%}"})

        t = self._check_tail_spike(mismatch_mask, shape)
        if t:
            hints.append(t)
        if len(shape) >= 2:
            c = self._check_dim_concentration(mismatch_mask, shape)
            if c:
                hints.append(c)
        m = self._check_magnitude_correlation(golden, mismatch_mask)
        if m:
            hints.append(m)
        b = self._check_boundary_concentration(mismatch_mask, shape)
        if b:
            hints.append(b)

        if not hints:
            hints.append({"pattern": "scattered", "confidence": 0.4,
                          "evidence": f"mismatch 分散, 比例={mismatch_ratio:.2%}"})

        # 语义加权
        plist = self.op_type_info.get("pattern_priority", [])
        if plist:
            for h in hints:
                if h["pattern"] in plist:
                    rank = plist.index(h["pattern"])
                    boost = max(0, 0.1 - rank * 0.03)
                    h["confidence"] = min(0.99, h["confidence"] + boost)
                    h["semantic_boosted"] = True

        hints.sort(key=lambda h: h["confidence"], reverse=True)
        return {
            "primary_hint": hints[0]["pattern"],
            "primary_confidence": hints[0]["confidence"],
            "primary_evidence": hints[0]["evidence"],
            "all_hints": hints,
        }

    def _check_tail_spike(self, mm, shape):
        ld = shape[-1]
        for ts in [8, 16, 32, 64, 128, 256]:
            if ld <= ts:
                continue
            tl = ld % ts
            if tl == 0:
                continue
            t_s = tuple([slice(None)] * (len(shape) - 1) + [slice(-tl, None)])
            b_s = tuple([slice(None)] * (len(shape) - 1) + [slice(0, -tl)])
            tr = float(np.mean(mm[t_s]))
            br = float(np.mean(mm[b_s]))
            if tr > 0.3 and (br < 0.01 or tr > br * 5):
                return {"pattern": "tail_spike",
                        "confidence": min(0.9, 0.6 + (tr - br)),
                        "evidence": f"last_dim={ld}, tile={ts}, 尾块({tl})={tr:.1%}, 主体={br:.1%}",
                        "detail": {"tile_size": ts, "tail_len": tl, "tail_rate": tr, "body_rate": br}}
        return None

    def _check_dim_concentration(self, mm, shape):
        for dim in range(len(shape)):
            if shape[dim] <= 1:
                continue
            rates = [float(np.mean(mm[tuple([slice(None)]*dim + [i] + [slice(None)]*(len(shape)-dim-1))]))
                     for i in range(shape[dim])]
            ra = np.array(rates)
            if np.max(ra) > 0.5 and np.min(ra) < 0.1:
                bad = [int(i) for i in np.where(ra > 0.3)[0]]
                return {"pattern": "dimension_concentration", "confidence": 0.8,
                        "evidence": f"dim={dim}(size={shape[dim]}), 索引 {bad} mismatch 偏高",
                        "detail": {"dim": dim, "bad_indices": bad,
                                   "rates": [round(r, 4) for r in rates]}}
        return None

    def _check_magnitude_correlation(self, golden, mm):
        gf, mf = np.abs(golden.flatten()), mm.flatten()
        if np.sum(mf) < 10 or np.sum(~mf) < 10:
            return None
        mmv, nmv = float(np.mean(gf[mf])), float(np.mean(gf[~mf]))
        if nmv < 1e-10:
            return None
        ratio = mmv / nmv
        if ratio > 3.0:
            return {"pattern": "magnitude_correlated",
                    "confidence": min(0.85, 0.5 + (ratio - 3) * 0.05),
                    "evidence": f"mismatch 位置 |golden| 均值({mmv:.4f}) 是匹配位置({nmv:.4f})的 {ratio:.1f} 倍（量级相关性，非误差大小）"}
        if ratio < 0.3:
            return {"pattern": "magnitude_correlated",
                    "confidence": min(0.85, 0.5 + (1/ratio - 3) * 0.05),
                    "evidence": f"mismatch 集中在小值区域"}
        return None

    def _check_boundary_concentration(self, mm, shape):
        if len(shape) < 2:
            return None
        tm = int(np.sum(mm))
        if tm == 0:
            return None
        bm, bt = 0, 0
        for dim in range(len(shape)):
            if shape[dim] <= 2:
                continue
            for edge in [0, shape[dim] - 1]:
                s = tuple([slice(None)]*dim + [edge] + [slice(None)]*(len(shape)-dim-1))
                bm += int(np.sum(mm[s]))
                bt += mm[s].size
        if bt > 0 and tm > 0 and bm / tm > 0.6 and bm / bt > 0.2:
            return {"pattern": "boundary_concentration", "confidence": 0.75,
                    "evidence": f"边界含 {bm/tm:.0%} 的 mismatch"}
        return None

    # ---- L4 ----

    def _worst_elements(self, abs_diff, golden, actual, top_k=10):
        flat = abs_diff.flatten()
        k = min(top_k, len(flat))
        idx = np.argpartition(flat, -k)[-k:]
        idx = idx[np.argsort(flat[idx])[::-1]]
        return [{"index": list(map(int, np.unravel_index(i, abs_diff.shape))),
                 "abs_diff": float(flat[i]),
                 "golden_value": float(golden.flat[i]),
                 "actual_value": float(actual.flat[i]),
                 "L7_gm_offset": None, "L7_source_line": None} for i in idx]

    def _tail_analysis(self, abs_diff, mm, shape):
        ld = shape[-1]
        results = {}
        for ts in [8, 16, 32, 64, 128, 256]:
            tl = ld % ts
            if tl == 0 or ld <= ts:
                continue
            t_s = tuple([slice(None)]*(len(shape)-1) + [slice(-tl, None)])
            b_s = tuple([slice(None)]*(len(shape)-1) + [slice(0, -tl)])
            results[f"tile_{ts}"] = {
                "tail_len": tl,
                "tail_mean_diff": float(np.mean(abs_diff[t_s])),
                "body_mean_diff": float(np.mean(abs_diff[b_s])),
                "tail_max_diff": float(np.max(abs_diff[t_s])),
                "body_max_diff": float(np.max(abs_diff[b_s])),
                "tail_mismatch_rate": float(np.mean(mm[t_s])),
                "body_mismatch_rate": float(np.mean(mm[b_s])),
            }
        if not results:
            return {"last_dim": ld, "note": "last_dim 是常见 tile 的整数倍"}
        results["last_dim"] = ld
        return results

    def _dimension_analysis(self, abs_diff, mm, shape):
        analysis = []
        for dim in range(len(shape)):
            if shape[dim] <= 1:
                continue
            rates, diffs = [], []
            for i in range(shape[dim]):
                s = tuple([slice(None)]*dim + [i] + [slice(None)]*(len(shape)-dim-1))
                rates.append(float(np.mean(mm[s])))
                diffs.append(float(np.mean(abs_diff[s])))
            analysis.append({
                "dim": dim, "size": shape[dim],
                "mismatch_rate_min": float(np.min(rates)),
                "mismatch_rate_max": float(np.max(rates)),
                "mismatch_rate_std": float(np.std(rates)),
                "mean_diff_min": float(np.min(diffs)),
                "mean_diff_max": float(np.max(diffs)),
                "per_index_rates": [round(r, 4) for r in rates] if shape[dim] <= 64 else None,
            })
        return analysis


# ============================================================
# L6: 内存布局分析
# ============================================================

class MemoryLayoutAnalyzer:

    TILE_SIZES = [8, 16, 32, 64, 128, 256]

    def analyze_tensors(self, tensors: list, label: str = "input") -> list:
        results = []
        for i, t in enumerate(tensors):
            if not isinstance(t, torch.Tensor):
                continue
            results.append(self._analyze_single(t, f"{label}_{i}"))
        return results

    def _analyze_single(self, t: torch.Tensor, name: str) -> dict:
        info = {
            "name": name, "shape": list(t.shape), "stride": list(t.stride()),
            "dtype": str(t.dtype), "is_contiguous": t.is_contiguous(),
            "storage_offset": t.storage_offset(),
            "element_size_bytes": t.element_size(),
        }
        last_dim = t.shape[-1] if t.ndim > 0 else 0
        info["last_dim_alignment"] = {
            f"tile_{ts}": {"remainder": last_dim % ts, "aligned": last_dim % ts == 0}
            for ts in self.TILE_SIZES
        }
        return info


# ============================================================
# L5: 中间结果探测 (TODO - Phase 2)
# ============================================================

class IntermediateProbe:
    """
    设计存根（不实现）

    原理: 在 {op_name}.cpp 的 aclnn 调用链中注入探针,
    dump 每个步骤的输入/输出 tensor, 定位误差引入点。

    不实现原因:
    - 需要修改 pybind C++ 代码并重编译，且每个算子步骤各不相同，无法通用化
    - 探针修改可能影响 buffer 对齐，改变精度问题的表现（Heisenbug 效应）
    - 由 Sub-step 2.3 的 L7 Agent 手动映射替代：
      worst element index → tiling 参数静态推算 → Core/block/K-Step 定位

    接口定义 (仅供参考，不调用):
    {
        "available": true,
        "error_origin_step": 2,
        "steps": [
            {"step_index": 0, "operation": "aclnnSub", "max_diff_at_output": 0.0},
            {"step_index": 1, "operation": "aclnnMul", "max_diff_at_output": 0.001},
            {"step_index": 2, "operation": "aclnnReduceSum", "max_diff_at_output": 0.015,
             "error_introduced_here": true}
        ],
        "error_magnification": 15.0
    }
    """

    def probe(self, op_name: str, output_path: str) -> dict | None:
        return None


# ============================================================
# L7: 代码位置映射 (TODO - Phase 2 / Agent)
# ============================================================

class CodeMapper:
    """
    TODO: 将数值误差位置映射到具体代码行号

    原理: worst_elements 的 numpy index → GM offset → tiling block → 代码行

    实现难点: 需要理解 tiling 策略, kernel 循环结构多样, 难以通用化。
    建议由 Agent 在审计时完成。

    接口定义 (未来实现时遵循):
    {
        "available": true,
        "error_locations": [
            {"index": [0, 45], "gm_offset": 18980, "block_type": "tail_block",
             "source_file": "op_kernel.cpp", "line_number": 234,
             "code_snippet": "ReduceSum(workBuf, src, count);"}
        ]
    }
    """

    def map(self, worst_elements: list, kernel_path: str) -> dict | None:
        return None


# ============================================================
# 历史轮次对比
# ============================================================

class HistoryComparator:

    def __init__(self, tuning_dir: str, current_attempt: int):
        self.tuning_dir = tuning_dir
        self.current_attempt = current_attempt

    def load_history(self) -> list:
        history = []
        for i in range(self.current_attempt):
            p = os.path.join(self.tuning_dir, "history", f"attempt_{i}", "forensics_report.json")
            if os.path.exists(p):
                with open(p) as f:
                    history.append({"attempt": i, "report": json.load(f)})
        return history

    def build_trend(self, current_report: dict) -> dict | None:
        history = self.load_history()
        if not history:
            return None
        trend = []
        for h in history:
            r = h["report"]
            if r.get("status") != "completed" or not r.get("outputs"):
                continue
            s = r["outputs"][0].get("basic_stats", {})
            trend.append({"attempt": h["attempt"], "mismatch_ratio": s.get("mismatch_ratio"),
                          "max_abs_diff": s.get("max_abs_diff"), "primary_hint": r.get("primary_hint")})
        if current_report.get("outputs"):
            cs = current_report["outputs"][0].get("basic_stats", {})
            trend.append({"attempt": self.current_attempt, "mismatch_ratio": cs.get("mismatch_ratio"),
                          "max_abs_diff": cs.get("max_abs_diff"), "primary_hint": current_report.get("primary_hint")})
        return {"num_attempts": len(trend), "trend": trend,
                "mismatch_improving": self._improving(trend)}

    def _improving(self, trend):
        ratios = [t["mismatch_ratio"] for t in trend if t["mismatch_ratio"] is not None]
        return ratios[-1] < ratios[-2] if len(ratios) >= 2 else True


# ============================================================
# 多测试案例预留 (TODO)
# ============================================================

class MultiCaseForensics:
    """
    TODO: 当 evaluate.py 扩展为多 test case 后启用。

    接口定义:
    {
        "num_cases": int, "pass_count": int, "fail_count": int,
        "pattern_distribution": {"tail_spike": 3, "all_wrong": 1},
        "shape_dependency": {"passing_shapes": [...], "failing_shapes": [...],
                             "critical_dimension": "last_dim"},
        "minimal_failing_case": {...},
        "tolerance_sensitivity": {"threshold_1e-2": 0.3, "threshold_1e-1": 0.85}
    }
    """

    def analyze_all_cases(self, case_results: list) -> dict:
        raise NotImplementedError("等待 evaluate.py 多案例扩展")


# ============================================================
# 主入口
# ============================================================

class PrecisionForensics:

    def __init__(self, op_name: str, output_path: str, attempt: int = 0):
        self.op_name = op_name
        self.output_path = output_path
        self.attempt = attempt
        self.tuning_dir = os.path.join(output_path, "precision_tuning")
        os.makedirs(self.tuning_dir, exist_ok=True)

    def run(self) -> dict:
        try:
            op_type_info = OperatorTypeDetector().detect(self.op_name, self.output_path)
            data = OperatorExecutor(self.op_name, self.output_path).load_and_execute()

            analyzer = DiffAnalyzer(op_type_info=op_type_info)
            output_reports = []
            for i, (ref_t, new_t) in enumerate(zip(data["ref_outputs"], data["new_outputs"])):
                r = analyzer.analyze(ref_t.float().numpy(), new_t.float().numpy())
                r["output_index"] = i
                r["output_shape"] = list(ref_t.shape)
                r["output_dtype"] = str(ref_t.dtype)
                output_reports.append(r)

            layout = MemoryLayoutAnalyzer()
            l5 = IntermediateProbe().probe(self.op_name, self.output_path)
            l7 = CodeMapper().map(output_reports[0]["worst_elements"] if output_reports else [], "")

            worst = max(output_reports, key=lambda r: r["basic_stats"]["mismatch_ratio"])
            ph = worst["pattern_hint"]
            comparator = HistoryComparator(self.tuning_dir, self.attempt)

            final = {
                "version": "2.0",
                "op_name": self.op_name,
                "attempt": self.attempt,
                "status": "completed",
                "num_outputs": len(output_reports),
                "L0_pass": worst["pass_fail"],
                "outputs": output_reports,
                "L5_intermediate": l5,
                "L6_memory_layout": {
                    "inputs": layout.analyze_tensors(data.get("input_tensors", []), "input"),
                    "outputs": layout.analyze_tensors(data["ref_outputs"], "output"),
                },
                "L7_code_mapping": l7,
                "L8_operator": op_type_info,
                "primary_hint": ph["primary_hint"],
                "primary_confidence": ph["primary_confidence"],
                "primary_evidence": ph["primary_evidence"],
                "all_hints": ph["all_hints"],
                "history_trend": None,
                "multi_case_analysis": None,
                "num_test_cases": 1,
                "available_files": {
                    "reference": os.path.exists(os.path.join(self.output_path, f"{self.op_name}_reference.py")),
                    "dsl": os.path.exists(os.path.join(self.output_path, f"{self.op_name}_dsl.py")),
                    "op_desc": os.path.exists(os.path.join(self.output_path, f"{self.op_name}_op_desc.json")),
                    "functional": os.path.exists(os.path.join(self.output_path, f"{self.op_name}_functional.py")),
                    "custom": os.path.exists(os.path.join(self.output_path, f"{self.op_name}_custom.py")),
                },
            }

            trend = comparator.build_trend(final)
            if trend:
                final["history_trend"] = trend

            report_path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
            with open(report_path, "w") as f:
                json.dump(final, f, indent=2, ensure_ascii=False)

            stats = worst["basic_stats"]
            print(f"[FORENSICS] ✅ 精度取证完成 (attempt={self.attempt})")
            print(f"  op_type: {op_type_info['op_type']} (source={op_type_info['source']})")
            print(f"  primary_hint: {final['primary_hint']} "
                  f"(confidence={final['primary_confidence']:.2f})")
            print(f"  evidence: {final['primary_evidence']}")
            print(f"  mismatch: {stats['mismatch_ratio']:.2%} "
                  f"({stats['num_mismatched']}/{stats['total_elements']})")
            print(f"  max_diff: {stats['max_abs_diff']:.6f}")
            print(f"  L5: {'available' if l5 else 'not available (TODO)'}")
            print(f"  L7: {'available' if l7 else 'not available (TODO/Agent)'}")
            if trend:
                print(f"  趋势: {'↓ 改善中' if trend['mismatch_improving'] else '↑ 未改善'}")
            print(f"  report: {report_path}")
            return final

        except Exception as e:
            err = {"version": "2.0", "op_name": self.op_name, "attempt": self.attempt,
                   "status": "error", "error": str(e), "traceback": traceback.format_exc()}
            rp = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
            with open(rp, "w") as f:
                json.dump(err, f, indent=2, ensure_ascii=False)
            print(f"[FORENSICS] ❌ 取证失败: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="精度取证数值分析")
    parser.add_argument("op_name", help="算子名称")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()
    PrecisionForensics(args.op_name, args.output_path, args.attempt).run()


if __name__ == "__main__":
    main()
