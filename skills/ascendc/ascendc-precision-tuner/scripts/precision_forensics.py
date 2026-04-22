#!/usr/bin/env python3
"""
precision_forensics.py — AscendOpGenAgent adapted precision forensics

Adapts OpenOps precision-tuning forensics to AscendOpGenAgent structure:
- Uses {task_dir}/model.py + {task_dir}/model_new_ascendc.py
- Outputs to {task_dir}/precision_tuning/forensics_report_{attempt}.json
- Provides L0-L8 structured analysis for precision debugging

Usage:
    python3 precision_forensics.py <task_name> [--workdir <path>] [--attempt <N>]

Dependencies: torch, numpy
"""

import argparse
import copy
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# AscendOpGenAgent-compatible model loader (from verification_ascendc.py)
# ============================================================

def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_model_class(module, preferred_name: str):
    candidate = getattr(module, preferred_name, None)
    if inspect.isclass(candidate) and issubclass(candidate, nn.Module):
        return candidate
    for _, value in vars(module).items():
        if inspect.isclass(value) and issubclass(value, nn.Module) and value is not nn.Module:
            return value
    raise AttributeError(f"No nn.Module subclass found in {module.__file__}")


def _clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _get_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_input_groups(module):
    if hasattr(module, "get_input_groups"):
        input_groups = module.get_input_groups()
        if isinstance(input_groups, list) and input_groups:
            return input_groups
    if hasattr(module, "get_inputs"):
        inputs = module.get_inputs()
        if isinstance(inputs, list):
            return [inputs]
    raise AttributeError(f"Neither get_input_groups() nor get_inputs() found")


def _normalize_output(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_normalize_output(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_output(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize_output(item) for key, item in value.items()}
    return value


# ============================================================
# Operator type detection (L8)
# ============================================================

class OperatorTypeDetector:
    OP_TYPE_PATTERN_PRIORITY = {
        "reduction": ["magnitude_correlated", "tail_spike", "uniform_offset", "scattered"],
        "pooling": ["tail_spike", "boundary_concentration", "uniform_offset"],
        "loss": ["magnitude_correlated", "uniform_offset", "nan_inf_contamination"],
        "matmul": ["dimension_concentration", "scattered", "magnitude_correlated"],
        "activation": ["nan_inf_contamination", "uniform_offset", "boundary_concentration"],
        "normalization": ["nan_inf_contamination", "magnitude_correlated", "tail_spike"],
        "convolution": ["dimension_concentration", "boundary_concentration", "tail_spike"],
    }

    def detect(self, op_name: str) -> dict:
        op_type = self._infer_from_name(op_name)
        return {
            "op_type": op_type,
            "source": "name_inference",
            "pattern_priority": self.OP_TYPE_PATTERN_PRIORITY.get(op_type, []),
            "attributes": {},
        }

    def _infer_from_name(self, op_name: str) -> str:
        name = op_name.lower()
        rules = [
            (["pool", "avg_pool", "max_pool"], "pooling"),
            (["reduce", "sum", "mean", "prod", "cumsum"], "reduction"),
            (["loss", "mse", "cross_entropy", "nll"], "loss"),
            (["matmul", "bmm", "linear", "gemm"], "matmul"),
            (["relu", "gelu", "silu", "sigmoid", "tanh", "leaky", "softmax", "log_softmax"], "activation"),
            (["norm", "layer_norm", "batch_norm", "group_norm", "rms_norm"], "normalization"),
            (["conv", "conv2d", "conv1d"], "convolution"),
        ]
        for keywords, t in rules:
            if any(kw in name for kw in keywords):
                return t
        return "unknown"


# ============================================================
# Diff analysis engine (L0-L4 + L6 + L8)
# ============================================================

class DiffAnalyzer:
    ATOL = 1e-02
    RTOL = 1e-02

    def __init__(self, op_type_info: dict = None):
        self.op_type_info = op_type_info or {"op_type": "unknown", "pattern_priority": []}

    def analyze(self, golden: np.ndarray, actual: np.ndarray) -> dict:
        abs_diff = np.abs(golden - actual)
        threshold = self.ATOL + self.RTOL * np.abs(golden)
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
            "L8_op_type": self.op_type_info.get("op_type", "unknown"),
        }

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

    def _classify_pattern(self, golden, actual, abs_diff, mismatch_mask) -> dict:
        shape = golden.shape
        total = max(golden.size, 1)
        mismatch_ratio = np.sum(mismatch_mask) / total
        hints = []

        nan_n = int(np.sum(np.isnan(actual)))
        inf_n = int(np.sum(np.isinf(actual)))
        if nan_n > 0 or inf_n > 0:
            hints.append({"pattern": "nan_inf_contamination", "confidence": 0.95,
                          "evidence": f"NPU output contains NaN={nan_n}, Inf={inf_n}"})

        if mismatch_ratio > 0.9:
            dv = (actual - golden).flatten()
            dm, ds = float(np.mean(dv)), float(np.std(dv))
            if ds < 0.1 * abs(dm) and abs(dm) > 1e-3:
                hints.append({"pattern": "uniform_offset", "confidence": 0.85,
                              "evidence": f"global offset mean={dm:.6f}, std={ds:.6f}"})
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
                          "evidence": f"mismatch scattered, ratio={mismatch_ratio:.2%}"})

        # Semantic boosting
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
                        "evidence": f"last_dim={ld}, tile={ts}, tail({tl})={tr:.1%}, body={br:.1%}",
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
                        "evidence": f"dim={dim}(size={shape[dim]}), indices {bad} mismatch high",
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
                    "evidence": f"mismatch mean({mmv:.4f}) is {ratio:.1f}x normal({nmv:.4f})"}
        if ratio < 0.3:
            return {"pattern": "magnitude_correlated",
                    "confidence": min(0.85, 0.5 + (1/ratio - 3) * 0.05),
                    "evidence": f"mismatch concentrated in small-value regions"}
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
                    "evidence": f"boundary contains {bm/tm:.0%} of mismatch"}
        return None

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
            return {"last_dim": ld, "note": "last_dim is integer multiple of common tile sizes"}
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
# L6: Memory layout analysis
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
# History comparator
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
# Main forensics runner
# ============================================================

class PrecisionForensics:
    def __init__(self, task_name: str, workdir: str = ".", attempt: int = 0):
        self.task_name = task_name
        self.workdir = Path(workdir).resolve()
        self.task_dir = self.workdir / task_name
        self.attempt = attempt
        self.tuning_dir = self.task_dir / "precision_tuning"
        self.tuning_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        try:
            ref_path = self.task_dir / "model.py"
            cand_path = self.task_dir / "model_new_ascendc.py"

            if not ref_path.is_file():
                raise FileNotFoundError(f"Reference model not found: {ref_path}")
            if not cand_path.is_file():
                raise FileNotFoundError(f"Candidate model not found: {cand_path}")

            ref_module = _load_module(ref_path, f"{self.task_name}_ref_model")
            cand_module = _load_module(cand_path, f"{self.task_name}_ascendc_model")

            ref_cls = _find_model_class(ref_module, "Model")
            cand_cls = _find_model_class(cand_module, "ModelNew")

            torch.manual_seed(0)
            if hasattr(cand_module, "get_init_inputs"):
                init_inputs = cand_module.get_init_inputs()
            else:
                init_inputs = getattr(ref_module, "get_init_inputs", lambda: [])()
            input_groups = _get_input_groups(ref_module)
            device = _get_device()

            ref_model = ref_cls(*_clone_value(init_inputs)).to(device).eval()
            cand_model = cand_cls(*_clone_value(init_inputs)).to(device).eval()

            op_type_info = OperatorTypeDetector().detect(self.task_name)
            analyzer = DiffAnalyzer(op_type_info=op_type_info)
            layout = MemoryLayoutAnalyzer()

            output_reports = []
            input_summaries = []
            all_ok = True

            for index, inputs in enumerate(input_groups):
                ref_inputs = _move_to_device(_clone_value(inputs), device)
                cand_inputs = _move_to_device(_clone_value(inputs), device)
                input_summaries.extend([f"inputs[{index}]: Tensor(shape={list(t.shape)}, dtype={t.dtype})"
                                        for t in ref_inputs if isinstance(t, torch.Tensor)])

                with torch.no_grad():
                    ref_out = ref_model(*ref_inputs)
                    cand_out = cand_model(*cand_inputs)

                ref_out = _normalize_output(ref_out)
                cand_out = _normalize_output(cand_out)

                # Handle single tensor output
                if isinstance(ref_out, torch.Tensor):
                    ref_out = [ref_out]
                    cand_out = [cand_out]
                elif isinstance(ref_out, (list, tuple)):
                    ref_out = list(ref_out)
                    cand_out = list(cand_out)
                else:
                    raise TypeError(f"Unsupported output type: {type(ref_out)}")

                for i, (ref_t, new_t) in enumerate(zip(ref_out, cand_out)):
                    if ref_t.shape != new_t.shape:
                        raise ValueError(f"Shape mismatch at output[{i}]: ref={ref_t.shape}, cand={new_t.shape}")
                    r = analyzer.analyze(ref_t.float().numpy(), new_t.float().numpy())
                    r["output_index"] = i
                    r["output_shape"] = list(ref_t.shape)
                    r["output_dtype"] = str(ref_t.dtype)
                    output_reports.append(r)
                    all_ok = all_ok and r["pass_fail"]

            worst = max(output_reports, key=lambda r: r["basic_stats"]["mismatch_ratio"])
            ph = worst["pattern_hint"]
            comparator = HistoryComparator(str(self.tuning_dir), self.attempt)

            final = {
                "version": "2.0",
                "task_name": self.task_name,
                "attempt": self.attempt,
                "status": "completed",
                "num_outputs": len(output_reports),
                "L0_pass": all_ok,
                "outputs": output_reports,
                "L6_memory_layout": {
                    "inputs": layout.analyze_tensors(input_groups[0] if input_groups else [], "input"),
                },
                "L8_operator": op_type_info,
                "primary_hint": ph["primary_hint"],
                "primary_confidence": ph["primary_confidence"],
                "primary_evidence": ph["primary_evidence"],
                "all_hints": ph["all_hints"],
                "history_trend": None,
                "num_test_cases": len(input_groups),
                "available_files": {
                    "reference": ref_path.is_file(),
                    "candidate": cand_path.is_file(),
                    "kernel_dir": (self.task_dir / "kernel").is_dir(),
                },
            }

            trend = comparator.build_trend(final)
            if trend:
                final["history_trend"] = trend

            report_path = self.tuning_dir / f"forensics_report_{self.attempt}.json"
            with open(report_path, "w") as f:
                json.dump(final, f, indent=2, ensure_ascii=False)

            stats = worst["basic_stats"]
            print(f"[FORENSICS] Precision forensics completed (attempt={self.attempt})")
            print(f"  op_type: {op_type_info['op_type']} (source={op_type_info['source']})")
            print(f"  primary_hint: {final['primary_hint']} (confidence={final['primary_confidence']:.2f})")
            print(f"  evidence: {final['primary_evidence']}")
            print(f"  mismatch: {stats['mismatch_ratio']:.2%} ({stats['num_mismatched']}/{stats['total_elements']})")
            print(f"  max_diff: {stats['max_abs_diff']:.6f}")
            if trend:
                print(f"  trend: {'improving' if trend['mismatch_improving'] else 'not improving'}")
            print(f"  report: {report_path}")
            return final

        except Exception as e:
            err = {"version": "2.0", "task_name": self.task_name, "attempt": self.attempt,
                   "status": "error", "error": str(e), "traceback": traceback.format_exc()}
            rp = self.tuning_dir / f"forensics_report_{self.attempt}.json"
            with open(rp, "w") as f:
                json.dump(err, f, indent=2, ensure_ascii=False)
            print(f"[FORENSICS] Forensics failed: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AscendOpGenAgent precision forensics")
    parser.add_argument("task_name", help="Task directory name")
    parser.add_argument("--workdir", default=".", help="Path to AscendOpGenAgent workdir (default: current dir)")
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()
    PrecisionForensics(args.task_name, args.workdir, args.attempt).run()


if __name__ == "__main__":
    main()
