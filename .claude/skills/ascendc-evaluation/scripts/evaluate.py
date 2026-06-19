#!/usr/bin/env python3
"""
AscendC 自定义算子评估工具

用于评估自定义算子的正确性和性能表现。

所有性能评测统一使用 advanced 模式（带预热和清缓存）以确保测量精度。

命令行用法:
    python evaluate.py <op_name>
    python evaluate.py <op_name> --work-dir /path/to/work_dir

示例:
    python evaluate.py Add
    python evaluate.py Conv1d --work-dir /path/to/Conv1d_evo/round_1/parallel_0

API 用法:
    from evaluate import AscendBackend

    # 初始化后端
    backend = AscendBackend(eval_code, ref_code)

    # 1. 精度评估
    success, message = backend.evaluate_correctness()
    # 返回: (bool, str) - 是否通过及详细信息

    # 2. 高级性能测试（带预热和清缓存）
    ref_time, ref_perf_data, ref_profile_dir, ref_cv_pct, custom_time, custom_perf_data, custom_profile_dir, custom_cv_pct = backend.compare_performance_advanced(
        profile_root=Path("output/{Operator_name}/profiling"),
        num_trials=50,
        task_type="vector"
    )
    # 返回: (float, list, Path, float, float, list, Path, float) - 耗时（us）、性能数据、profiling 子目录、cv_pct

输入代码要求:
    评估代码必须包含以下组件:
    - Model: 参考实现类 (torch.nn.Module)
    - ModelNew: 自定义算子实现类 (torch.nn.Module)
    - get_inputs(): 返回测试输入数据的函数
    - get_init_inputs(): 返回模型初始化参数的函数

目录结构:
    output/<op_name>/
        <op_name>_reference.py  # 参考代码
        <op_name>_custom.py     # 自定义算子代码
        test_cases.csv          # 测试用例文件（自动生成）
        profiling/              # Profiling 输出目录
            Model_*/            # 参考模型 profiling
            ModelNew_*/         # 自定义算子 profiling
                op_summary_*.csv # Profiling 结果（自动过滤预热算子）

环境要求:
    - vendors/customize/op_api/lib/ 库文件必须存在
    - 需要设置 ASCEND_CUSTOM_OPP_PATH 环境变量
"""

import os
import sys
import csv
import json
import logging
import argparse
import itertools
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import torch
import torch_npu
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

# Per-operator pybind_lib isolation for parallel evaluation
def _inject_pybind_lib(work_dir_path):
    """Add work_dir/pybind_lib/ to sys.path[0] so operator-specific .so takes priority."""
    pybind_lib = os.path.join(str(work_dir_path), "pybind_lib")
    if os.path.isdir(pybind_lib) and pybind_lib not in sys.path:
        sys.path.insert(0, pybind_lib)
        # Force reimport if already cached
        if "custom_ops_lib" in sys.modules:
            del sys.modules["custom_ops_lib"]
        logging.info(f"Injected pybind_lib: {pybind_lib}")

from AscendPerformanceTest import (
    MsprofProfiler,
    AdvancedPerformanceEngine,
    extract_prof_rows_tasktype,
    extract_simple_perf_info
)
from precision import (
    dual_inspect, ComponentResult,
)
from constants import DEFAULT_TOLERANCES, DEFAULT_ULP_CONFIG
from perf import PerfResult, measure_time


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)


def _normalize_case_value(value: Any) -> str:
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, (list, tuple)):
        return str(list(value))
    return str(value)


def _expand_case_spec(case_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = []

    raw_cases = case_spec.get("cases") or []
    for case in raw_cases:
        if isinstance(case, dict):
            cases.append({k: _normalize_case_value(v) for k, v in case.items()})

    grid = case_spec.get("grid") or {}
    if isinstance(grid, dict) and grid:
        keys = list(grid.keys())
        values_list = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
        for combo in itertools.product(*values_list):
            case = {k: _normalize_case_value(v) for k, v in zip(keys, combo)}
            cases.append(case)

    return cases


def _assign_case_ids(cases: List[Dict[str, Any]], start_id: int, existing_ids: set) -> List[Dict[str, Any]]:
    next_id = start_id
    for case in cases:
        if "case_id" in case:
            try:
                case_id = int(case["case_id"])
            except Exception:
                case_id = None
        else:
            case_id = None

        if case_id is None or case_id in existing_ids:
            while next_id in existing_ids:
                next_id += 1
            case_id = next_id
            next_id += 1

        case["case_id"] = case_id
        existing_ids.add(case_id)

    return cases


def _extract_cases_from_op_desc(op_desc_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    从 op_desc.json 中提取测试用例信息。

    Args:
        op_desc_path: op_desc.json 文件路径

    Returns:
        测试用例列表，如果无法提取则返回 None
    """
    try:
        with open(op_desc_path, "r", encoding="utf-8") as f:
            op_desc = json.load(f)
    except Exception as e:
        logging.warning(f"Failed to read op_desc.json: {e}")
        return None

    shape_info = op_desc.get("shape_info")
    if not shape_info:
        logging.warning("No shape_info found in op_desc.json")
        return None

    input_shapes = shape_info.get("input_shapes")
    if not input_shapes:
        logging.warning("No input_shapes found in op_desc.json shape_info")
        return None

    case_data: Dict[str, Any] = {"case_id": 0}

    for i, inp in enumerate(input_shapes):
        shape = inp.get("shape")
        dtype = inp.get("dtype")
        if shape is not None:
            case_data[f"var{i}_shape"] = str(list(shape)) if isinstance(shape, (list, tuple)) else str(shape)
        if dtype is not None:
            case_data[f"var{i}_dtype"] = str(dtype).replace("torch.", "")

    # 提取 attributes 作为额外参数
    attributes = op_desc.get("attributes")
    if isinstance(attributes, dict):
        for key, value in attributes.items():
            if isinstance(value, (int, float, str, bool)):
                case_data[key] = str(value)

    return [case_data]


def generate_test_cases_csv(
    project_root: Path,
    op_name: str,
    eval_code: str,
    ref_code: str,
    case_spec: Optional[Dict[str, Any]] = None,
    append: bool = False,
    op_desc_path: Optional[Path] = None,
) -> Path:
    """
    从评估代码中提取测试用例信息并生成 test_cases.csv 文件。

    Args:
        project_root: 项目根目录路径
        op_name: 算子名称
        eval_code: 评估代码内容
        ref_code: 参考代码内容

    Returns:
        test_cases.csv 文件路径
    """
    # 创建输出目录
    output_dir = project_root
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "test_cases.csv"

    existing_df = None
    existing_ids = set()
    if append and csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path, delimiter=';')
            if "case_id" in existing_df.columns:
                existing_ids = set(int(x) for x in existing_df["case_id"].dropna().tolist())
        except Exception as e:
            logging.warning(f"Failed to read existing test_cases.csv: {e}")

    if case_spec:
        cases = _expand_case_spec(case_spec)
        if not cases:
            logging.warning("case_spec provided but produced no cases")
            return csv_path

        start_id = int(case_spec.get("case_id_start", 0))
        cases = _assign_case_ids(cases, start_id, existing_ids)

        case_rows = []
        for case in cases:
            case_rows.append({k: _normalize_case_value(v) for k, v in case.items()})

        if existing_df is not None:
            new_df = pd.DataFrame(case_rows)
            merged_df = pd.concat([existing_df, new_df], ignore_index=True)
            merged_df.to_csv(csv_path, sep=';', index=False)
        else:
            new_df = pd.DataFrame(case_rows)
            new_df.to_csv(csv_path, sep=';', index=False)

        logging.info(f"Generated test_cases.csv at {csv_path}")
        return csv_path

    # 尝试从 op_desc.json 提取测试用例
    if op_desc_path and Path(op_desc_path).exists():
        cases = _extract_cases_from_op_desc(Path(op_desc_path))
        if cases:
            cases = _assign_case_ids(cases, 0, existing_ids)

            if existing_df is not None:
                new_df = pd.DataFrame(cases)
                merged_df = pd.concat([existing_df, new_df], ignore_index=True)
                merged_df.to_csv(csv_path, sep=';', index=False)
            else:
                new_df = pd.DataFrame(cases)
                new_df.to_csv(csv_path, sep=';', index=False)

            logging.info(f"Generated test_cases.csv from op_desc.json at {csv_path}")
            return csv_path
        else:
            logging.warning("Failed to extract cases from op_desc.json, falling back to code execution")

    # 解析代码获取输入信息
    eval_context = {}
    ref_context = {}

    try:
        exec(eval_code, eval_context)
        exec(ref_code, ref_context)
    except Exception as e:
        logging.warning(f"Failed to execute code for test case extraction: {e}")
        # 返回默认路径但不创建文件
        return csv_path

    # 获取 get_inputs 函数
    get_inputs = eval_context.get("get_inputs") or ref_context.get("get_inputs")
    if not get_inputs:
        logging.warning("get_inputs function not found in code")
        return csv_path

    # 调用 get_inputs 获取输入数据
    try:
        inputs = get_inputs()
    except Exception as e:
        logging.warning(f"Failed to call get_inputs: {e}")
        return csv_path

    # 解析输入数据，提取变量信息
    test_cases = []

    case_data = {
        "case_id": 0,
    }

    # 处理每个输入张量
    for i, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor):
            # 形状
            case_data[f"var{i}_shape"] = str(list(inp.shape))
            # 数据类型
            case_data[f"var{i}_dtype"] = str(inp.dtype).replace("torch.", "")
        elif isinstance(inp, (int, float, str, bool)):
            case_data[f"var{i}_value"] = str(inp)

    # 添加特殊参数（如果代码中定义了）
    # 检查是否有 batch_size, seq_len 等变量定义
    special_vars = ["batch_size", "seq_len", "dim", "hidden_size", "heads",
                    "head_dim", "eps", "axis", "keepdim", "dtype"]
    for var in special_vars:
        if var in eval_context:
            val = eval_context[var]
            if isinstance(val, (int, float, str, bool)):
                case_data[var] = str(val)
            elif isinstance(val, torch.dtype):
                case_data[var] = str(val).replace("torch.", "")

    test_cases.append(case_data)

    # 写入CSV文件
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if test_cases:
            fieldnames = list(test_cases[0].keys())
            var_cols = [k for k in fieldnames if k.startswith("var")]
            param_cols = [k for k in fieldnames if k not in ["case_id"] and not k.startswith("var")]
            fieldnames = ["case_id"] + var_cols + param_cols

            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            writer.writerows(test_cases)

    logging.info(f"Generated test_cases.csv at {csv_path}")
    return csv_path


class AscendBackend:
    def __init__(self, eval_src: str, ref_src: str, seed_num: int = 1024, num_correct_trials: int = 5, device_id: int = 0):
        self.context = {}
        self.eval_context = {}
        self.ref_context = {}
        self.device_id = device_id
        self.device = torch.device(f'npu:{device_id}')
        self.seed_num = seed_num
        self.num_correct_trials = num_correct_trials
        self.profiler = MsprofProfiler(device_id=device_id)
        self.advanced_engine = None  # 按需创建

        # 显式设置当前 NPU 设备，确保 getCurrentNPUStream() 返回正确设备的流
        torch_npu.npu.set_device(device_id)
        self._set_context(eval_src, ref_src)

    def _set_context(self, eval_src: str, ref_src: str):
        try:
            exec(eval_src, self.eval_context)
            exec(ref_src, self.ref_context)
            # 合并上下文用于 Model 创建
            self.context = {**self.ref_context, **self.eval_context}
        except Exception as e:
            raise RuntimeError(f"Failed to compile reference model: {str(e)}")

    def _synchronize(self):
        """Synchronize NPU operations."""
        torch_npu.npu.synchronize(self.device)

    def _move_to_device(self, data):
        """Move tensor data to NPU device."""
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._move_to_device(x) for x in data]
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return data

    def _prepare_inputs(self):
        """Get inputs from context and move to device."""
        get_inputs = self.context["get_inputs"]
        inputs = get_inputs()
        return self._move_to_device(inputs)

    def _prepare_init_inputs(self):
        """Get init_inputs from context and move to device."""
        get_init_inputs = self.context["get_init_inputs"]
        init_inputs = get_init_inputs()
        return self._move_to_device(init_inputs)

    def _build_golden_output(self, inputs, init_inputs):
        """Run Model on CPU in fp64 to get golden reference."""
        if isinstance(inputs, dict):
            cpu_inputs = {k: (t.detach().cpu().double() if isinstance(t, torch.Tensor) and t.is_floating_point()
                               else (t.detach().cpu() if isinstance(t, torch.Tensor) else t))
                           for k, t in inputs.items()}
        else:
            cpu_inputs = [
                t.detach().cpu().double() if isinstance(t, torch.Tensor) and t.is_floating_point()
                else (t.detach().cpu() if isinstance(t, torch.Tensor) else t)
                for t in inputs
            ]
        cpu_init = [
            t.detach().cpu().double() if isinstance(t, torch.Tensor) and t.is_floating_point()
            else (t.detach().cpu() if isinstance(t, torch.Tensor) else t)
            for t in init_inputs
        ]
        ModelClass = self.context['Model']
        model = ModelClass(*cpu_init).cpu().double()
        with torch.no_grad():
            return model(*cpu_inputs)

    def _create_model(self, model_name='ModelNew', init_inputs=None):
        """Create model instance and move to device."""
        ModelClass = self.context[model_name]
        if init_inputs is None:
            init_inputs = self._prepare_init_inputs()

        model = ModelClass(*init_inputs).to(self.device)
        self._synchronize()
        return model

    def _normalize_output(self, output, index):
        """Extract single output tensor from list/tuple or return as-is."""
        if isinstance(output, (list, tuple)):
            return output[index]
        return output

    def _check_shape(self, ref_output, new_output, output_idx):
        """Check if output shapes match. Returns error message or None."""
        if ref_output.shape != new_output.shape:
            return f"[FAIL] Output shape mismatch at output {output_idx}: Expected {ref_output.shape}, got {new_output.shape}"
        return None

    def _check_values(self, ref_output, new_output, output_idx, atol=1e-02, rtol=1e-02):
        """Check if output values are close. Returns (error_msg, pass_info) tuple."""
        if ref_output.dtype in (torch.float16, torch.bfloat16):
            ref_output = ref_output.float()
            new_output = new_output.float()
        close_mask = torch.isclose(ref_output, new_output, atol=atol, rtol=rtol)
        total = close_mask.numel()
        matched = close_mask.sum().item()
        match_rate = matched / total

        if torch.allclose(ref_output, new_output, atol=atol, rtol=rtol):
            max_diff = (ref_output - new_output).abs().max().item()
            mean_diff = (ref_output - new_output).abs().mean().item()
            pass_info = (
                f"Output {output_idx}: shape={list(ref_output.shape)}, "
                f"match_rate=100.00% ({matched}/{total}), "
                f"max_diff={max_diff:.5e}, mean_diff={mean_diff:.5e}"
            )
            return None, pass_info

        mismatch_idx = (~close_mask).nonzero(as_tuple=False)[0]
        mismatch_idx_tuple = tuple(mismatch_idx.tolist())
        ref_val = ref_output[mismatch_idx_tuple].item()
        new_val = new_output[mismatch_idx_tuple].item()

        error_msg = (
            f"[FAIL] Output {output_idx} mismatch\n"
            f"Match rate: {match_rate * 100:.2f}% ({matched}/{total})\n"
            f"Example mismatch at index {tuple(mismatch_idx.tolist())}: "
            f"ref={ref_val}, new={new_val}"
        )
        return error_msg, None

    def evaluate_correctness(self, json_output_path=None, op_name="unknown", case_id=None):
        """Three-way precision check: golden(CPU-fp64) vs ref(NPU) vs ans(NPU).

        Falls back to legacy two-way comparison if golden model fails.
        """
        self.json_output_path = json_output_path
        try:
            set_seed(self.seed_num)
            inputs = self._prepare_inputs()
            init_inputs = self._prepare_init_inputs()

            # Extract case params from inputs before they're moved to device
            case_params = self._extract_case_params(inputs)

            # Try to build golden output (CPU fp64)
            try:
                golden_output = self._build_golden_output(inputs, init_inputs)
            except Exception as e:
                logging.warning(f"Golden model failed, falling back to legacy comparison: {e}")
                golden_output = None

            ref_model = self._create_model('Model', init_inputs)
            new_model = self._create_model('ModelNew', init_inputs)
            with torch.no_grad():
                if isinstance(inputs, dict):
                    ref_output = ref_model(**inputs)
                    new_output = new_model(**inputs)
                else:
                    ref_output = ref_model(*inputs)
                    new_output = new_model(*inputs)
            self._synchronize()

            if golden_output is not None:
                has_error, message, json_data = self._compare_outputs_threeway(
                    golden_output, ref_output, new_output)
                if json_output_path:
                    self._write_precision_json(json_data, op_name=op_name,
                                              case_id=case_id,
                                              case_params=case_params)
            else:
                has_error, message = self._compare_outputs(ref_output, new_output)

            return not has_error, message

        except Exception as e:
            logging.error("[FAIL] runtime error when evaluating correctness")
            return False, f"[FAIL] {str(e)}"

    def _compare_outputs(self, ref_output, new_output):
        """Compare model outputs and return (has_error, message)."""
        error_parts = []
        pass_parts = []
        num_outputs = len(ref_output) if isinstance(ref_output, (list, tuple)) else 1

        for i in range(num_outputs):
            ref_out = self._normalize_output(ref_output, i).to("cpu")
            new_out = self._normalize_output(new_output, i).to("cpu")

            error = self._check_shape(ref_out, new_out, i)
            if error:
                error_parts.append(error)
                continue

            error, pass_info = self._check_values(ref_out, new_out, i)
            if error:
                error_parts.append(error)
            else:
                pass_parts.append(pass_info)

        if error_parts:
            return True, "\n".join(error_parts)
        return False, "[PASS]\n" + "\n".join(pass_parts)

    def _compare_outputs_threeway(self, golden_output, ref_output, new_output):
        """Three-way comparison using dual_inspect. Returns (has_error, message, json_components)."""
        error_parts = []
        pass_parts = []
        json_components = {}
        num_outputs = len(ref_output) if isinstance(ref_output, (list, tuple)) else 1

        for i in range(num_outputs):
            ref_out = self._normalize_output(ref_output, i).detach().cpu()
            new_out = self._normalize_output(new_output, i).detach().cpu()
            golden_out = self._normalize_output(golden_output, i).detach().cpu()

            if ref_out.shape != new_out.shape:
                error_parts.append(
                    f"[FAIL] Output {i} shape mismatch: expected {ref_out.shape}, got {new_out.shape}")
                continue

            if golden_out.shape != ref_out.shape:
                logging.warning(f"Output {i} golden shape mismatch: golden={golden_out.shape}, ref={ref_out.shape}, skipping three-way")
                # Fall back to simple two-way for this output
                error_parts.append(f"[WARN] Output {i}: golden shape mismatch, using legacy comparison")
                continue

            result = dual_inspect(new_out, ref_out, golden_out, f"Output_{i}",
                                  dtype=ref_out.dtype)
            json_components[f"Output_{i}"] = result.to_dict()

            if result.passed:
                pass_parts.append(self._format_human_pass(i, ref_out, result))
            else:
                error_parts.append(self._format_human_fail(i, ref_out, result))

        has_error = len(error_parts) > 0
        if has_error:
            all_parts = pass_parts + error_parts
            message = "\n".join(all_parts)
        else:
            message = "[PASS]\n" + "\n".join(pass_parts)

        return has_error, message, json_components

    def _format_human_pass(self, output_idx, ref_out, result):
        """Format a passing result for human-readable stderr."""
        r = result.ratios
        avg = result.ans_vs_golden
        return (
            f"Output {output_idx}: shape={list(ref_out.shape)}, dtype={ref_out.dtype}, "
            f"PASSED (ratios: max_re={r.max_re:.2f}, mean_re={r.mean_re:.2f}, "
            f"rmse={r.rmse:.2f}, mismatch_rate={avg.mismatch_rate:.4f})"
        )

    def _format_human_fail(self, output_idx, ref_out, result):
        """Format a failing result with box-formatted diagnostics for stderr."""
        r = result.ratios
        avg = result.ans_vs_golden
        rvg = result.ref_vs_golden
        diag = result.diagnosis

        lines = []
        lines.append(f"[FAIL] Output {output_idx} precision check failed")
        lines.append("\u2550" * 57)
        lines.append(f"  Shape: {list(ref_out.shape)}, dtype: {ref_out.dtype}")
        lines.append(f"  Mode: three-way (golden=CPU-fp64, ref=NPU, ans=NPU)")
        lines.append("")
        lines.append("  \u250c\u2500 Ratios (ans_vs_golden / ref_vs_golden) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
        for name, val, limit in [
            ("max_re", r.max_re, 10.0), ("mean_re", r.mean_re, 2.0),
            ("rmse", r.rmse, 2.0), ("svec", r.svec, 2.0),
        ]:
            status = "\u2190 EXCEEDED" if val > limit else "\u2713"
            lines.append(f"  \u2502 {name:8s}: {val:7.2f}  (limit: {limit:.1f})  {status:14s}\u2502")
        lines.append("  \u2514" + "\u2500" * 51 + "\u2518")
        lines.append("")
        lines.append("  \u250c\u2500 Element Stats " + "\u2500" * 39 + "\u2510")
        for label, counts in [("ans", result.ans_counts), ("ref", result.ref_counts), ("golden", result.golden_counts)]:
            lines.append(f"  \u2502 {label:7s}: NaN={counts.nan}, Inf={counts.inf}, "
                        f"subnormal={counts.subnormal}, small={counts.small:>6d}  \u2502")
        lines.append("  \u2514" + "\u2500" * 51 + "\u2518")
        lines.append("")
        lines.append("  \u250c\u2500 Key Metrics " + "\u2500" * 41 + "\u2510")
        lines.append("  \u2502           ans_vs_golden  ref_vs_golden    ratio   \u2502")
        for name in ["re_max", "re_mean", "rmse", "ae_max", "ae_mean"]:
            a_val = getattr(avg, name)
            r_val = getattr(rvg, name)
            ratio_val = a_val / max(r_val, 1e-7) if name in ("re_max", "re_mean", "rmse") else ""
            ratio_str = f"{ratio_val:>7.2f}" if isinstance(ratio_val, float) else "       "
            lines.append(f"  \u2502 {name:9s}: {a_val:>12.5e}  {r_val:>12.5e}  {ratio_str}  \u2502")
        lines.append(f"  \u2502 mismatch:  {avg.mismatch_rate*100:>10.2f}%   {rvg.mismatch_rate*100:>10.2f}%            \u2502")
        lines.append("  \u2514" + "\u2500" * 51 + "\u2518")
        lines.append("")

        if diag.get("root_causes"):
            cause = diag["root_causes"][0]
            lines.append(f"  Diagnosis: {diag.get('pattern', 'unknown')}")
            lines.append(f"  {cause['description']}")
            for s in cause["suggestions"][:2]:
                lines.append(f"  \u2192 {s}")
        lines.append("\u2550" * 57)

        return "\n".join(lines)

    def _extract_case_params(self, inputs):
        """Extract shape/dtype params from input tensors for precision JSON."""
        params = {}
        iterable = inputs.items() if isinstance(inputs, dict) else enumerate(inputs)
        for i, inp in iterable:
            if isinstance(inp, torch.Tensor):
                params[f"var{i}_shape"] = list(inp.shape)
                params[f"var{i}_dtype"] = str(inp.dtype).replace("torch.", "")
            elif isinstance(inp, (int, float, str, bool)):
                params[f"var{i}_value"] = inp
        return params

    def _measure_event_timing(self, n_warmup=10, n_runs=100):
        """Measure ref and ans models with NPU event timing.

        Returns dict with 'ans', 'ref' PerfResult dicts and 'speedup'.
        Returns error dict if measurement fails.
        """
        try:
            inputs = self._prepare_inputs()
            init_inputs = self._prepare_init_inputs()
            ref_model = self._create_model('Model', init_inputs)
            new_model = self._create_model('ModelNew', init_inputs)

            if isinstance(inputs, dict):
                ref_perf = measure_time(
                    lambda: ref_model(**inputs),
                    n_warmup=n_warmup, n_runs=n_runs, name="ref",
                )
                ans_perf = measure_time(
                    lambda: new_model(**inputs),
                    n_warmup=n_warmup, n_runs=n_runs, name="ans",
                )
            else:
                ref_perf = measure_time(
                    lambda: ref_model(*inputs),
                    n_warmup=n_warmup, n_runs=n_runs, name="ref",
                )
                ans_perf = measure_time(
                    lambda: new_model(*inputs),
                    n_warmup=n_warmup, n_runs=n_runs, name="ans",
                )

            speedup = round(
                ref_perf.median_ms / max(ans_perf.median_ms, 1e-9), 4)

            return {
                "ans": ans_perf.to_dict(),
                "ref": ref_perf.to_dict(),
                "speedup": speedup,
            }
        except Exception as e:
            logging.warning(f"NPU event timing failed: {e}")
            return {"error": str(e)}

    @staticmethod
    def _build_msprof_dict(total_us, perf_data):
        """Convert msprof profiling results to JSON-serializable dict."""
        result = {}
        if total_us is not None:
            result["total_us"] = round(total_us, 3)
        if perf_data and isinstance(perf_data, list):
            result["pipeline"] = perf_data
        elif perf_data and isinstance(perf_data, str):
            result["error"] = perf_data
        return result

    def _append_perf_to_json(self, performance_dict, msprof_dict=None, case_id=0):
        """Append performance and msprof data to an existing precision JSON case.

        Loads the JSON file at self.json_output_path, finds the case with
        matching case_id, and adds 'performance' and 'msprof' fields.
        """
        if not self.json_output_path:
            return

        json_path = Path(self.json_output_path)
        if not json_path.exists():
            logging.warning(f"Precision JSON not found at {json_path}, skipping perf append")
            return

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Failed to load precision JSON for perf append: {e}")
            return

        # Find the target case
        for case in data.get("cases", []):
            if case.get("id") == case_id:
                if performance_dict:
                    case["performance"] = performance_dict
                if msprof_dict:
                    case["msprof"] = msprof_dict
                break
        else:
            # case_id not found — append to last case if any
            cases = data.get("cases", [])
            if cases:
                if performance_dict:
                    cases[-1]["performance"] = performance_dict
                if msprof_dict:
                    cases[-1]["msprof"] = msprof_dict

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Performance data appended to precision JSON: {json_path}")

    def _collect_environment(self):
        """Collect environment info for precision JSON."""
        import platform
        env = {"torch_version": torch.__version__}
        try:
            env["torch_npu_version"] = torch_npu.__version__
            env["device"] = torch.npu.get_device_name(self.device_id)
        except Exception:
            pass
        cann = os.environ.get("ASCEND_TOOLKIT_HOME", "")
        if cann:
            env["cann_version"] = os.path.basename(cann)
        env["python_version"] = platform.python_version()
        return env

    def _write_precision_json(self, json_components, op_name="unknown",
                              case_id=None, case_params=None, case_seed=None):
        """Write precision JSON file, appending cases if file already exists."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        json_path = Path(self.json_output_path)

        # Load existing JSON if appending
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                # Auto-increment case_id from existing cases
                if case_id is None:
                    existing_ids = [c["id"] for c in data.get("cases", [])]
                    case_id = max(existing_ids, default=-1) + 1
            except (json.JSONDecodeError, KeyError):
                logging.warning(f"Failed to parse existing JSON at {json_path}, overwriting")
                data = None
        else:
            data = None

        if data is None:
            data = {
                "schema_version": 2,
                "run_id": f"run_{now.strftime('%Y%m%d_%H%M%S')}",
                "timestamp": now.isoformat(),
                "op_name": op_name,
                "environment": self._collect_environment(),
                "kernel": {"name": op_name},
                "ulp": DEFAULT_ULP_CONFIG,
                "tolerances": {},
                "cases": [],
            }
            if case_id is None:
                case_id = 0

        # Append new case
        data["cases"].append({
            "id": case_id,
            "params": case_params or {},
            "seed": case_seed if case_seed is not None else self.seed_num,
            "distr": "normal(0,1)",
            "forward": json_components,
        })

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        self._last_case_id = case_id
        logging.info(f"Precision JSON written to: {json_path} (case_id={case_id})")

    def measure_performance(
        self,
        model_name='ModelNew',
        profile_root: Path = None,
        num_warmup=10,
        num_profile_trials=20
    ):
        """Profile a single model and return op_summary path and total duration."""
        model = self._create_model(model_name)
        inputs = self._prepare_inputs()
        profile_root = Path(profile_root).resolve() if profile_root else Path("profiling").resolve()
        return self.profiler.profile_model(
            model,
            inputs,
            profile_root,
            sync_fn=self._synchronize,
            num_warmup=num_warmup,
            num_trials=num_profile_trials
        )

    def compare_performance(
        self,
        profile_root: Path = None,
        num_warmup=10,
        num_profile_trials=20
    ):
        """Profile reference and custom models via msprof."""
        model_ref = self._create_model('Model')
        model_new = self._create_model('ModelNew')
        inputs = self._prepare_inputs()
        profile_root = Path(profile_root).resolve() if profile_root else Path("profiling").resolve()
        ref_summary, ref_total_us = self.profiler.profile_model(
            model_ref,
            inputs,
            profile_root,
            sync_fn=self._synchronize,
            num_warmup=num_warmup,
            num_trials=num_profile_trials
        )
        custom_summary, custom_total_us = self.profiler.profile_model(
            model_new,
            inputs,
            profile_root,
            sync_fn=self._synchronize,
            num_warmup=num_warmup,
            num_trials=num_profile_trials
        )
        return ref_summary, ref_total_us, custom_summary, custom_total_us

    def compare_performance_advanced(
        self,
        profile_root: Path = None,
        num_trials=50,
        task_type="vector"
    ):
        """
        高级性能测试：使用预热和清缓存逻辑。

        Returns:
            tuple: (ref_time, ref_perf_data, ref_profile_dir, ref_cv_pct,
                    custom_time, custom_perf_data, custom_profile_dir, custom_cv_pct)
                - ref_time: 参考模型耗时（微秒）
                - ref_perf_data: 参考模型性能数据列表
                - ref_profile_dir: 参考模型 profiling 子目录路径
                - ref_cv_pct: 参考模型测量变异系数（%）
                - custom_time: 自定义算子耗时（微秒）
                - custom_perf_data: 自定义算子性能数据列表
                - custom_profile_dir: 自定义算子 profiling 子目录路径
                - custom_cv_pct: 自定义算子测量变异系数（%）
        """
        if self.advanced_engine is None:
            self.advanced_engine = AdvancedPerformanceEngine(logging.getLogger("AscendBackend"))

        model_ref = self._create_model('Model')
        model_new = self._create_model('ModelNew')
        inputs = self._prepare_inputs()
        profile_root = Path(profile_root).resolve() if profile_root else Path("profiling").resolve()

        # 测试参考模型
        ref_time, ref_perf_data, ref_profile_dir, ref_cv_pct = self.advanced_engine.warmup_and_measure(
            model_ref,
            inputs,
            device_id=self.device_id,
            profile_root=profile_root,
            num_trials=num_trials,
            task_type=task_type,
            model_tag="Model"
        )

        # 测试自定义算子
        custom_time, custom_perf_data, custom_profile_dir, custom_cv_pct = self.advanced_engine.warmup_and_measure(
            model_new,
            inputs,
            device_id=self.device_id,
            profile_root=profile_root,
            num_trials=num_trials,
            task_type=task_type,
            model_tag="ModelNew"
        )

        return ref_time, ref_perf_data, ref_profile_dir, ref_cv_pct, custom_time, custom_perf_data, custom_profile_dir, custom_cv_pct

    def cleanup(self):
        del self.context
        torch_npu.npu.empty_cache()
        self._synchronize()


def check_op_name_conflicts(
    eval_src_path: Path,
    ref_src_path: Path,
    project_root_path: Path,
) -> List[str]:
    """Check if custom operator names conflict with built-in torch_npu operators.

    Detects three types of conflicts:
    1. C++ TORCH_LIBRARY_IMPL registering in 'aten' namespace (overrides built-in dispatch)
    2. aclnn API name in EXEC_NPU_CMD colliding with standard libopapi.so symbols
    3. Python-level torch.ops.aten overrides

    Args:
        eval_src_path: Path to the custom operator Python source
        ref_src_path: Path to the reference model Python source
        project_root_path: Path to the project root (for finding C++ and .so files)

    Returns:
        List of warning/error messages (empty if no conflicts)
    """
    import re
    import subprocess

    warnings_list: List[str] = []

    # --- 1. Check C++ source for TORCH_LIBRARY_IMPL namespace & EXEC_NPU_CMD names ---
    cpp_search_dirs = [
        project_root_path,
        project_root_path / "CppExtension" / "csrc",
    ]
    # Also search in any *Custom directories
    for d in project_root_path.iterdir():
        if d.is_dir() and d.name.endswith("Custom"):
            cpp_search_dirs.append(d)

    cpp_files: List[Path] = []
    for d in cpp_search_dirs:
        if d.exists():
            cpp_files.extend(d.glob("*.cpp"))

    custom_aclnn_apis: List[str] = []
    for cpp_file in cpp_files:
        try:
            content = cpp_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # Check TORCH_LIBRARY_IMPL namespace
        lib_impls = re.findall(r'TORCH_LIBRARY_IMPL\s*\(\s*(\w+)', content)
        for ns in lib_impls:
            if ns == "aten":
                warnings_list.append(
                    f"[CRITICAL] {cpp_file.name}: TORCH_LIBRARY_IMPL registers in 'aten' "
                    f"namespace. This WILL override built-in torch_npu dispatch and corrupt "
                    f"reference model results. Use a custom namespace (e.g., 'myops') instead."
                )

        # Extract EXEC_NPU_CMD aclnn API names
        aclnn_apis = re.findall(r'EXEC_NPU_CMD\s*\(\s*(aclnn\w+)', content)
        custom_aclnn_apis.extend(aclnn_apis)

    # --- 2. Check aclnn API names against standard libopapi.so ---
    std_aclnn_symbols: set = set()
    if custom_aclnn_apis:
        ascend_home = os.environ.get("ASCEND_HOME_PATH",
                                     "/usr/local/Ascend/ascend-toolkit/latest")
        # Search common locations for libopapi.so
        opapi_candidates = [
            Path(ascend_home) / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe"
            / "op_api" / "lib" / "linux" / "aarch64" / "libopapi.so",
            Path(ascend_home) / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe"
            / "op_api" / "lib" / "linux" / "x86_64" / "libopapi.so",
            Path(ascend_home) / "opp" / "lib64" / "libopapi.so",
        ]
        opapi_path = None
        for candidate in opapi_candidates:
            if candidate.exists():
                opapi_path = candidate
                break

        if opapi_path is not None:
            try:
                result = subprocess.run(
                    ["nm", "-D", "--defined-only", str(opapi_path)],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        parts = line.split()
                        if len(parts) >= 3 and parts[1] == "T":
                            sym = parts[2]
                            if sym.startswith("aclnn") and not sym.endswith("GetWorkspaceSize"):
                                std_aclnn_symbols.add(sym)
            except (subprocess.TimeoutExpired, OSError):
                pass

        for api_name in custom_aclnn_apis:
            if api_name in std_aclnn_symbols:
                warnings_list.append(
                    f"[CRITICAL] Custom aclnn API '{api_name}' collides with a standard "
                    f"built-in operator in libopapi.so. When ASCEND_CUSTOM_OPP_PATH is set, "
                    f"libcust_opapi.so is loaded first and will shadow the built-in, causing "
                    f"the reference model to use the custom implementation instead. "
                    f"Rename to '{api_name}Custom' or similar."
                )
            elif not api_name.endswith("Custom"):
                # Soft warning for unconventional naming
                warnings_list.append(
                    f"[WARNING] Custom aclnn API '{api_name}' does not follow the 'Custom' "
                    f"suffix convention. Consider renaming to '{api_name}Custom' to avoid "
                    f"potential conflicts with future standard operators."
                )

    # --- 3. Check Python-level torch.ops overrides ---
    for src_path, label in [(eval_src_path, "eval"), (ref_src_path, "ref")]:
        if not src_path.exists():
            continue
        try:
            src = src_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        # Check for aten namespace overrides
        if re.search(r'torch\.ops\.aten\.\w+\s*=', src):
            warnings_list.append(
                f"[CRITICAL] {src_path.name} ({label}): assigns to torch.ops.aten namespace, "
                f"which will override built-in operator dispatch."
            )

    # --- 4. Cross-check: custom lib exports vs standard lib ---
    custom_opp_path = project_root_path / "vendors" / "customize"
    cust_lib = custom_opp_path / "op_api" / "lib" / "libcust_opapi.so"
    if cust_lib.exists() and std_aclnn_symbols:
        try:
            result = subprocess.run(
                ["nm", "-D", "--defined-only", str(cust_lib)],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] == "T":
                        sym = parts[2]
                        if sym.startswith("aclnn") and not sym.endswith("GetWorkspaceSize"):
                            if sym in std_aclnn_symbols:
                                # Already reported in step 2 if from EXEC_NPU_CMD,
                                # but catch any extra exports
                                msg = (
                                    f"[CRITICAL] libcust_opapi.so exports symbol '{sym}' "
                                    f"which collides with standard libopapi.so. This will "
                                    f"shadow the built-in operator for ALL callers in the "
                                    f"process, including the reference model."
                                )
                                if msg not in warnings_list:
                                    warnings_list.append(msg)
        except (subprocess.TimeoutExpired, OSError):
            pass

    return warnings_list


def setup_ascend_runtime_environment(project_root: Path) -> None:
    """
    设置 AscendC 自定义算子所需的运行时环境变量。

    Args:
        project_root (Path): 项目根目录路径
    Sets:
        - ASCEND_CUSTOM_OPP_PATH
        - LD_LIBRARY_PATH (追加自定义库路径 + CANN 系统库路径)
    """
    project_root = Path(project_root).resolve()

    # 1. 设置 ASCEND_CUSTOM_OPP_PATH
    custom_opp_path = project_root.joinpath("vendors/customize")

    if not custom_opp_path.exists():
        raise FileNotFoundError(f"ASCEND custom OPP directory not found: {custom_opp_path}")

    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp_path)
    logging.info(f"Set ASCEND_CUSTOM_OPP_PATH={custom_opp_path}")

    # 2. 更新 LD_LIBRARY_PATH — 包括自定义算子库 + CANN 系统库
    #    CANN 系统库在非交互 shell 中可能缺失（.bashrc 有 PS1 guard），
    #    必须显式添加，否则 torch_npu import 会报 libhccl.so / libascend_hal.so 找不到。
    custom_lib_path = Path(custom_opp_path).joinpath("op_api/lib").resolve()

    if not custom_lib_path.exists():
        raise FileNotFoundError(f"ASCEND custom OP API library directory not found: {custom_lib_path}")

    # CANN system library paths (required for torch_npu in non-interactive shells)
    import platform
    arch = platform.machine()  # aarch64 or x86_64
    arch_suffix = f"{arch}-linux"

    cann_system_lib_paths = [
        "/usr/local/Ascend/driver/lib64/driver",  # libascend_hal.so
    ]
    # Find toolkit lib64 path
    import glob
    toolkit_lib_candidates = sorted(glob.glob(f"/usr/local/Ascend/ascend-toolkit/*/{arch_suffix}/lib64"))
    if toolkit_lib_candidates:
        cann_system_lib_paths.append(toolkit_lib_candidates[-1])  # libhccl.so etc.

    all_lib_paths = [str(custom_lib_path)] + cann_system_lib_paths
    existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    paths_to_add = [p for p in all_lib_paths if p not in existing_ld_path]
    if paths_to_add:
        new_ld_path = ":".join(paths_to_add) + ":" + existing_ld_path
        new_ld_path = new_ld_path.rstrip(":")
        os.environ["LD_LIBRARY_PATH"] = new_ld_path
        logging.info(f"Updated LD_LIBRARY_PATH, added: {paths_to_add}")
    else:
        logging.info("LD_LIBRARY_PATH already contains all required paths")


def load_case_spec(case_spec_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not case_spec_path:
        return None

    spec_path = Path(case_spec_path).expanduser().resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"case_spec file not found: {spec_path}")

    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_operator(
    eval_src_path: Path,
    ref_src_path: Path,
    project_root_path: Path,
    skip_env_setup: bool = False,
    task_type: str = "vector",
    case_spec: Optional[Dict[str, Any]] = None,
    append_cases: bool = False,
    device_id: int = 0,
    json_output_path: Optional[str] = None,
    case_id: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    评估算子代码正确性和性能。

    统一使用 advanced-perf 模式（带预热和清缓存）进行性能测试。

    Args:
        eval_src (Path): 要评估的代码文件路径（包含自定义算子实现的 ModelNew 类）
        ref_src (Path): 参考代码文件路径（包含参考实现的 Model 类）
        project_root_path (Path): 项目根目录路径，用于设置运行时环境
        skip_env_setup (bool): 是否跳过环境变量设置（远程包装器已完成时为True）
        task_type (str): 算子类型（vector/cube/cv-mix）

    Returns:
        Tuple of (success, output_message):
            - success (bool): 评估是否成功
            - output_message (str): 评估结果详情或错误信息
    """
    # Inject per-operator pybind_lib path for parallel isolation
    _work_dir_for_pybind = project_root_path.parent if project_root_path.name.endswith("Custom") else project_root_path
    _inject_pybind_lib(_work_dir_for_pybind)
    # 读取文件内容
    if not skip_env_setup:
        setup_ascend_runtime_environment(project_root_path)

    if not eval_src_path.exists():
        raise FileNotFoundError(f'Evaluation code file not found: {eval_src_path}')
    if not ref_src_path.exists():
        raise FileNotFoundError(f'Reference code file not found: {ref_src_path}')

    eval_code = eval_src_path.read_text(encoding='utf-8')
    ref_code = ref_src_path.read_text(encoding='utf-8')

    # 检查算子名冲突（自定义算子 vs 内置 torch_npu 算子）
    conflicts = check_op_name_conflicts(eval_src_path, ref_src_path, project_root_path)
    for msg in conflicts:
        if msg.startswith("[CRITICAL]"):
            logging.error(msg)
        else:
            logging.warning(msg)
    if any(msg.startswith("[CRITICAL]") for msg in conflicts):
        return False, (
            "[FAIL] Operator name conflict detected — custom op may shadow built-in "
            "torch_npu operators, corrupting reference model results.\n"
            + "\n".join(conflicts)
        )

    # 生成 test_cases.csv
    op_name = project_root_path.name

    # 自动检测 op_desc.json
    op_desc_path = project_root_path / f"{op_name}_op_desc.json"
    if not op_desc_path.exists():
        op_desc_path = None

    test_cases_path = generate_test_cases_csv(
        project_root_path,
        op_name,
        eval_code,
        ref_code,
        case_spec=case_spec,
        append=append_cases,
        op_desc_path=op_desc_path,
    )

    ascend_backend = AscendBackend(eval_code, ref_code, device_id=device_id)
    flag_bool, run_info = ascend_backend.evaluate_correctness(
        json_output_path=json_output_path,
        op_name=project_root_path.name,
        case_id=case_id,
    )
    logging.info(f"Evaluation correctness: {run_info}")

    speedup = 0.0
    ref_time_us = 0.0
    custom_time_us = 0.0

    if flag_bool:
        profile_root = project_root_path.joinpath("profiling")

        ref_msprof = {}
        custom_msprof = {}

        # 统一使用高级性能测试（带预热和清缓存）
        ref_time, ref_perf_data, ref_profile_dir, ref_cv_pct, custom_time, custom_perf_data, custom_profile_dir, custom_cv_pct = ascend_backend.compare_performance_advanced(
            profile_root=profile_root,
            num_trials=50,
            task_type=task_type
        )
        logging.info(f"Advanced performance test completed")
        logging.info(f"Reference time: {ref_time:.3f}us, Custom time: {custom_time:.3f}us")

        if ref_time is not None and custom_time is not None and custom_time > 0:
            speedup = ref_time / custom_time
            ref_time_us = ref_time
            custom_time_us = custom_time
            logging.info(f"Speedup: {speedup:.2f}x")

        # 记录性能数据详情
        if custom_perf_data:
            logging.info(f"Custom operator performance data: {custom_perf_data}")

        # 写 report.txt 到 profiling 根目录，方便人工回看
        try:
            import datetime as _dt
            report_path = profile_root / "report.txt"
            with open(report_path, "a", encoding="utf-8") as _f:
                _f.write(f"[{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
                _f.write(f"  Reference dir  : {ref_profile_dir}\n")
                _f.write(f"  Custom dir     : {custom_profile_dir}\n")
                _f.write(f"  Reference time : {ref_time:.3f} us\n" if ref_time is not None else "  Reference time : N/A\n")
                _f.write(f"  Custom time    : {custom_time:.3f} us\n" if custom_time is not None else "  Custom time    : N/A\n")
                _f.write(f"  Speedup        : {speedup:.2f}x\n" if speedup is not None else "  Speedup        : N/A\n")
                _f.write("\n")
        except Exception as _e:
            logging.warning(f"Failed to write report.txt: {_e}")

        ref_msprof = AscendBackend._build_msprof_dict(ref_time, ref_perf_data)
        custom_msprof = AscendBackend._build_msprof_dict(custom_time, custom_perf_data)

        # Append perf data to precision JSON if enabled
        if json_output_path:
            event_perf = ascend_backend._measure_event_timing(n_warmup=10, n_runs=100)
            msprof_combined = {}
            if ref_msprof:
                msprof_combined["ref"] = ref_msprof
            if custom_msprof:
                msprof_combined["ans"] = custom_msprof

            # Use the actual case_id assigned by _write_precision_json, falling back
            # to the caller-supplied case_id or 0 if precision JSON was not written.
            actual_case_id = getattr(ascend_backend, '_last_case_id', None)
            if actual_case_id is None:
                actual_case_id = case_id if case_id is not None else 0
            ascend_backend._append_perf_to_json(
                event_perf, msprof_combined or None, case_id=actual_case_id)
    perf_data = {
        "speedup": speedup,
        "ref_time_us": ref_time_us,
        "custom_time_us": custom_time_us,
        "cv_pct": custom_cv_pct if custom_cv_pct is not None else 0.0,
        "measurement_quality": _measurement_quality(custom_cv_pct if custom_cv_pct is not None else 0.0),
    }

    return flag_bool, run_info, perf_data


def _measurement_quality(cv_pct: float) -> str:
    """Classify measurement quality based on coefficient of variation."""
    if cv_pct < 5.0:
        return "good"
    elif cv_pct < 15.0:
        return "acceptable"
    return "noisy"


def save_evaluation_results(work_dir: Path, compilation_success: bool, precision_passed: bool,
                             speedup: float, base_time_ms: float, gen_time_ms: float,
                             correctness_message: str,
                             cv_pct: float = 0.0) -> None:
    """Save evaluation results as JSON for tool-driven workflows (e.g. evolution)."""
    results = {
        "compilation_success": compilation_success,
        "precision_passed": precision_passed,
        "speedup": round(speedup, 4),
        "base_time_ms": round(base_time_ms, 4),
        "gen_time_ms": round(gen_time_ms, 4),
        "correctness_message": correctness_message,
        "cv_pct": round(cv_pct, 2),
        "measurement_quality": _measurement_quality(cv_pct),
    }
    json_path = work_dir / "evaluation_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logging.info(f"Saved evaluation results to {json_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Evaluate AscendC custom operator correctness")
    parser.add_argument("op_name", type=str, help="Operator name")
    parser.add_argument("--task-type", type=str, default="vector",
                       choices=["vector", "cube", "cv-mix", "unknown"],
                       help="Operator type for performance analysis (default: vector)")
    parser.add_argument("--case-spec", type=str, default=None,
                       help="Path to JSON case spec for test_cases.csv generation")
    parser.add_argument("--append-cases", action="store_true",
                       help="Append generated cases to existing test_cases.csv")
    parser.add_argument("--generate-cases-only", action="store_true",
                       help="Only generate test_cases.csv and exit")
    parser.add_argument("--device-id", type=int,
                       default=int(os.environ.get("ASCEND_DEVICE_ID", "0")),
                       help="NPU device ID (default: from ASCEND_DEVICE_ID env var, or 0)")
    parser.add_argument("--work-dir", type=str, default=None,
                       help="Work directory containing op files (default: output/<op_name>)")
    parser.add_argument("--json-output", type=str, default=None,
                       help="Path to write precision JSON output file")

    args = parser.parse_args()

    try:
        if args.work_dir:
            work_dir = Path(args.work_dir).resolve()
        else:
            work_dir = Path("output").joinpath(args.op_name).resolve()
        _inject_pybind_lib(work_dir)
        eval_src = work_dir.joinpath(f"{args.op_name}_custom.py")
        ref_src = work_dir.joinpath(f"{args.op_name}_reference.py")
        case_spec = load_case_spec(args.case_spec)

        # 自动检测 op_desc.json
        op_desc_path = work_dir / f"{args.op_name}_op_desc.json"
        if not op_desc_path.exists():
            op_desc_path = None

        if args.generate_cases_only:
            eval_code = eval_src.read_text(encoding='utf-8')
            ref_code = ref_src.read_text(encoding='utf-8')
            generate_test_cases_csv(
                work_dir,
                args.op_name,
                eval_code,
                ref_code,
                case_spec=case_spec,
                append=args.append_cases,
                op_desc_path=op_desc_path,
            )
            sys.exit(0)

        flag_bool, run_info, perf_data = evaluate_operator(
            eval_src,
            ref_src,
            work_dir,
            task_type=args.task_type,
            case_spec=case_spec,
            append_cases=args.append_cases,
            device_id=args.device_id,
            json_output_path=args.json_output,
        )
        # Save evaluation results to JSON for tool-driven workflows (e.g. evolution)
        try:
            save_evaluation_results(
                work_dir=work_dir,
                compilation_success=True,
                precision_passed=flag_bool,
                speedup=perf_data.get("speedup", 0.0),
                base_time_ms=perf_data.get("ref_time_us", 0.0) / 1000.0,
                gen_time_ms=perf_data.get("custom_time_us", 0.0) / 1000.0,
                correctness_message=run_info,
                cv_pct=perf_data.get("cv_pct", 0.0),
            )
        except Exception as save_err:
            logging.warning(f"Failed to save evaluation results: {save_err}")
    except Exception as e:
        logging.error(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        try:
            save_evaluation_results(
                work_dir=work_dir,
                compilation_success=False,
                precision_passed=False,
                speedup=0.0,
                base_time_ms=0.0,
                gen_time_ms=0.0,
                correctness_message=str(e),
            )
        except Exception:
            pass

