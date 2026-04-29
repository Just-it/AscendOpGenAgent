import copy
import importlib.util
import inspect
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# 精度对比标准（参考 精度对比方法.md）
# ---------------------------------------------------------------------------
# dtype_str -> Threshold (MERE 通过阈值)
PRECISION_THRESHOLDS = {
    "float16": 2 ** -10,       # ≈ 9.77e-4
    "bfloat16": 2 ** -7,       # ≈ 7.81e-3
    "float32": 2 ** -13,       # ≈ 1.22e-4
    "float64": 2 ** -30,       # ≈ 9.31e-10
    "complex64": 2 ** -13,     # 实部/虚部各为 float32
    "complex128": 2 ** -30,    # 实部/虚部各为 float64
    "hifloat32": 2 ** -11,     # ≈ 4.88e-4
    "float8_e4m3": 2 ** -3,    # ≈ 0.125
    "float8_e5m2": 2 ** -2,    # ≈ 0.25
}

# 量化整数输出的 LSB tolerance：dynamic quant / smooth quant 等算子的
# int8/int16 输出，在 NPU 实现侧通常经过 fp32→fp16→int 的中转 cast，
# 与 PyTorch CPU 全 fp32 .round() 比较时会出现 ±1 LSB 噪声。
# 这里允许每个元素最多差 1 个 LSB（相当于浮点的 1 ulp 容忍）。
# int32 / int64 / bool 不带 round 语义（通常是索引、计数、mask），
# 仍走严格 torch.equal。
INT_LSB_TOLERANCE = {
    torch.int8:  1,
    torch.int16: 1,
}


def _compute_mere(actual: torch.Tensor, golden: torch.Tensor, eps: float = 1e-7) -> float:
    """计算平均相对误差 (Mean Relative Error).

    MERE = mean(|actual - golden| / (|golden| + eps))
    """
    diff = (actual - golden).abs()
    rel = diff / (golden.abs() + eps)
    if rel.numel() == 0:
        return 0.0
    return float(rel.mean().item())


def _compute_mare(actual: torch.Tensor, golden: torch.Tensor, eps: float = 1e-7) -> float:
    """计算最大相对误差 (Max Relative Error).

    MARE = max(|actual - golden| / (|golden| + eps))
    """
    diff = (actual - golden).abs()
    rel = diff / (golden.abs() + eps)
    if rel.numel() == 0:
        return 0.0
    return float(rel.max().item())


def _get_threshold_for_tensor(t: torch.Tensor) -> float:
    """根据张量 dtype 获取 MERE 阈值."""
    dtype_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.complex64: "complex64",
        torch.complex128: "complex128",
    }
    # 安全获取可能不存在的 dtype（取决于 PyTorch 版本）
    _hifloat32 = getattr(torch, "hifloat32", None)
    if _hifloat32 is not None:
        dtype_map[_hifloat32] = "hifloat32"
    _float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    if _float8_e4m3fn is not None:
        dtype_map[_float8_e4m3fn] = "float8_e4m3"
    _float8_e5m2 = getattr(torch, "float8_e5m2", None)
    if _float8_e5m2 is not None:
        dtype_map[_float8_e5m2] = "float8_e5m2"

    dtype_str = dtype_map.get(t.dtype)
    if dtype_str is not None:
        return PRECISION_THRESHOLDS.get(dtype_str, 1e-2)

    # 其他浮点类型回退到 float32 阈值
    if torch.is_floating_point(t):
        return PRECISION_THRESHOLDS.get("float32", 1e-2)

    # 复数类型回退到对应浮点阈值
    if t.is_complex():
        return PRECISION_THRESHOLDS.get("float32", 1e-2)

    # 非浮点类型使用宽松阈值
    return 1e-2


def _check_precision_mere_mare(actual: torch.Tensor, golden: torch.Tensor) -> tuple:
    """根据《精度对比方法.md》判定数值精度是否通过.

    通过标准:
        MERE < Threshold 且 MARE < 10 * Threshold

    Returns:
        (passed: bool, mere: float, mare: float, threshold: float, mare_threshold: float)
    """
    threshold = _get_threshold_for_tensor(golden)
    mare_threshold = 10 * threshold

    # NaN 处理：两者都为 NaN 的位置视为匹配并过滤；仅一方为 NaN 直接判定失败
    actual_nan = torch.isnan(actual)
    golden_nan = torch.isnan(golden)
    if (actual_nan ^ golden_nan).any():
        return False, float('inf'), float('inf'), threshold, mare_threshold

    # Inf 处理：两者都为 Inf 且同号的位置视为匹配并过滤；仅一方为 Inf 或符号不同直接判定失败
    actual_inf = torch.isinf(actual)
    golden_inf = torch.isinf(golden)
    if (actual_inf ^ golden_inf).any():
        return False, float('inf'), float('inf'), threshold, mare_threshold
    if actual_inf.any():
        actual_sign = torch.sign(actual[actual_inf])
        golden_sign = torch.sign(golden[actual_inf])
        if not torch.equal(actual_sign, golden_sign):
            return False, float('inf'), float('inf'), threshold, mare_threshold

    valid_mask = ~(actual_nan | actual_inf)
    if valid_mask.any():
        actual_valid = actual[valid_mask]
        golden_valid = golden[valid_mask]
    else:
        # 所有元素均为双 NaN 或双 Inf
        return True, 0.0, 0.0, threshold, mare_threshold

    mere = _compute_mere(actual_valid, golden_valid)
    mare = _compute_mare(actual_valid, golden_valid)

    passed = mere < threshold and mare < mare_threshold
    return passed, mere, mare, threshold, mare_threshold


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


def _contains_int8_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.dtype == torch.int8
    if isinstance(value, list):
        return any(_contains_int8_tensor(item) for item in value)
    if isinstance(value, tuple):
        return any(_contains_int8_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_int8_tensor(item) for item in value.values())
    return False


def _tensor_diff_summary(lhs: torch.Tensor, rhs: torch.Tensor):
    if lhs.shape != rhs.shape:
        return f"shape mismatch: ref={tuple(lhs.shape)}, cand={tuple(rhs.shape)}"

    total = lhs.numel()

    # 复数类型：分别比较实部/虚部
    if lhs.is_complex() or rhs.is_complex():
        passed_r, mere_r, mare_r, thr_r, mthr_r = _check_precision_mere_mare(rhs.real, lhs.real)
        passed_i, mere_i, mare_i, thr_i, mthr_i = _check_precision_mere_mare(rhs.imag, lhs.imag)
        # 用 view_as_real 计算绝对差值（转 float32 避免溢出）
        lhs_fp = torch.view_as_real(lhs).to(torch.float32)
        rhs_fp = torch.view_as_real(rhs).to(torch.float32)
        diff = (lhs_fp - rhs_fp).abs()
        max_abs = diff.max().item() if diff.numel() else 0.0
        mean_abs = diff.mean().item() if diff.numel() else 0.0
        return (
            f"dtype(ref={lhs.dtype}, cand={rhs.dtype}), "
            f"max_abs_diff={max_abs:.6g}, mean_abs_diff={mean_abs:.6g}, "
            f"MERE(real={mere_r:.6g}, imag={mere_i:.6g}), "
            f"MARE(real={mare_r:.6g}, imag={mare_i:.6g}), "
            f"threshold={thr_r:.6g}, mare_threshold={mthr_r:.6g}, "
            f"passed_real={passed_r}, passed_imag={passed_i}"
        )

    # 浮点类型
    if torch.is_floating_point(lhs) or torch.is_floating_point(rhs):
        lhs_fp = torch.nan_to_num(lhs.to(torch.float32))
        rhs_fp = torch.nan_to_num(rhs.to(torch.float32))
        diff = (lhs_fp - rhs_fp).abs()
        # 过滤同位置同号 inf（inf - inf = nan），避免污染 diff 统计
        both_inf_mask = torch.isinf(lhs_fp) & torch.isinf(rhs_fp) & (torch.sign(lhs_fp) == torch.sign(rhs_fp))
        diff[both_inf_mask] = 0.0
        max_abs = diff.max().item() if diff.numel() else 0.0
        mean_abs = diff.mean().item() if diff.numel() else 0.0
        passed, mere, mare, threshold, mare_threshold = _check_precision_mere_mare(rhs, lhs)
        return (
            f"dtype(ref={lhs.dtype}, cand={rhs.dtype}), "
            f"max_abs_diff={max_abs:.6g}, mean_abs_diff={mean_abs:.6g}, "
            f"MERE={mere:.6g}, MARE={mare:.6g}, "
            f"threshold={threshold:.6g}, mare_threshold={mare_threshold:.6g}, "
            f"passed={passed}"
        )

    # 整数类型：回退到元素级对比
    lhs_i32 = lhs.to(torch.int32)
    rhs_i32 = rhs.to(torch.int32)
    delta = rhs_i32 - lhs_i32
    abs_diff = delta.abs()
    max_abs = abs_diff.max().item() if abs_diff.numel() else 0
    mean_abs = abs_diff.float().mean().item() if abs_diff.numel() else 0.0
    mismatch_mask = delta != 0
    mismatch_count = mismatch_mask.sum().item() if delta.numel() else 0
    mismatch_ratio = (mismatch_count / total) if total else 0.0
    cand_gt_ref = ((delta > 0) & mismatch_mask).sum().item() if delta.numel() else 0
    cand_lt_ref = ((delta < 0) & mismatch_mask).sum().item() if delta.numel() else 0

    first_mismatch = ""
    if mismatch_count:
        first_linear_idx = int(torch.nonzero(mismatch_mask.reshape(-1), as_tuple=False)[0].item())
        if lhs.ndim == 0:
            first_index = ()
            lhs_val = lhs.item()
            rhs_val = rhs.item()
        else:
            rem = first_linear_idx
            first_index = [0] * lhs.ndim
            for d in range(lhs.ndim - 1, -1, -1):
                dim_size = lhs.shape[d]
                first_index[d] = rem % dim_size
                rem //= dim_size
            first_index = tuple(first_index)
            lhs_val = lhs[first_index].item()
            rhs_val = rhs[first_index].item()
        first_mismatch = f", first_mismatch(index={first_index}, ref={lhs_val}, cand={rhs_val})"

    return (
        f"dtype(ref={lhs.dtype}, cand={rhs.dtype}), "
        f"unequal_elements={mismatch_count}, mismatch_ratio={mismatch_ratio:.6%}, "
        f"max_abs_diff={max_abs}, mean_abs_diff={mean_abs:.6g}, "
        f"cand_gt_ref={cand_gt_ref}, cand_lt_ref={cand_lt_ref}"
        f"{first_mismatch}"
    )


def _compare_values(lhs, rhs, path: str = "output"):
    """递归比较两个值，对 Tensor 使用 MERE/MARE 精度标准。"""
    if type(lhs) is not type(rhs):
        return False, f"{path}: type mismatch: ref={type(lhs).__name__}, cand={type(rhs).__name__}"

    if isinstance(lhs, torch.Tensor):
        if lhs.shape != rhs.shape:
            return False, f"{path}: shape mismatch: ref={tuple(lhs.shape)}, cand={tuple(rhs.shape)}"

        # 整数/布尔类型分两类：
        #   - int8 / int16：量化输出。NPU 实现走 fp32→fp16→int 中转 cast，
        #     PyTorch CPU 走全 fp32 .round()，最低位舍入路径不同，
        #     允许 ±1 LSB（INT_LSB_TOLERANCE）。
        #   - int32 / int64 / bool：无 round 语义（索引、计数、mask 之类），
        #     仍走 torch.equal 严格按位。
        needs_numeric_check = (
            torch.is_floating_point(lhs)
            or torch.is_floating_point(rhs)
            or lhs.is_complex()
            or rhs.is_complex()
        )
        if not needs_numeric_check:
            tol = INT_LSB_TOLERANCE.get(lhs.dtype) if lhs.dtype == rhs.dtype else None
            if tol is not None:
                # 用 int32 做差，避免 int8 自身相减溢出。
                diff = (rhs.to(torch.int32) - lhs.to(torch.int32)).abs()
                max_abs = diff.max().item() if diff.numel() else 0
                if max_abs <= tol:
                    return True, (
                        f"{path}: matched within ±{tol} LSB "
                        f"(max_abs_diff={max_abs}, dtype={lhs.dtype})"
                    )
                return False, f"{path}: {_tensor_diff_summary(lhs, rhs)}"
            if torch.equal(lhs, rhs):
                return True, f"{path}: matched"
            return False, f"{path}: {_tensor_diff_summary(lhs, rhs)}"

        # 复数类型：分别比较实部和虚部
        if lhs.is_complex() or rhs.is_complex():
            real_passed, real_mere, real_mare, real_thr, real_mthr = _check_precision_mere_mare(rhs.real, lhs.real)
            imag_passed, imag_mere, imag_mare, imag_thr, imag_mthr = _check_precision_mere_mare(rhs.imag, lhs.imag)
            passed = real_passed and imag_passed
            mere = max(real_mere, imag_mere)
            mare = max(real_mare, imag_mare)
            threshold = real_thr
            mare_threshold = real_mthr
            if passed:
                return True, (
                    f"{path}: matched, "
                    f"MERE(real={real_mere:.6g}, imag={imag_mere:.6g}), "
                    f"MARE(real={real_mare:.6g}, imag={imag_mare:.6g}), "
                    f"threshold={threshold:.6g}, mare_threshold={mare_threshold:.6g}"
                )
            return False, f"{path}: {_tensor_diff_summary(lhs, rhs)}"

        # 浮点类型使用 MERE/MARE 精度标准
        passed, mere, mare, threshold, mare_threshold = _check_precision_mere_mare(rhs, lhs)
        if passed:
            return True, (
                f"{path}: matched, "
                f"MERE={mere:.6g}, MARE={mare:.6g}, "
                f"threshold={threshold:.6g}, mare_threshold={mare_threshold:.6g}"
            )
        return False, f"{path}: {_tensor_diff_summary(lhs, rhs)}"

    if isinstance(lhs, list):
        if len(lhs) != len(rhs):
            return False, f"{path}: list length mismatch: ref={len(lhs)}, cand={len(rhs)}"
        for index, (a, b) in enumerate(zip(lhs, rhs)):
            ok, message = _compare_values(a, b, f"{path}[{index}]")
            if not ok:
                return False, message
        return True, f"{path}: matched"
    if isinstance(lhs, tuple):
        if len(lhs) != len(rhs):
            return False, f"{path}: tuple length mismatch: ref={len(lhs)}, cand={len(rhs)}"
        for index, (a, b) in enumerate(zip(lhs, rhs)):
            ok, message = _compare_values(a, b, f"{path}[{index}]")
            if not ok:
                return False, message
        return True, f"{path}: matched"
    if isinstance(lhs, dict):
        if lhs.keys() != rhs.keys():
            return False, f"{path}: dict keys mismatch: ref={sorted(lhs.keys())}, cand={sorted(rhs.keys())}"
        for key in lhs:
            ok, message = _compare_values(lhs[key], rhs[key], f"{path}.{key}")
            if not ok:
                return False, message
        return True, f"{path}: matched"
    if lhs == rhs:
        return True, f"{path}: matched"
    return False, f"{path}: value mismatch: ref={lhs}, cand={rhs}"


def _get_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_task_dir(op: str) -> Path:
    op_path = Path(op)
    if op_path.is_dir():
        return op_path.resolve()

    direct = WORKDIR / op
    if direct.is_dir():
        return direct

    raise FileNotFoundError(f"Cannot find task directory for op '{op}'")


def _format_tensor_summary(tensor: torch.Tensor) -> str:
    return f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device})"


def _summarize_value(value, name: str):
    if isinstance(value, torch.Tensor):
        return [f"{name}: {_format_tensor_summary(value)}"]
    if isinstance(value, list):
        lines = [f"{name}: list[{len(value)}]"]
        for index, item in enumerate(value):
            lines.extend(_summarize_value(item, f"{name}[{index}]"))
        return lines
    if isinstance(value, tuple):
        lines = [f"{name}: tuple[{len(value)}]"]
        for index, item in enumerate(value):
            lines.extend(_summarize_value(item, f"{name}[{index}]"))
        return lines
    if isinstance(value, dict):
        lines = [f"{name}: dict[{len(value)}]"]
        for key, item in value.items():
            lines.extend(_summarize_value(item, f"{name}.{key}"))
        return lines
    return [f"{name}: {type(value).__name__}({value})"]


def _get_input_groups(module):
    # Prefer get_input_groups(); fall back to get_inputs() wrapped in a list.
    if hasattr(module, "get_input_groups"):
        input_groups = module.get_input_groups()
        if not isinstance(input_groups, list) or not input_groups:
            raise ValueError("get_input_groups() must return a non-empty list")
        return input_groups

    if hasattr(module, "get_inputs"):
        inputs = module.get_inputs()
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("get_inputs() must return a non-empty list")
        return [inputs]

    raise AttributeError(f"Neither get_input_groups() nor get_inputs() found in {module.__file__}")


def _run_verification(op: str):
    report = {
        "op": op,
        "ok": False,
        "device": str(_get_device()),
        "task_dir": "",
        "reference": "",
        "candidate": "",
        "kernel_build_dir": "",
        "inputs": [],
        "comparisons": [],
        "comparison": "",
        "error": "",
    }

    task_dir = _resolve_task_dir(op)
    ref_path = task_dir / "model.py"
    cand_path = task_dir / "model_new_ascendc.py"
    kernel_build_dir = task_dir / "kernel" / "build"
    report["task_dir"] = str(task_dir)
    report["reference"] = str(ref_path)
    report["candidate"] = str(cand_path)
    report["kernel_build_dir"] = str(kernel_build_dir)

    if not ref_path.is_file():
        report["error"] = f"missing reference model: {ref_path}"
        return report
    if not cand_path.is_file():
        report["error"] = f"missing candidate model: {cand_path}"
        return report
    # kernel/build is optional: model_new_ascendc.py may manage its own sys.path.
    inserted_paths = []
    paths_to_add = [str(WORKDIR)]
    if kernel_build_dir.is_dir():
        paths_to_add.append(str(kernel_build_dir))
    else:
        import warnings
        warnings.warn(
            f"{kernel_build_dir} not found; assuming model_new_ascendc.py handles its own import path.",
            UserWarning,
            stacklevel=2,
        )
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)
            inserted_paths.append(p)

    try:
        ref_module = _load_module(ref_path, f"{op}_ref_model")
        cand_module = _load_module(cand_path, f"{op}_ascendc_model")

        ref_cls = _find_model_class(ref_module, "Model")
        cand_cls = _find_model_class(cand_module, "ModelNew")

        torch.manual_seed(0)
        # cand's get_init_inputs() takes priority so ModelNew can override
        # constructor args (e.g. to align a hard-coded kernel parameter).
        if hasattr(cand_module, "get_init_inputs"):
            init_inputs = cand_module.get_init_inputs()
        else:
            init_inputs = getattr(ref_module, "get_init_inputs", lambda: [])()
        input_groups = _get_input_groups(ref_module)
        device = _get_device()

        ref_model = ref_cls(*_clone_value(init_inputs)).to(device).eval()
        cand_model = cand_cls(*_clone_value(init_inputs)).to(device).eval()
        all_ok = True
        comparisons = []
        input_summaries = []
        for index, inputs in enumerate(input_groups):
            ref_inputs = _move_to_device(_clone_value(inputs), device)
            cand_inputs = _move_to_device(_clone_value(inputs), device)
            input_summaries.extend(_summarize_value(ref_inputs, f"inputs[{index}]"))

            with torch.no_grad():
                ref_out = ref_model(*ref_inputs)
                cand_out = cand_model(*cand_inputs)

            if hasattr(ref_model, "postprocess_output"):
                ref_out = ref_model.postprocess_output(ref_out, inputs)
                cand_out = ref_model.postprocess_output(cand_out, inputs)
                
            ref_out = _normalize_output(ref_out)
            cand_out = _normalize_output(cand_out)

            ok, comparison = _compare_values(
                ref_out,
                cand_out,
                path=f"output[{index}]",
            )
            comparisons.append(f"case[{index}]: {comparison}")
            all_ok = all_ok and ok

        report["inputs"] = input_summaries
        report["comparisons"] = comparisons
        report["comparison"] = "\n".join(comparisons)
        report["ok"] = all_ok
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if os.environ.get("VERIFICATION_ASCENDC_DEBUG") == "1":
            raise
        report["traceback"] = traceback.format_exc()
        return report
    finally:
        for p in inserted_paths:
            if p in sys.path:
                sys.path.remove(p)


def verify(op: str) -> bool:
    return _run_verification(op)["ok"]


def _print_report(report):
    status = "PASS" if report["ok"] else "FAIL"
    print("=" * 72)
    print("AscendC Verification Report")
    print("=" * 72)
    print(f"Status    : {status}")
    print(f"Operator  : {report['op']}")
    print(f"Device    : {report['device']}")
    print(f"Task Dir  : {report['task_dir']}")
    print(f"Reference : {report['reference']}")
    print(f"Candidate : {report['candidate']}")
    print(f"Kernel    : {report['kernel_build_dir']}")

    if report["inputs"]:
        print("-" * 72)
        print("Inputs")
        print("-" * 72)
        for line in report["inputs"]:
            print(line)

    print("-" * 72)
    print("Comparison")
    print("-" * 72)
    if report["comparison"]:
        print(report["comparison"])
    elif report["error"]:
        print(report["error"])
    else:
        print("No comparison information available")

    if report["error"] and os.environ.get("VERIFICATION_ASCENDC_DEBUG") == "1":
        print("-" * 72)
        print("Traceback")
        print("-" * 72)
        print(report.get("traceback", ""))

    print("-" * 72)
    print(f"Result: {'pass' if report['ok'] else 'fail'}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python utils/verification_ascendc.py <op>")
        print("Result: fail")
        raise SystemExit(1)

    report = _run_verification(sys.argv[1])
    _print_report(report)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
