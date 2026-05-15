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


def _tensor_diff_summary(lhs: torch.Tensor, rhs: torch.Tensor, atol: float = 0.0, rtol: float = 0.0):
    if lhs.shape != rhs.shape:
        return f"shape mismatch: ref={tuple(lhs.shape)}, cand={tuple(rhs.shape)}"

    lhs_fp = torch.nan_to_num(lhs.to(torch.float32))
    rhs_fp = torch.nan_to_num(rhs.to(torch.float32))
    diff = (lhs_fp - rhs_fp).abs()
    allowed = atol + rtol * rhs_fp.abs()
    mismatch_mask = diff > allowed
    mismatch_count = mismatch_mask.sum().item() if diff.numel() else 0
    total = lhs.numel()
    mismatch_ratio = (mismatch_count / total) if total else 0.0
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff[mismatch_mask].mean().item() if mismatch_count else 0.0

    if torch.is_floating_point(lhs) or torch.is_floating_point(rhs):
        return (
            f"dtype(ref={lhs.dtype}, cand={rhs.dtype}), "
            f"unequal_elements={mismatch_count}, mismatch_ratio={mismatch_ratio:.6%}, "
            f"max_abs_diff={max_abs:.6g}, mean_abs_diff={mean_abs:.6g}"
        )

    lhs_i32 = lhs.to(torch.int32)
    rhs_i32 = rhs.to(torch.int32)
    delta = rhs_i32 - lhs_i32
    abs_diff = delta.abs()
    max_abs = abs_diff.max().item() if abs_diff.numel() else 0
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


def _compare_values(lhs, rhs, atol: float, rtol: float, path: str = "output"):
    if type(lhs) is not type(rhs):
        return False, f"{path}: type mismatch: ref={type(lhs).__name__}, cand={type(rhs).__name__}"

    if isinstance(lhs, torch.Tensor):
        if lhs.shape != rhs.shape:
            return False, f"{path}: shape mismatch: ref={tuple(lhs.shape)}, cand={tuple(rhs.shape)}"
        lhs_fp = torch.nan_to_num(lhs.to(torch.float32))
        rhs_fp = torch.nan_to_num(rhs.to(torch.float32))
        if torch.allclose(lhs_fp, rhs_fp, atol=atol, rtol=rtol):
            return True, f"{path}: matched"
        return False, f"{path}: {_tensor_diff_summary(lhs, rhs, atol=atol, rtol=rtol)}"

    if isinstance(lhs, list):
        if len(lhs) != len(rhs):
            return False, f"{path}: list length mismatch: ref={len(lhs)}, cand={len(rhs)}"
        for index, (a, b) in enumerate(zip(lhs, rhs)):
            ok, message = _compare_values(a, b, atol, rtol, f"{path}[{index}]")
            if not ok:
                return False, message
        return True, f"{path}: matched"
    if isinstance(lhs, tuple):
        if len(lhs) != len(rhs):
            return False, f"{path}: tuple length mismatch: ref={len(lhs)}, cand={len(rhs)}"
        for index, (a, b) in enumerate(zip(lhs, rhs)):
            ok, message = _compare_values(a, b, atol, rtol, f"{path}[{index}]")
            if not ok:
                return False, message
        return True, f"{path}: matched"
    if isinstance(lhs, dict):
        if lhs.keys() != rhs.keys():
            return False, f"{path}: dict keys mismatch: ref={sorted(lhs.keys())}, cand={sorted(rhs.keys())}"
        for key in lhs:
            ok, message = _compare_values(lhs[key], rhs[key], atol, rtol, f"{path}.{key}")
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
    if not hasattr(module, "get_input_groups"):
        raise AttributeError(f"get_input_groups() not found in {module.__file__}")

    input_groups = module.get_input_groups()
    if not isinstance(input_groups, list) or not input_groups:
        raise ValueError("get_input_groups() must return a non-empty list")
    return input_groups


def _run_verification(op: str):
    report = {
        "op": op,
        "ok": False,
        "device": str(_get_device()),
        "task_dir": "",
        "reference": "",
        "candidate": "",
        "atol": 1e-2,
        "rtol": 1e-2,
        "inputs": [],
        "comparisons": [],
        "comparison": "",
        "error": "",
    }

    task_dir = _resolve_task_dir(op)
    ref_path = task_dir / "model.py"
    cand_path = task_dir / "model_new_tilelang.py"
    report["task_dir"] = str(task_dir)
    report["reference"] = str(ref_path)
    report["candidate"] = str(cand_path)

    if not ref_path.is_file():
        report["error"] = f"missing reference model: {ref_path}"
        return report
    if not cand_path.is_file():
        report["error"] = f"missing candidate model: {cand_path}"
        return report

    sys.path.insert(0, str(WORKDIR))
    try:
        ref_module = _load_module(ref_path, f"{op}_ref_model")
        cand_module = _load_module(cand_path, f"{op}_tilelang_model")

        ref_cls = _find_model_class(ref_module, "Model")
        cand_cls = _find_model_class(cand_module, "ModelNew")

        torch.manual_seed(0)
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

            ref_out = _normalize_output(ref_out)
            cand_out = _normalize_output(cand_out)

            atol = report["atol"]
            rtol = report["rtol"]
            if _contains_int8_tensor(ref_out) and _contains_int8_tensor(cand_out):
                atol = 1.5
                rtol = 0.0

            ok, comparison = _compare_values(
                ref_out,
                cand_out,
                atol=atol,
                rtol=rtol,
                path=f"output[{index}]",
            )
            comparisons.append(f"case[{index}]: {comparison}")
            all_ok = all_ok and ok
            report["atol"] = max(report["atol"], atol)
            report["rtol"] = min(report["rtol"], rtol) if rtol == 0.0 else report["rtol"]

        report["inputs"] = input_summaries
        report["comparisons"] = comparisons
        report["comparison"] = "\n".join(comparisons)
        report["ok"] = all_ok
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if os.environ.get("VERIFICATION_TILELANG_DEBUG") == "1":
            raise
        report["traceback"] = traceback.format_exc()
        return report
    finally:
        if str(WORKDIR) in sys.path:
            sys.path.remove(str(WORKDIR))


def verify(op: str) -> bool:
    return _run_verification(op)["ok"]


def _print_report(report):
    status = "PASS" if report["ok"] else "FAIL"
    print("=" * 72)
    print("TileLang Verification Report")
    print("=" * 72)
    print(f"Status    : {status}")
    print(f"Operator  : {report['op']}")
    print(f"Device    : {report['device']}")
    print(f"Task Dir  : {report['task_dir']}")
    print(f"Reference : {report['reference']}")
    print(f"Candidate : {report['candidate']}")
    if report["atol"] == 1.5:
        print(f"Tolerance : atol={report['atol']}")
    else:
        print(f"Tolerance : atol={report['atol']}, rtol={report['rtol']}")

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

    if report["error"] and os.environ.get("VERIFICATION_TILELANG_DEBUG") == "1":
        print("-" * 72)
        print("Traceback")
        print("-" * 72)
        print(report.get("traceback", ""))

    print("-" * 72)
    print(f"Result: {'pass' if report['ok'] else 'fail'}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python .claude/skills/tilelang-designer/scripts/verification_tilelang.py <op>")
        print("Result: fail")
        raise SystemExit(1)

    report = _run_verification(sys.argv[1])
    _print_report(report)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
