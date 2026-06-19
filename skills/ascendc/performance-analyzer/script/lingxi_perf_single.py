#!/usr/bin/env python3
"""单 case kernel-level 性能测试（被 perf_kernel_driver.py 通过子进程调用）。
输出 JSON 到 stdout 最后一行。"""
import argparse, importlib.util, inspect, json, os, shutil, sys, time
from pathlib import Path
import torch
import torch.nn as nn


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _find_cls(module, preferred):
    c = getattr(module, preferred, None)
    if inspect.isclass(c) and issubclass(c, nn.Module):
        return c
    for _, v in vars(module).items():
        if inspect.isclass(v) and issubclass(v, nn.Module) and v is not nn.Module:
            return v
    raise AttributeError("no nn.Module")


def _move(v, d):
    if isinstance(v, torch.Tensor):
        return v.to(d)
    if isinstance(v, (list, tuple)):
        return type(v)(_move(x, d) for x in v)
    return v


def _clone(v):
    if isinstance(v, torch.Tensor):
        return v.clone()
    if isinstance(v, (list, tuple)):
        return type(v)(_clone(x) for x in v)
    return v


def _find_file(root, name):
    for r, _, files in os.walk(root):
        if name in files:
            return os.path.join(r, name)
    return None


def _parse_kernels(profile_path, active):
    csv_path = _find_file(profile_path, "kernel_details.csv")
    if not csv_path:
        return None, "no_csv"
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, f"read_err:{e}"
    if df.empty or "Duration(us)" not in df.columns:
        return None, "empty_or_bad"
    total_us = df["Duration(us)"].astype(float).sum()
    avg_ms = (total_us / active) / 1000.0
    breakdown = {}
    for n, g in df.groupby("Name"):
        breakdown[str(n)] = {
            "calls": int(len(g)),
            "total_us": float(g["Duration(us)"].astype(float).sum()),
            "avg_us": float(g["Duration(us)"].astype(float).mean()),
        }
    return avg_ms, breakdown


def _profile(model, inputs, warmup, active, tag):
    import torch_npu

    with torch.no_grad():
        _ = model(*inputs)
    torch.npu.synchronize()

    profile_path = f"/tmp/perfk_{tag}_{int(time.time()*1000)}_{os.getpid()}"
    if os.path.exists(profile_path):
        shutil.rmtree(profile_path, ignore_errors=True)

    exp = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=None,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    skip_first = 1
    total = skip_first + warmup + active

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU,
                    torch_npu.profiler.ProfilerActivity.CPU],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=warmup, active=active, repeat=1, skip_first=skip_first
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=False,
        experimental_config=exp,
    ) as prof:
        for _ in range(total):
            with torch.no_grad():
                _ = model(*inputs)
            prof.step()
            torch.npu.synchronize()

    avg_ms, breakdown = None, "init"
    for _ in range(15):
        time.sleep(1.0)
        avg_ms, breakdown = _parse_kernels(profile_path, active)
        if avg_ms is not None:
            break
    shutil.rmtree(profile_path, ignore_errors=True)
    return avg_ms, breakdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--case_idx", type=int, required=True)
    ap.add_argument("--impl", choices=["reference", "ascendc"], required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--active", type=int, default=20)
    ap.add_argument("--device", type=int, default=None,
                    help="NPU 设备 id；默认跟随 ASCEND_RT_VISIBLE_DEVICES，否则使用 npu:0")
    args = ap.parse_args()

    # 若显式传入 --device，覆写环境变量；driver 已通过 env 传入时会命中这一行之前
    if args.device is not None:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)

    out_dir = Path(args.output_dir).resolve()
    sys.path.insert(0, str(out_dir / "kernel" / "build"))
    sys.path.insert(0, str(out_dir))

    if args.impl == "reference":
        mod = _load(out_dir / "model.py", "ref_mod")
        cls = _find_cls(mod, "Model")
    else:
        mod = _load(out_dir / "model_new_ascendc.py", "asc_mod")
        cls = _find_cls(mod, "ModelNew")

    init_args = getattr(_load(out_dir / "model.py", "ref_for_init"),
                        "get_init_inputs", lambda: [])()
    input_groups = _load(out_dir / "model.py", "ref_for_inputs").get_input_groups()

    device = torch.device("npu")
    torch.manual_seed(0)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed(0)

    model = cls(*_clone(init_args)).to(device).eval()
    inputs = _move(_clone(input_groups[args.case_idx]), device)

    avg_ms, breakdown = _profile(model, inputs, args.warmup, args.active,
                                 f"{args.impl}_c{args.case_idx}")

    result = {
        "case_idx": args.case_idx,
        "impl": args.impl,
        "avg_kernel_ms": avg_ms,
        "breakdown": breakdown if isinstance(breakdown, dict) else None,
        "error": None if avg_ms is not None else str(breakdown),
    }
    print("__RESULT_JSON__" + json.dumps(result))


if __name__ == "__main__":
    main()
