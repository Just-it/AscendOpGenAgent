#!/usr/bin/env python3
"""Driver: 对每个 case × {reference, ascendc} 都用独立子进程跑 profiler，避免互相干扰。

公平性保证：通过 torch_npu.profiler 解析 kernel_details.csv 拿到设备侧 kernel 总时延，
覆盖 PyTorch 内置 aten op (aclnnXxx) 和 AscendC 自定义 kernel，避免 host wall-time
中的 launch / dispatch 开销和不对等的 fallback 问题。

用法:
    python .claude/skills/performance-analyzer/scripts/lingxi_perf_driver.py \\
        --output_dir /path/to/task_dir \\
        --warmup 5 --active 20 --retry 2
"""
import argparse, json, os, re, statistics, subprocess, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_WORKER = SCRIPT_DIR / "lingxi_perf_single.py"


def pick_idle_npu(default=0):
    """解析 `npu-smi info` 选空闲的 NPU（按 AICore% + HBM/显存占用排序）。失败回退 default。"""
    try:
        p = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10)
        if p.returncode != 0:
            return default
    except Exception:
        return default
    # 兼容两种常见 npu-smi 输出（每卡两行）：
    #   row1: | <dev_id>  <Name>  | OK | Power  Temp  Hugepages |
    #   row2: | <chip>            | <Bus-Id> | AICore%  MemUsed/MemTot [HBMUsed/HBMTot] |
    devices = {}
    cur = None
    head_re = re.compile(r"^\|\s+(\d+)\s+\S+\s+\|\s+\w+\s+\|")
    # 通过 Bus-Id 判定第二行；AICore% 后紧跟 a/b [c/d] 数值
    bus_re = re.compile(r"^\|\s+\d+\s+\|\s+[0-9A-Fa-f:.]+\s+\|\s+(\d+)\s+(\d+)\s*/\s*(\d+)(?:\s+(\d+)\s*/\s*(\d+))?")
    for line in p.stdout.splitlines():
        m2 = bus_re.match(line)
        if m2 and cur is not None:
            aicore = int(m2.group(1))
            mem_used = int(m2.group(2)); mem_total = max(int(m2.group(3)), 1)
            hbm_used = int(m2.group(4)) if m2.group(4) else 0
            hbm_total = max(int(m2.group(5)), 1) if m2.group(5) else 1
            mem_ratio = max(mem_used / mem_total, hbm_used / hbm_total)
            devices[cur] = (aicore, mem_ratio)
            cur = None
            continue
        m1 = head_re.match(line)
        if m1:
            cur = int(m1.group(1))
    if not devices:
        return default
    best_id, _ = min(devices.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    return best_id


def run_single(out_dir, idx, impl, warmup, active, device_id):
    cmd = [
        sys.executable, str(SINGLE_WORKER),
        "--output_dir", str(out_dir),
        "--case_idx", str(idx),
        "--impl", impl,
        "--warmup", str(warmup),
        "--active", str(active),
    ]
    env = os.environ.copy()
    # 子进程内 `torch.device("npu")` 会映射到这里指定的可见设备（逻辑 index 0）
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT_JSON__"):
            return json.loads(line[len("__RESULT_JSON__"):])
    return {"case_idx": idx, "impl": impl, "avg_kernel_ms": None,
            "error": f"no_result; stderr_tail={proc.stderr[-300:]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--active", type=int, default=20)
    ap.add_argument("--retry", type=int, default=2,
                    help="重试次数（解析失败时）")
    ap.add_argument("--output", help="输出 JSON 报告路径（供 agent 下游消费）")
    ap.add_argument("--device", type=int, default=None,
                    help="NPU 设备 id；未指定时自动选择空闲卡（按 AICore%% 最低）")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()

    # 设备选择：显式 --device > 环境变量 ASCEND_RT_VISIBLE_DEVICES > 自动挑空闲卡
    if args.device is not None:
        device_id = args.device
        device_src = "cli"
    elif os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        device_id = int(os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")[0])
        device_src = "env"
    else:
        device_id = pick_idle_npu(default=0)
        device_src = "auto"
    print(f"[INFO] Using NPU device {device_id} (source={device_src})")

    json_files = sorted(out_dir.glob("*.json"))
    json_path = next((f for f in json_files if not f.name.endswith(".bak")), None)
    cases = []
    if json_path:
        with open(json_path) as f:
            cases = [json.loads(line) for line in f if line.strip()]
    n_cases = len(cases)

    print("=" * 100)
    print(f"Kernel-level Performance: {out_dir.name}  (warmup={args.warmup}, active={args.active})")
    print("=" * 100)
    print(f"{'Case':<5} {'Shape':<35} {'dtype':<10} {'Ref(ms)':>12} {'Asc(ms)':>12} {'Speedup':>10}")
    print("-" * 100)

    rows = []
    speedups = []
    for idx in range(n_cases):
        ref_res, asc_res = None, None
        for attempt in range(1 + args.retry):
            ref_res = run_single(out_dir, idx, "reference", args.warmup, args.active, device_id)
            if ref_res.get("avg_kernel_ms") is not None:
                break
        for attempt in range(1 + args.retry):
            asc_res = run_single(out_dir, idx, "ascendc", args.warmup, args.active, device_id)
            if asc_res.get("avg_kernel_ms") is not None:
                break

        shape = str(cases[idx]["inputs"][0]["shape"])
        dtype = cases[idx]["inputs"][0]["dtype"]
        ref_ms = ref_res.get("avg_kernel_ms")
        asc_ms = asc_res.get("avg_kernel_ms")

        if ref_ms is not None and asc_ms is not None and asc_ms > 0:
            sp = ref_ms / asc_ms
            speedups.append(sp)
            print(f"{idx:<5} {shape:<35} {dtype:<10} {ref_ms:>12.6f} {asc_ms:>12.6f} {sp:>9.2f}x")
        else:
            print(f"{idx:<5} {shape:<35} {dtype:<10} "
                  f"{'N/A' if ref_ms is None else f'{ref_ms:.6f}':>12} "
                  f"{'N/A' if asc_ms is None else f'{asc_ms:.6f}':>12} "
                  f"{'N/A':>10}  (ref_err={ref_res.get('error')}, asc_err={asc_res.get('error')})")

        rows.append({"case": idx, "shape": shape, "dtype": dtype,
                     "ref_kernel_ms": ref_ms, "asc_kernel_ms": asc_ms,
                     "speedup": (ref_ms / asc_ms) if (ref_ms and asc_ms and asc_ms > 0) else None,
                     "ref_breakdown": ref_res.get("breakdown") if isinstance(ref_res.get("breakdown"), dict) else None,
                     "asc_breakdown": asc_res.get("breakdown") if isinstance(asc_res.get("breakdown"), dict) else None,
                     "ref_error": ref_res.get("error"),
                     "asc_error": asc_res.get("error")})

    summary = {
        "task": out_dir.name,
        "task_dir": str(out_dir),
        "n_cases_total": n_cases,
        "n_cases_valid": len(speedups),
        "geomean_speedup": statistics.geometric_mean(speedups) if speedups else None,
        "mean_speedup": statistics.mean(speedups) if speedups else None,
        "median_speedup": statistics.median(speedups) if speedups else None,
        "min_speedup": min(speedups) if speedups else None,
        "max_speedup": max(speedups) if speedups else None,
        "warmup": args.warmup,
        "active": args.active,
        "device_id": device_id,
        "device_select_source": device_src,
        "timing_method": "torch_npu.profiler.kernel_details",
        "per_case": rows,
    }

    print("-" * 100)
    if speedups:
        print(f"Mean speedup    : {summary['mean_speedup']:.2f}x")
        print(f"Geomean speedup : {summary['geomean_speedup']:.2f}x  ← 主指标")
        print(f"Median speedup  : {summary['median_speedup']:.2f}x")
        print(f"Valid cases     : {len(speedups)}/{n_cases}")
    print("=" * 100)
    print("\nKernel breakdown:")
    for r in rows:
        if r["ref_breakdown"] or r["asc_breakdown"]:
            print(f"\n[case {r['case']}] {r['shape']} {r['dtype']}")
            if r["ref_breakdown"]:
                print("  REF kernels:")
                for k, v in r["ref_breakdown"].items():
                    print(f"    {k:<45} calls={v['calls']:<3} total={v['total_us']:>8.2f}us  avg={v['avg_us']:>7.3f}us")
            if r["asc_breakdown"]:
                print("  ASC kernels:")
                for k, v in r["asc_breakdown"].items():
                    print(f"    {k:<45} calls={v['calls']:<3} total={v['total_us']:>8.2f}us  avg={v['avg_us']:>7.3f}us")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] JSON report saved to: {out_path}")


if __name__ == "__main__":
    main()
