"""Compute per-case performance ratios (reference vs ascendc) and optionally update A5_RESULTS.md.

Usage:
    # Single op — print summary
    python3 utils/perf_ratio.py output/npukernelbench/1_GELU

    # Single op — print + update A5_RESULTS.md
    python3 utils/perf_ratio.py output/npukernelbench/1_GELU --update-results

    # Batch — scan all completed ops under a parent directory
    python3 utils/perf_ratio.py output/npukernelbench --update-results

    # Custom warmup/repeat
    python3 utils/perf_ratio.py output/npukernelbench/1_GELU --warmup 5 --repeat 10
"""

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = SCRIPT_DIR.parent
RESULTS_PATH = WORKDIR / "benchmarks" / "NPUKernelBench" / "A5_RESULTS.md"

# Import performance.py from same directory
sys.path.insert(0, str(SCRIPT_DIR))
import performance as perf_module


def compute_ratios(op_dir: str, warmup: int = 5, repeat: int = 10, seed: int = 0) -> dict:
    """Run performance.py and compute per-case ratios."""
    report = perf_module._run_performance(op_dir, ["reference", "ascendc"], warmup, repeat, seed)

    ref_result = None
    asc_result = None
    for r in report["results"]:
        if r["impl"] == "reference":
            ref_result = r
        elif r["impl"] == "ascendc":
            asc_result = r

    if ref_result is None or not ref_result["ok"]:
        raise RuntimeError(f"Reference impl failed: {ref_result['error'] if ref_result else 'not found'}")
    if asc_result is None or not asc_result["ok"]:
        raise RuntimeError(f"AscendC impl failed: {asc_result['error'] if asc_result else 'not found'}")

    ratios = []
    case_details = []
    for ref_case, asc_case in zip(ref_result["case_results"], asc_result["case_results"]):
        if asc_case["mean_ms"] > 0:
            ratio = ref_case["mean_ms"] / asc_case["mean_ms"]
        else:
            ratio = float("inf")
        ratios.append(ratio)
        case_details.append({
            "index": ref_case["index"],
            "ref_mean_ms": round(ref_case["mean_ms"], 4),
            "asc_mean_ms": round(asc_case["mean_ms"], 4),
            "ratio": round(ratio, 4),
        })

    total = len(ratios)
    best = max(ratios)
    worst = min(ratios)
    mean = sum(ratios) / total if total > 0 else 0.0
    ge_06 = sum(1 for r in ratios if r >= 0.6)
    ge_10 = sum(1 for r in ratios if r >= 1.0)

    op_name = Path(op_dir).name

    return {
        "op": op_name,
        "op_dir": op_dir,
        "total_cases": total,
        "best": round(best, 2),
        "worst": round(worst, 2),
        "mean": round(mean, 2),
        "ge_06": ge_06,
        "ge_10": ge_10,
        "cases": case_details,
    }


def print_summary(result: dict):
    """Print a one-line + detailed summary."""
    op = result["op"]
    total = result["total_cases"]
    print(f"\n{'=' * 72}")
    print(f"  {op}: {total} cases, best={result['best']:.2f}x, worst={result['worst']:.2f}x, "
          f"mean={result['mean']:.2f}x, >=0.6x={result['ge_06']}/{total}, >=1.0x={result['ge_10']}/{total}")
    print(f"{'=' * 72}")

    # Per-case detail
    print(f"  {'Case':>6}  {'Ref(ms)':>10}  {'Asc(ms)':>10}  {'Ratio':>8}")
    print(f"  {'-' * 6}  {'-' * 10}  {'-' * 10}  {'-' * 8}")
    for c in result["cases"]:
        tag = " *" if c["ratio"] >= 1.0 else ""
        print(f"  {c['index']:>6}  {c['ref_mean_ms']:>10.4f}  {c['asc_mean_ms']:>10.4f}  {c['ratio']:>7.2f}x{tag}")
    print()


def update_results_md(result: dict, results_path: Path = RESULTS_PATH):
    """Update the matching row in A5_RESULTS.md Summary Table."""
    if not results_path.is_file():
        print(f"  [WARN] {results_path} not found, skipping update.")
        return False

    op_name = result["op"]
    # Extract problem ID: "1_GELU" -> 1, "13_Cat" -> 13
    match = re.match(r"^(\d+)_", op_name)
    if not match:
        print(f"  [WARN] Cannot extract problem ID from '{op_name}', skipping update.")
        return False
    problem_id = match.group(1)

    lines = results_path.read_text().splitlines()
    updated = False
    mean_icon = "✅" if result["mean"] >= 0.6 else "⚠️"
    total = result["total_cases"]

    for i, line in enumerate(lines):
        # Match table row: "| 1 | GELU |" or "| 13 | Cat |"
        row_pattern = rf"^\|\s*{problem_id}\s*\|"
        if re.match(row_pattern, line):
            cols = [c.strip() for c in line.split("|")]
            # cols: ['', '#', 'Problem', 'Type', 'Mode', 'Precision', 'Best', 'Worst', 'Mean', '>=0.6x', '>=1.0x', 'Status', '']
            if len(cols) >= 12:
                cols[6] = f"{result['best']:.2f}x"
                cols[7] = f"{result['worst']:.2f}x"
                cols[8] = f"**{result['mean']:.2f}x** {mean_icon}"
                cols[9] = f"{result['ge_06']}/{total}"
                cols[10] = f"{result['ge_10']}/{total}"
                if cols[11] == "Pending":
                    cols[11] = "Verified"
                lines[i] = "| " + " | ".join(cols[1:-1]) + " |"
                updated = True
                break

    if updated:
        results_path.write_text("\n".join(lines) + "\n")
        print(f"  [OK] Updated A5_RESULTS.md row for problem #{problem_id}")
    else:
        print(f"  [WARN] Row for problem #{problem_id} not found in {results_path}")

    return updated


def find_op_dirs(parent: Path) -> list[Path]:
    """Find all completed op directories under a parent (must have model.py + model_new_ascendc.py)."""
    dirs = []
    for d in sorted(parent.iterdir()):
        if d.is_dir() and (d / "model.py").is_file() and (d / "model_new_ascendc.py").is_file():
            dirs.append(d)
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Compute per-case perf ratios (ref vs ascendc)")
    parser.add_argument("op_dir", help="Op output directory, or parent dir for batch mode")
    parser.add_argument("--update-results", action="store_true", help="Update A5_RESULTS.md")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    parser.add_argument("--repeat", type=int, default=10, help="Repeat iterations (default: 10)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--json", action="store_true", help="Also print JSON output")
    args = parser.parse_args()

    op_path = Path(args.op_dir)

    # Determine single-op vs batch mode
    if (op_path / "model.py").is_file() and (op_path / "model_new_ascendc.py").is_file():
        op_dirs = [op_path]
    elif op_path.is_dir():
        op_dirs = find_op_dirs(op_path)
        if not op_dirs:
            print(f"No completed op directories found under {op_path}")
            sys.exit(1)
        print(f"Batch mode: found {len(op_dirs)} op(s): {[d.name for d in op_dirs]}")
    else:
        print(f"Not a valid directory: {op_path}")
        sys.exit(1)

    all_results = []
    for op_dir in op_dirs:
        print(f"\nRunning performance for {op_dir.name}...")
        try:
            result = compute_ratios(str(op_dir), args.warmup, args.repeat, args.seed)
            all_results.append(result)
            print_summary(result)

            if args.update_results:
                update_results_md(result)

        except Exception as e:
            print(f"  [ERROR] {op_dir.name}: {e}")

    if args.json and all_results:
        print("\n--- JSON ---")
        print(json.dumps(all_results, indent=2, ensure_ascii=False))

    # Final batch summary
    if len(all_results) > 1:
        print(f"\n{'=' * 72}")
        print(f"  Batch Summary ({len(all_results)} ops)")
        print(f"{'=' * 72}")
        for r in all_results:
            mean_tag = "✅" if r["mean"] >= 0.6 else "⚠️"
            print(f"  {r['op']:<28} mean={r['mean']:.2f}x {mean_tag}  "
                  f"best={r['best']:.2f}x  worst={r['worst']:.2f}x  "
                  f">=0.6x={r['ge_06']}/{r['total_cases']}")


if __name__ == "__main__":
    main()
