#!/usr/bin/env python3
"""
Forge adapter for ops-evo evaluation.

This adapter prefers the structured `forge_raw_result.json` emitted by the
forge-side Z-Search export scripts, then converts it into the same
`evaluation_results.json` shape produced by `evaluate_ops.py`.

If the structured raw result is unavailable, it falls back to the previous
stdout-timing parsing behavior so older forge configs keep working.
"""

import argparse
import csv
import fnmatch
import glob
import json
import logging
import math
import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


DEFAULT_ZSEARCH_OP_FILTER_MAP = (
    Path(__file__).resolve().parent.parent / "config" / "zsearch_op_filter_map.json"
)


def write_forge_args_json(output_path: str, args_map: dict[str, str]) -> str:
    """Write a flat forge `--arg @file.json` payload."""
    payload = {key: str(value) for key, value in args_map.items() if value is not None and str(value) != ""}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def resolve_zsearch_op_filter(
    op_name: str,
    explicit_filter: str = "",
    filter_map_path: str | None = None,
) -> str:
    """Resolve the profiling CSV OP Type filter for forge."""
    if explicit_filter:
        return explicit_filter

    path = Path(filter_map_path) if filter_map_path else DEFAULT_ZSEARCH_OP_FILTER_MAP
    mapping = _load_json(str(path)) or {}
    mapped = mapping.get(op_name)
    if isinstance(mapped, str) and mapped:
        return mapped
    return op_name


def update_forge_args_json(output_path: str, updates: dict[str, str]) -> str:
    """Update an existing forge arg file in place."""
    payload = _load_json(output_path) or {}
    for key, value in updates.items():
        if value is None or str(value) == "":
            payload.pop(key, None)
        else:
            payload[key] = str(value)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def run_forge_command(
    forge_bin: str,
    args: list[str],
    config_dir: str,
    arg_file: str | None = None,
) -> subprocess.CompletedProcess:
    """Run a forge CLI command and return the CompletedProcess."""
    cmd = [forge_bin, "--config-dir", config_dir] + args
    if arg_file:
        cmd.extend(["--arg", f"@{arg_file}"])
    logging.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        logging.error("forge command failed (rc=%s)", result.returncode)
        logging.error("stderr: %s", result.stderr[-2000:])
    return result


def forge_build(
    forge_bin: str,
    config_dir: str,
    config_name: str,
    arg_file: str | None = None,
) -> bool:
    """Run forge build. Returns True on success."""
    args = ["build", config_name]
    result = run_forge_command(forge_bin, args, config_dir, arg_file)
    return result.returncode == 0


def forge_accuracy_test(
    forge_bin: str,
    config_dir: str,
    config_name: str,
    arg_file: str | None = None,
) -> tuple[bool, str]:
    """Run forge accuracy workflow and return structured pass/fail info."""
    args = ["workflow", config_name, "accuracy_test"]
    result = run_forge_command(forge_bin, args, config_dir, arg_file)
    if result.returncode == 0:
        return True, "PASS"

    stdout = _strip_ansi(result.stdout).strip()
    stderr = _strip_ansi(result.stderr).strip()
    summary = stderr or stdout or "forge accuracy_test failed"
    summary = re.sub(r"\s+", " ", summary)
    if len(summary) > 300:
        summary = summary[:297] + "..."
    return False, summary


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_forge_perf_output(stdout: str, op_name: str) -> dict | None:
    """
    Parse forge stdout to extract per-operator timing results.

    Legacy forge output may print lines like:
        op_name: 123.45
    """
    results = {}
    for line in stdout.splitlines():
        clean = _strip_ansi(line.strip())
        if ":" not in clean:
            continue
        key, _, val_str = clean.partition(":")
        try:
            results[key.strip()] = float(val_str.strip())
        except ValueError:
            continue
    if not results:
        return None
    if op_name in results:
        return {op_name: results[op_name]}
    return results


def _parse_raw_result_path_from_stdout(stdout: str) -> str | None:
    for line in stdout.splitlines():
        clean = _strip_ansi(line.strip())
        match = re.search(r"ZSEARCH_RAW_RESULT=(.+)", clean)
        if match:
            return match.group(1).strip()
    return None


def _find_latest_raw_result(install_path: str, started_at: float) -> str | None:
    pattern = os.path.join(install_path, "forge_zsearch", "*", "forge_raw_result.json")
    matched = [
        path for path in glob.glob(pattern)
        if os.path.isfile(path) and os.path.getmtime(path) >= started_at - 5
    ]
    if not matched:
        return None
    return max(matched, key=os.path.getmtime)


def _load_json(path: str | None) -> dict | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def resolve_forge_profiling_dir(
    forge_config_dir: str,
    forge_config: str,
    forge_test_path: str | None = None,
) -> str | None:
    if forge_test_path:
        if os.path.isabs(forge_test_path):
            return os.path.abspath(os.path.normpath(forge_test_path))
        return os.path.abspath(os.path.normpath(os.path.join(forge_config_dir, forge_test_path)))

    config_path = None
    for suffix in (".json", ""):
        candidate = os.path.join(forge_config_dir, forge_config + suffix)
        if os.path.isfile(candidate):
            config_path = candidate
            break
    if not config_path:
        return None

    try:
        cfg = _load_json(config_path) or {}
        working_dir = cfg.get("working_dir", "") or ""
        if not working_dir:
            working_dir = forge_config_dir
        elif not os.path.isabs(working_dir):
            working_dir = os.path.normpath(os.path.join(forge_config_dir, working_dir))
        test_path = ((cfg.get("vars") or {}).get("test_path") or "").strip()
        if not test_path:
            return None
        if os.path.isabs(test_path):
            return os.path.abspath(os.path.normpath(test_path))
        return os.path.abspath(os.path.normpath(os.path.join(working_dir, test_path)))
    except (AttributeError, json.JSONDecodeError, OSError, TypeError):
        return None


def _normalize_op_name(value: str) -> str:
    return re.sub(r"[_\s]+", "", value or "").lower()


def _profiling_row_op_name(row: dict) -> str:
    for key in ("OP Type", "Op Type", "op_type", "OpType", "op_name", "OP Name", "Op Name"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def _profiling_row_duration_us(row: dict) -> float | None:
    for key in ("Task Duration(us)", "Task Duration (us)", "task_duration_us", "Task Duration"):
        value = row.get(key)
        parsed = _maybe_float(value.strip() if isinstance(value, str) else value)
        if parsed is not None:
            return parsed
    return None


def _profiling_op_matches(candidate: str, target: str) -> bool:
    norm_candidate = _normalize_op_name(candidate)
    norm_target = _normalize_op_name(target)
    if not norm_candidate or not norm_target:
        return False
    return (
        norm_candidate == norm_target
        or norm_target in norm_candidate
        or norm_candidate in norm_target
    )


def extract_perf_from_profiling_dir(profiling_dir: str, op_name: str) -> dict | None:
    if not profiling_dir or not os.path.isdir(profiling_dir):
        return None

    pattern = os.path.join(
        profiling_dir,
        "PROF_*",
        "mindstudio_profiler_output",
        "op_summary_*.csv",
    )
    matched_samples: dict[str, list[float]] = {}
    for csv_path in glob.glob(pattern):
        if not os.path.isfile(csv_path):
            continue
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    candidate_name = _profiling_row_op_name(row)
                    duration_us = _profiling_row_duration_us(row)
                    if not candidate_name or duration_us is None:
                        continue
                    if _profiling_op_matches(candidate_name, op_name):
                        stable_name = op_name or candidate_name
                        matched_samples.setdefault(stable_name, []).append(duration_us)
        except OSError:
            continue

    if not matched_samples:
        return None

    return {
        matched_name: sum(samples) / len(samples)
        for matched_name, samples in matched_samples.items()
        if samples
    } or None


def _matches_op_filter(op_type: str, op_filter: str) -> bool:
    norm_key = op_type.lower().replace("_", "")
    norm_pat = op_filter.lower().replace("_", "")
    return fnmatch.fnmatch(norm_key, norm_pat)


def _read_trial_rows(csv_path: str, op_filter: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            op_name = _profiling_row_op_name(row)
            if not op_name or not _matches_op_filter(op_name, op_filter):
                continue
            duration_us = _profiling_row_duration_us(row)
            if duration_us is None:
                continue
            normalized_row = dict(row)
            normalized_row["_parsed_duration_us"] = duration_us
            rows.append(normalized_row)
        return rows


def _clean_trial_records_iqr(trial_records: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(trial_records) < 4:
        return list(trial_records), []

    sorted_samples = sorted(record["sample"] for record in trial_records)
    q1 = sorted_samples[len(sorted_samples) // 4]
    q3 = sorted_samples[3 * len(sorted_samples) // 4]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered = [record for record in trial_records if lower <= record["sample"] <= upper]
    removed = [record for record in trial_records if record["sample"] < lower or record["sample"] > upper]

    if len(filtered) < 3:
        return list(trial_records), []
    return filtered, removed


def _select_representative_trial(trial_records: list[dict]) -> dict | None:
    if not trial_records:
        return None
    sorted_records = sorted(trial_records, key=lambda record: record["sample"])
    return sorted_records[len(sorted_records) // 2]


def _maybe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_pipeline_from_rows(rows: list[dict]) -> dict:
    column_aliases = {
        "vec_ratio": ["vec_ratio", "aiv_vec_ratio"],
        "scalar_ratio": ["scalar_ratio", "aiv_scalar_ratio", "aic_scalar_ratio"],
        "mte2_ratio": ["mte2_ratio", "aiv_mte2_ratio", "aic_mte2_ratio"],
        "mte3_ratio": ["mte3_ratio", "aiv_mte3_ratio"],
        "vec_pct": ["vec_pct", "aiv_vec_ratio"],
        "scalar_pct": ["scalar_pct", "aiv_scalar_ratio", "aic_scalar_ratio"],
        "mte2_pct": ["mte2_pct", "aiv_mte2_ratio", "aic_mte2_ratio"],
        "mte3_pct": ["mte3_pct", "aiv_mte3_ratio"],
    }
    pipeline = {}
    for key, aliases in column_aliases.items():
        for row in reversed(rows):
            for alias in aliases:
                value = _maybe_float(row.get(alias))
                if value is not None:
                    pipeline[key] = value
                    break
            if key in pipeline:
                break
    return pipeline


def _run_perf_workflow(
    forge_bin: str,
    config_dir: str,
    config_name: str,
    arg_file: str | None,
    max_retries: int = 5,
) -> tuple[str, subprocess.CompletedProcess]:
    workflow_name = "performance_test"
    result = None
    for attempt in range(1, max_retries + 1):
        args = ["workflow", config_name, workflow_name]
        result = run_forge_command(forge_bin, args, config_dir, arg_file)
        if result.returncode == 0:
            return workflow_name, result
        logging.warning(
            "Perf workflow attempt %d/%d failed (rc=%s)",
            attempt, max_retries, result.returncode,
        )
    assert result is not None
    return workflow_name, result


def forge_perf_test(
    forge_bin: str,
    config_dir: str,
    config_name: str,
    op_name: str,
    install_path: str,
    arg_file: str | None = None,
) -> tuple[dict | None, str | None, dict | None]:
    """
    Run forge performance workflow.

    Returns:
        raw_result: loaded forge_raw_result.json if found
        raw_result_path: file path to forge_raw_result.json if found
        legacy_perf_results: parsed stdout timings for backward compatibility
    """
    started_at = time.time()
    _, result = _run_perf_workflow(forge_bin, config_dir, config_name, arg_file)
    if result.returncode != 0:
        return None, None, None

    raw_result_path = _parse_raw_result_path_from_stdout(result.stdout)
    if not raw_result_path:
        raw_result_path = _find_latest_raw_result(install_path, started_at)

    raw_result = _load_json(raw_result_path)
    legacy_results = parse_forge_perf_output(result.stdout, op_name)
    if raw_result is None and legacy_results is None:
        forge_args = _load_json(arg_file) or {}
        profiling_dir = resolve_forge_profiling_dir(
            config_dir,
            config_name,
            forge_test_path=forge_args.get("test_path"),
        )
        legacy_results = extract_perf_from_profiling_dir(profiling_dir, op_name) if profiling_dir else None
    return raw_result, raw_result_path, legacy_results


def _compute_cv_pct(samples: list[float]) -> float:
    if len(samples) < 2:
        return 0.0
    mean = sum(samples) / len(samples)
    if math.isclose(mean, 0.0):
        return 0.0
    return statistics.pstdev(samples) / mean * 100.0


def _measurement_quality(cv_pct: float) -> str:
    if cv_pct < 5.0:
        return "good"
    if cv_pct < 15.0:
        return "acceptable"
    return "noisy"


def _normalize_pipeline(pipeline: dict | None) -> dict:
    if not isinstance(pipeline, dict):
        return {}
    allowed = {
        "mte2_ratio",
        "vec_ratio",
        "scalar_ratio",
        "mte3_ratio",
        "mte2_pct",
        "vec_pct",
        "scalar_pct",
        "mte3_pct",
    }
    return {key: value for key, value in pipeline.items() if key in allowed}


def _infer_bottleneck(pipeline: dict) -> str:
    if not pipeline:
        return "unknown"
    mte2 = pipeline.get("mte2_pct", pipeline.get("mte2_ratio", 0))
    vec = pipeline.get("vec_pct", pipeline.get("vec_ratio", 0))
    scalar = pipeline.get("scalar_pct", pipeline.get("scalar_ratio", 0))
    if mte2 > 50:
        return "memory_bound"
    if vec > 60:
        return "compute_bound"
    if scalar > 30:
        return "scalar_bound"
    return "balanced"


def _build_side_result(
    raw_side: dict | None,
    tag: str,
    op_name: str = "",
    fallback_time_us: float = -1.0,
    fallback_install_path: str = "",
    default_precision_passed: bool = True,
    default_correctness_message: str = "PASS",
    allow_time_fallback: bool = True,
) -> dict:
    raw_side = raw_side or {}
    if raw_side.get("build_success") is False:
        raw_side = dict(raw_side)
        raw_side["raw_op_summary_files"] = []
        allow_time_fallback = False
    op_filter = raw_side.get("op_filter") or raw_side.get("op_name") or op_name
    warmup_count = int(raw_side.get("warmup_count", 0) or 0)
    has_csv_files = any(
        os.path.isfile(f) for f in (raw_side.get("raw_op_summary_files") or []) if f
    )
    if warmup_count == 0 and has_csv_files:
        logging.warning(
            "[%s] warmup_count=0: cold-start samples will be included in "
            "performance measurement. This may inflate time_us. "
            "Ensure forge config sets warmup_count >= 3.",
            tag,
        )
    trial_records = []
    for csv_path in raw_side.get("raw_op_summary_files", []) or []:
        if not os.path.isfile(csv_path):
            logging.warning("forge raw result csv missing: %s", csv_path)
            continue
        rows_all = _read_trial_rows(csv_path, op_filter) if op_filter else []
        if not rows_all:
            logging.warning("forge raw result csv has no rows for %s: %s", op_filter or op_name, csv_path)
            continue
        rows_post = rows_all[warmup_count:]
        if not rows_post:
            logging.warning(
                "forge raw result csv has no rows after warmup_count=%d for %s: %s",
                warmup_count, op_filter or op_name, csv_path,
            )
            continue
        for row in rows_post:
            duration = _maybe_float(row.get("_parsed_duration_us"))
            if duration is None:
                continue
            trial_records.append({"path": csv_path, "rows": [row], "sample": duration})

    # Cold-start detection: warn if first sample is >20% above median of rest
    if trial_records and warmup_count == 0 and len(trial_records) >= 4:
        first_sample = trial_records[0]["sample"]
        rest_samples = sorted(r["sample"] for r in trial_records[1:])
        rest_median = rest_samples[len(rest_samples) // 2]
        if rest_median > 0 and first_sample > rest_median * 1.2:
            logging.warning(
                "[%s] Possible cold-start: first sample %.1fus is %.0f%% above "
                "median of remaining samples %.1fus. Consider setting "
                "warmup_count >= 3 in forge config.",
                tag, first_sample,
                (first_sample / rest_median - 1) * 100, rest_median,
            )

    representative_rows = []
    if trial_records:
        filtered_records, removed_records = _clean_trial_records_iqr(trial_records)
        clean_samples = [record["sample"] for record in filtered_records]
        removed_samples = [record["sample"] for record in removed_records]
        representative = _select_representative_trial(filtered_records)
        if representative is not None:
            representative_rows = representative["rows"]
        pipeline = _normalize_pipeline(_extract_pipeline_from_rows(representative_rows))
        n_samples_raw = len(trial_records)
    else:
        clean_samples = []
        removed_samples = []
        pipeline = _normalize_pipeline(
            raw_side.get("pipeline_candidates") or raw_side.get("pipeline") or {}
        )
        n_samples_raw = 0

    if clean_samples:
        time_us = float(statistics.median(clean_samples))
        cv_pct = _compute_cv_pct(clean_samples)
    else:
        if fallback_time_us is not None and fallback_time_us > 0:
            time_us = float(fallback_time_us)
        elif allow_time_fallback:
            time_us = float(fallback_time_us) if fallback_time_us is not None else -1.0
        else:
            time_us = -1.0
        cv_pct = 0.0

    return {
        "tag": tag,
        "install_path": raw_side.get("install_path", fallback_install_path),
        "precision_passed": raw_side.get("precision_passed", default_precision_passed),
        "correctness_message": raw_side.get(
            "correctness_message",
            default_correctness_message,
        ),
        "time_us": time_us if time_us > 0 else -1,
        "ref_time_us": raw_side.get("ref_time_us", -1),
        "cv_pct": cv_pct,
        "pipeline": pipeline,
        "bottleneck": _infer_bottleneck(pipeline),
        "profiling_dir": raw_side.get("profiling_dir", ""),
        "profile_rows": representative_rows,
        "n_samples_raw": n_samples_raw,
        "n_outliers_removed": len(removed_samples),
        "warmup_count": warmup_count,
        "measurement_note": "kernel_time (msprof task_duration)",
    }


def _build_comparison(baseline_result: dict, evolved_result: dict, compilation_success: bool) -> dict:
    baseline_time = baseline_result.get("time_us", -1)
    evolved_time = evolved_result.get("time_us", -1)
    comparison = {
        "compilation_success": compilation_success,
        "precision_passed": (
            baseline_result.get("precision_passed", False)
            and evolved_result.get("precision_passed", False)
        ),
        "speedup": 0.0,
        "time_delta_us": 0.0,
        "bottleneck_change": (
            f"{baseline_result.get('bottleneck', 'unknown')} -> "
            f"{evolved_result.get('bottleneck', 'unknown')}"
        ),
        "cv_pct": evolved_result.get("cv_pct", 0.0),
    }
    if compilation_success and baseline_time > 0 and evolved_time > 0:
        comparison["speedup"] = baseline_time / evolved_time
        comparison["time_delta_us"] = evolved_time - baseline_time
    comparison["measurement_quality"] = _measurement_quality(comparison["cv_pct"])
    return comparison


def build_evaluation_result(
    compilation_success: bool,
    baseline_time_us: float,
    raw_result: dict | None = None,
    evolved_install_path: str = "",
    legacy_evolved_time_us: float | None = None,
) -> dict:
    """Build `evaluation_results.json` in the same core shape as `evaluate_ops.py`."""
    raw_result_present = raw_result is not None
    raw_result = raw_result or {}
    baseline_raw = raw_result.get("baseline")
    evolved_raw = raw_result.get("evolved")
    if not isinstance(baseline_raw, dict):
        baseline_raw = None
    if not isinstance(evolved_raw, dict):
        evolved_raw = None

    baseline_result = _build_side_result(
        baseline_raw,
        tag="baseline",
        op_name=raw_result.get("op_name", ""),
        fallback_time_us=baseline_time_us,
        default_precision_passed=True,
        default_correctness_message="PASS",
    )

    evolved_raw_present = raw_result_present
    evolved_has_valid_csv = evolved_raw and any(
        os.path.isfile(f) for f in evolved_raw.get("raw_op_summary_files", []) if f
    )
    evolved_fallback_time = (
        -1.0 if evolved_has_valid_csv else (legacy_evolved_time_us if legacy_evolved_time_us is not None else -1.0)
    )
    evolved_result = _build_side_result(
        evolved_raw,
        tag="evolved",
        op_name=raw_result.get("op_name", ""),
        fallback_time_us=evolved_fallback_time,
        fallback_install_path=evolved_install_path,
        default_precision_passed=bool(compilation_success),
        default_correctness_message="" if compilation_success else "forge build failed",
        allow_time_fallback=not evolved_raw_present,
    )

    if not compilation_success:
        evolved_result["precision_passed"] = False
        if not evolved_result["correctness_message"]:
            evolved_result["correctness_message"] = "forge build failed"
        evolved_result["time_us"] = -1
        evolved_result["cv_pct"] = 0.0
        evolved_result["pipeline"] = {}
        evolved_result["bottleneck"] = "unknown"

    result = {
        "op_name": raw_result.get("op_name", ""),
        "repo_type": raw_result.get("repo_type", ""),
        "soc": raw_result.get("soc", ""),
        "baseline": baseline_result,
        "evolved": evolved_result,
    }
    comparison = _build_comparison(baseline_result, evolved_result, compilation_success)
    result["comparison"] = comparison
    result["compilation_success"] = comparison["compilation_success"]
    result["precision_passed"] = comparison["precision_passed"]
    result["speedup"] = comparison["speedup"]
    result["cv_pct"] = comparison["cv_pct"]
    result["measurement_quality"] = comparison["measurement_quality"]
    result["eval_backend"] = "forge"
    return result


def _build_failed_precision_result(
    op_name: str,
    repo_type: str,
    soc: str,
    install_path: str,
    baseline_time_us: float,
    correctness_message: str,
) -> dict:
    baseline_result = {
        "tag": "baseline",
        "install_path": "",
        "precision_passed": True,
        "correctness_message": "PASS",
        "time_us": baseline_time_us if baseline_time_us > 0 else -1,
        "ref_time_us": -1,
        "cv_pct": 0.0,
        "pipeline": {},
        "bottleneck": "unknown",
        "profiling_dir": "",
        "profile_rows": [],
        "n_samples_raw": 0,
        "n_outliers_removed": 0,
    }
    evolved_result = {
        "tag": "evolved",
        "install_path": install_path,
        "precision_passed": False,
        "correctness_message": correctness_message,
        "time_us": -1,
        "ref_time_us": -1,
        "cv_pct": 0.0,
        "pipeline": {},
        "bottleneck": "unknown",
        "profiling_dir": "",
        "profile_rows": [],
        "n_samples_raw": 0,
        "n_outliers_removed": 0,
    }
    comparison = _build_comparison(baseline_result, evolved_result, True)
    result = {
        "op_name": op_name,
        "repo_type": repo_type,
        "soc": soc,
        "baseline": baseline_result,
        "evolved": evolved_result,
        "comparison": comparison,
        "compilation_success": comparison["compilation_success"],
        "precision_passed": comparison["precision_passed"],
        "speedup": comparison["speedup"],
        "cv_pct": comparison["cv_pct"],
        "measurement_quality": comparison["measurement_quality"],
        "eval_backend": "forge",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Forge adapter for ops-evo evaluation")
    parser.add_argument(
        "--forge-config",
        required=True,
        help="Forge config name (e.g. omni_ops_performance_pytest)",
    )
    parser.add_argument(
        "--forge-config-dir",
        required=True,
        help="Path to forge configs directory",
    )
    parser.add_argument(
        "--op-name",
        required=True,
        help="Operator name (passed to forge --op)",
    )
    parser.add_argument(
        "--install-path",
        required=True,
        help="Install path (passed to forge --ip)",
    )
    parser.add_argument(
        "--baseline-time-us",
        type=float,
        default=0.0,
        help="Baseline operator time in microseconds",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for evaluation_results.json",
    )
    parser.add_argument(
        "--forge-bin",
        default="forge",
        help="Path to forge executable (default: forge)",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["build", "accuracy", "perf", "both"],
        help="Run mode: build only, accuracy only, perf only, or both (default: both)",
    )
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--test-path", default="")
    parser.add_argument("--test-script", default="")
    parser.add_argument("--build-artifact", default="",
                        help="Forge build artifact path (e.g. output/CANN-omni_training_custom_ops--linux.aarch64.run)")
    parser.add_argument("--set-env-path", default="",
                        help="Path to set_env.bash for sourcing custom ops environment")
    parser.add_argument("--zsearch-side", default="evolved")
    parser.add_argument("--zsearch-precision-passed", default="true")
    parser.add_argument("--zsearch-correctness-message", default="PASS")
    parser.add_argument("--zsearch-build-success", default="true")
    parser.add_argument("--zsearch-repo-type", default="")
    parser.add_argument("--zsearch-soc", default="")
    parser.add_argument("--zsearch-task-type", default="performance")
    parser.add_argument(
        "--zsearch-op-filter",
        default="",
        help="Profiling CSV OP Type filter passed to forge as zsearch_op_filter",
    )
    parser.add_argument(
        "--zsearch-op-filter-map",
        default=str(DEFAULT_ZSEARCH_OP_FILTER_MAP),
        help="JSON map from op_name to profiling CSV OP Type filter",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    compilation_success = True
    raw_result = None
    raw_result_path = None
    legacy_evolved_time_us = None
    repo_type = args.zsearch_repo_type
    soc = args.zsearch_soc
    zsearch_op_filter = resolve_zsearch_op_filter(
        op_name=args.op_name,
        explicit_filter=args.zsearch_op_filter,
        filter_map_path=args.zsearch_op_filter_map,
    )

    with tempfile.TemporaryDirectory(prefix="forge_eval_") as tmpdir:
        forge_args_payload = {
            "repo_root": args.repo_root,
            "op_name": args.op_name,
            "install_path": args.install_path,
            "test_path": args.test_path,
            "test_script": args.test_script,
            "zsearch_side": args.zsearch_side,
            "zsearch_precision_passed": args.zsearch_precision_passed,
            "zsearch_correctness_message": args.zsearch_correctness_message,
            "zsearch_build_success": args.zsearch_build_success,
            "zsearch_repo_type": args.zsearch_repo_type,
            "zsearch_soc": args.zsearch_soc,
            "zsearch_task_type": args.zsearch_task_type,
            "zsearch_op_filter": zsearch_op_filter,
        }
        if args.build_artifact:
            forge_args_payload["build_artifact"] = args.build_artifact
        if args.set_env_path:
            forge_args_payload["set_env_path"] = args.set_env_path
        arg_file = write_forge_args_json(
            os.path.join(tmpdir, "forge_args.json"),
            forge_args_payload,
        )

        if args.mode in ("build", "both"):
            compilation_success = forge_build(
                args.forge_bin,
                args.forge_config_dir,
                args.forge_config,
                arg_file=arg_file,
            )
            if not compilation_success:
                logging.error("Forge build failed")

        accuracy_passed = args.zsearch_precision_passed.lower() == "true"
        correctness_message = args.zsearch_correctness_message
        if args.mode in ("accuracy", "both") and compilation_success:
            accuracy_passed, correctness_message = forge_accuracy_test(
                args.forge_bin,
                args.forge_config_dir,
                args.forge_config,
                arg_file=arg_file,
            )
            update_forge_args_json(
                arg_file,
                {
                    "zsearch_precision_passed": "true" if accuracy_passed else "false",
                    "zsearch_correctness_message": correctness_message,
                },
            )
            logging.info("Forge accuracy result: %s (%s)", "PASS" if accuracy_passed else "FAIL", correctness_message)

        if args.mode in ("perf", "both") and compilation_success and accuracy_passed:
            raw_result, raw_result_path, legacy_results = forge_perf_test(
                args.forge_bin,
                args.forge_config_dir,
                args.forge_config,
                args.op_name,
                args.install_path,
                arg_file=arg_file,
            )
            if raw_result_path:
                logging.info("Loaded forge raw result: %s", raw_result_path)
            if legacy_results:
                if args.op_name in legacy_results:
                    legacy_evolved_time_us = legacy_results[args.op_name]
                else:
                    legacy_evolved_time_us = next(iter(legacy_results.values()))
                logging.info("Legacy forge perf result: %.2f us", legacy_evolved_time_us)
            if raw_result is None and legacy_evolved_time_us is None:
                logging.error("Forge performance test returned no usable results")

    if not compilation_success:
        result = build_evaluation_result(
            compilation_success=compilation_success,
            baseline_time_us=args.baseline_time_us,
            raw_result=raw_result,
            evolved_install_path=args.install_path,
            legacy_evolved_time_us=legacy_evolved_time_us,
        )
    elif args.mode in ("accuracy", "both") and not accuracy_passed:
        result = _build_failed_precision_result(
            op_name=args.op_name,
            repo_type=repo_type,
            soc=soc,
            install_path=args.install_path,
            baseline_time_us=args.baseline_time_us,
            correctness_message=correctness_message,
        )
    else:
        result = build_evaluation_result(
            compilation_success=compilation_success,
            baseline_time_us=args.baseline_time_us,
            raw_result=raw_result,
            evolved_install_path=args.install_path,
            legacy_evolved_time_us=legacy_evolved_time_us,
        )

    if not result.get("op_name"):
        result["op_name"] = args.op_name
    if not result.get("repo_type"):
        result["repo_type"] = repo_type
    if not result.get("soc"):
        result["soc"] = soc

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logging.info("Evaluation result saved to: %s", args.output)

    print(f"\n{'=' * 50}")
    print("Forge Evaluation Summary")
    print(f"{'=' * 50}")
    print(f"  Build:    {'OK' if compilation_success else 'FAILED'}")
    print(f"  Time:     {result['evolved']['time_us']:.2f} us")
    print(f"  Baseline: {result['baseline']['time_us']:.2f} us")
    print(f"  Speedup:  {result['speedup']:.3f}x")
    if raw_result_path:
        print(f"  Raw JSON: {raw_result_path}")
    print(f"{'=' * 50}")

    if not compilation_success or ((args.mode in ("accuracy", "both")) and not accuracy_passed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
