#!/usr/bin/env python3
"""Evolution report generator for ops-evo pipeline.

Parses output directory files and renders a standardized HTML report.
"""

import argparse
import difflib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any


def parse_output_dir_name(output_dir: str) -> tuple[str, str]:
    """Extract op_name and timestamp from output directory name."""
    dirname = os.path.basename(output_dir.rstrip("/"))
    m = re.match(r"(.+?)_ops-evo_(\d{8}_\d{6})$", dirname)
    if m:
        return m.group(1), m.group(2)
    m2 = re.match(r"(.+?)_(\d{8}_\d{6})$", dirname)
    if m2:
        return m2.group(1), m2.group(2)
    return dirname, datetime.now().strftime("%Y%m%d_%H%M%S")


def get_model_from_config() -> str | None:
    """Read model name from Claude Code settings.json in CLAUDE_CONFIG_DIR.

    Priority:
    1. $CLAUDE_CONFIG_DIR/settings.json -> env.ANTHROPIC_MODEL
    2. ~/.claude/settings.json -> env.ANTHROPIC_MODEL
    """
    config_paths = []
    claude_config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if claude_config_dir:
        config_paths.append(os.path.join(claude_config_dir, "settings.json"))
    home = os.path.expanduser("~")
    config_paths.append(os.path.join(home, ".claude", "settings.json"))

    for path in config_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            model = settings.get("env", {}).get("ANTHROPIC_MODEL")
            if model:
                return model
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return None


def load_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def normalize_eval_json(eval_data: dict | None) -> dict | None:
    """Normalize evaluation_results.json to the full nested format.

    Handles two input formats:
    1. Full format (baseline eval output): {baseline: {...}, evolved: {...}, comparison: {...}}
    2. Flat format (single-variant eval output): {tag, time_us, precision_passed, ...}

    Flat format is produced when ops-partial subagents save only the evolved
    result from run_single_version(). We normalize it to full format so all
    downstream consumers work uniformly.
    """
    if not eval_data:
        return None

    # Already full format
    if "baseline" in eval_data and "evolved" in eval_data and "comparison" in eval_data:
        return eval_data

    # Flat format: wrap in full structure
    time_us = eval_data.get("time_us", -1)
    precision_passed = eval_data.get("precision_passed", False)
    # If precision passed and we have a positive time, compilation implicitly succeeded
    compilation_success = eval_data.get("compilation_success")
    if compilation_success is None:
        compilation_success = precision_passed and time_us is not None and time_us > 0

    normalized = {
        "op_name": eval_data.get("op_name", ""),
        "baseline": {},
        "evolved": {
            "tag": eval_data.get("tag", "evolved"),
            "install_path": eval_data.get("install_path", ""),
            "precision_passed": precision_passed,
            "correctness_message": eval_data.get("correctness_message", ""),
            "time_us": time_us,
            "pipeline": eval_data.get("pipeline", {}),
            "bottleneck": eval_data.get("bottleneck", "unknown"),
            "cv_pct": eval_data.get("cv_pct", 0.0),
        },
        "comparison": {
            "compilation_success": compilation_success,
            "precision_passed": precision_passed,
            "speedup": 0.0,
            "time_delta_us": 0.0,
            "cv_pct": eval_data.get("cv_pct", 0.0),
            "measurement_quality": "unknown",
        },
    }
    return normalized


def load_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def parse_session_stats(session_dir: str, session_jsonl: str | None = None) -> dict[str, Any]:
    """Parse session records to extract timing, token statistics, and model info.

    Args:
        session_dir: Path to .claude/projects directory containing session files
        session_jsonl: Path to the specific JSONL file for this session (optional)

    Returns:
        Dictionary with timing, token statistics, and model information
    """
    stats = {
        "timing": {
            "main_session": {"start": None, "end": None, "duration_minutes": 0},
            "evo_agent": {"start": None, "end": None, "duration_minutes": 0},
            "rounds": {},
            "total_duration_minutes": 0,
        },
        "tokens": {
            "main_session": {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0},
            "evo_agent": {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0},
            "aside_agents": [],
            "total": {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0},
        },
        "model": None,  # Extracted AI model name (e.g., "kimi-for-coding")
    }

    def parse_jsonl(filepath: str) -> list[dict]:
        records = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except FileNotFoundError:
            pass
        return records

    def extract_timestamps(records: list[dict]) -> tuple[float | None, float | None]:
        timestamps = []
        for r in records:
            ts = r.get("timestamp")
            if ts:
                try:
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        timestamps.append(dt.timestamp() * 1000)
                    elif isinstance(ts, (int, float)):
                        timestamps.append(ts)
                except (ValueError, TypeError):
                    pass
        if timestamps:
            return min(timestamps), max(timestamps)
        return None, None

    def count_tokens(records: list[dict]) -> dict[str, int]:
        tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
        for r in records:
            if r.get("type") == "assistant":
                msg = r.get("message", {})
                usage = msg.get("usage", {})
                tokens["input"] += usage.get("input_tokens", 0)
                tokens["output"] += usage.get("output_tokens", 0)
                tokens["cache_read"] += usage.get("cache_read_input_tokens", 0)
                tokens["cache_creation"] += usage.get("cache_creation_input_tokens", 0)
        return tokens

    def extract_model(records: list[dict]) -> str | None:
        """Extract model name from the first assistant message with a model field."""
        for r in records:
            if r.get("type") == "assistant":
                msg = r.get("message", {})
                model = msg.get("model")
                if model:
                    return model
        return None

    def find_first_user_message_ts(records: list[dict]) -> float | None:
        """Find timestamp of the first genuine user input (not tool_result wrapper)."""
        for r in records:
            if r.get("type") == "user":
                msg = r.get("message", {})
                content = msg.get("content", "")
                # Genuine user input is a string; tool_result wrappers are lists
                if isinstance(content, str) and content.strip():
                    ts = r.get("timestamp")
                    if ts:
                        try:
                            if isinstance(ts, str):
                                return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000
                            elif isinstance(ts, (int, float)):
                                return ts
                        except (ValueError, TypeError):
                            pass
        return None

    def find_last_task_completion_ts(records: list[dict]) -> float | None:
        """Find timestamp of the last task-completion / agent-done notification.

        Returns the last task-notification timestamp (not the first), because
        earlier ones may be stale notifications from a resumed session.
        """
        last_ts = None
        for r in records:
            if r.get("type") == "user":
                msg = r.get("message", {})
                content = str(msg.get("content", ""))
                if "task-notification" in content.lower() or "task_notification" in content.lower():
                    ts = r.get("timestamp")
                    if ts:
                        try:
                            if isinstance(ts, str):
                                last_ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000
                            elif isinstance(ts, (int, float)):
                                last_ts = ts
                        except (ValueError, TypeError):
                            pass
        return last_ts

    def calc_active_duration_ms(timestamps: list[float], gap_threshold_ms: float = 600_000) -> float:
        """Calculate active duration by summing intervals, completely excluding large gaps.

        Large gaps (> gap_threshold_ms) represent idle/wait time and are excluded
        entirely — no buffer is added. This prevents inflated timing when the session
        was left open or waiting for external processes.
        """
        if len(timestamps) < 2:
            return 0.0
        sorted_ts = sorted(timestamps)
        total = 0.0
        for i in range(1, len(sorted_ts)):
            gap = sorted_ts[i] - sorted_ts[i - 1]
            if gap <= gap_threshold_ms:
                total += gap
            # else: gap is idle time, exclude entirely
        return total

    # Find session files
    session_path = Path(session_dir)
    session_files = list(session_path.glob("*.jsonl"))

    # Determine subagent directory location
    subagent_dir = None
    if session_jsonl:
        jsonl_path = Path(session_jsonl)
        subagent_dir = jsonl_path.parent / jsonl_path.stem / "subagents"

    # Parse main session
    main_records: list[dict] = []
    if session_jsonl:
        main_records = parse_jsonl(session_jsonl)
    else:
        for sf in session_files:
            main_records = parse_jsonl(str(sf))
            if main_records:
                break

    if main_records:
        # Main session start = first genuine user message (not session-start hook)
        first_user_ts = find_first_user_message_ts(main_records)
        # Main session end = last task completion, but only if it is close to the
        # actual session end.  Stale task-notifications from resumed sessions can
        # appear early in the transcript and must not truncate the session.
        last_task_done_ts = find_last_task_completion_ts(main_records)
        raw_start, raw_end = extract_timestamps(main_records)

        start = first_user_ts if first_user_ts is not None else raw_start
        # Use task-completion time only when it is within 30 min of the real end.
        # Otherwise the session continued after the agent finished (e.g. report
        # regeneration) and we should measure the full span.
        if last_task_done_ts and raw_end and (raw_end - last_task_done_ts) < 1_800_000:
            end = last_task_done_ts
        else:
            end = raw_end

        if start and end:
            stats["timing"]["main_session"]["start"] = start
            stats["timing"]["main_session"]["end"] = end
            # Calculate active duration: filter out large idle gaps (>10 min)
            all_ts = []
            for r in main_records:
                ts = r.get("timestamp")
                if ts:
                    try:
                        if isinstance(ts, str):
                            all_ts.append(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)
                        elif isinstance(ts, (int, float)):
                            all_ts.append(ts)
                    except (ValueError, TypeError):
                        pass
            # Only count timestamps within [start, end]
            filtered_ts = [t for t in all_ts if start <= t <= end]
            active_ms = calc_active_duration_ms(filtered_ts, gap_threshold_ms=600_000)
            range_ms = end - start
            # When the session was mostly waiting for a background agent
            # (active time < 15% of wall-clock span), the wall-clock span
            # is more representative of the user-facing duration.
            # Otherwise use active duration to exclude idle gaps.
            if range_ms > 0 and active_ms / range_ms < 0.15 and range_ms > 300_000:
                stats["timing"]["main_session"]["duration_minutes"] = range_ms / 1000 / 60
            else:
                # Use active duration directly — no artificial floor.
                # This excludes post-task wait time and long idle gaps.
                stats["timing"]["main_session"]["duration_minutes"] = active_ms / 1000 / 60
        else:
            if raw_start and raw_end:
                stats["timing"]["main_session"]["start"] = raw_start
                stats["timing"]["main_session"]["end"] = raw_end
                stats["timing"]["main_session"]["duration_minutes"] = (raw_end - raw_start) / 1000 / 60

        tokens = count_tokens(main_records)
        stats["tokens"]["main_session"] = tokens
        stats["model"] = extract_model(main_records)

    # Parse subagents: identify evo agent by highest token usage (not by name)
    evo_agent_end_ts: float | None = None
    if subagent_dir and subagent_dir.is_dir():
        agent_files = list(subagent_dir.glob("*.jsonl"))
        agent_stats = []
        for agent_file in agent_files:
            records = parse_jsonl(str(agent_file))
            agent_name = agent_file.stem
            tokens = count_tokens(records)
            total_tokens = sum(tokens.values())
            start, end = extract_timestamps(records)
            agent_stats.append({
                "name": agent_name,
                "records": records,
                "tokens": tokens,
                "total_tokens": total_tokens,
                "start": start,
                "end": end,
            })

        # Sort by total token usage descending
        agent_stats.sort(key=lambda x: x["total_tokens"], reverse=True)

        # The agent with highest token usage is the evo agent (main orchestrator)
        if agent_stats:
            evo = agent_stats[0]
            if evo["total_tokens"] > 1000:  # Must have meaningful token usage
                stats["timing"]["evo_agent"]["start"] = evo["start"]
                stats["timing"]["evo_agent"]["end"] = evo["end"]
                evo_agent_end_ts = evo["end"]
                if evo["start"] and evo["end"]:
                    stats["timing"]["evo_agent"]["duration_minutes"] = (
                        evo["end"] - evo["start"]
                    ) / 1000 / 60
                stats["tokens"]["evo_agent"] = evo["tokens"]
                # Also extract model from evo agent if main session didn't have it
                if not stats["model"]:
                    stats["model"] = extract_model(evo["records"])

            # Remaining agents are aside agents (if they have token usage)
            for aside in agent_stats[1:]:
                if aside["total_tokens"] > 1000:
                    stats["tokens"]["aside_agents"].append(
                        {"name": aside["name"], **aside["tokens"]}
                    )

    # If main session end was not found via task notification but we have evo agent end,
    # use evo agent end as a fallback for main session end (the main session waited for it)
    if stats["timing"]["main_session"]["end"] is None and evo_agent_end_ts:
        stats["timing"]["main_session"]["end"] = evo_agent_end_ts
        if stats["timing"]["main_session"]["start"]:
            start = stats["timing"]["main_session"]["start"]
            stats["timing"]["main_session"]["duration_minutes"] = (evo_agent_end_ts - start) / 1000 / 60

    # Calculate totals
    total_tokens = stats["tokens"]["total"]
    for component in ["main_session", "evo_agent"]:
        comp_tokens = stats["tokens"][component]
        for key in ["input", "output", "cache_read", "cache_creation"]:
            total_tokens[key] += comp_tokens[key]
    for aside in stats["tokens"]["aside_agents"]:
        for key in ["input", "output", "cache_read", "cache_creation"]:
            total_tokens[key] += aside[key]

    # Calculate total duration: prefer evo_agent duration as it represents the
    # actual optimization work. Main session may include post-task idle time.
    # If evo_agent ran, total = evo_agent duration (main session was just waiting).
    # If no evo_agent, total = main session duration.
    evo_dur = stats["timing"]["evo_agent"]["duration_minutes"]
    main_dur = stats["timing"]["main_session"]["duration_minutes"]
    if evo_dur > 0:
        stats["timing"]["total_duration_minutes"] = evo_dur
    elif main_dur > 0:
        stats["timing"]["total_duration_minutes"] = main_dur
    else:
        stats["timing"]["total_duration_minutes"] = 0

    return stats


def _get_dir_birthtime(path: Path) -> float | None:
    """Get directory creation (birth) time.

    On Linux, os.stat().st_ctime is the inode change time (updated when files
    are written into the directory), NOT the creation time.  We use `stat -c '%W'`
    to obtain the true birth time.  Falls back to st_ctime if unavailable.
    Returns None only if the path does not exist or is inaccessible.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["stat", "-c", "%W", str(path)],
            capture_output=True, text=True, check=True,
        )
        ts = float(result.stdout.strip())
        if ts > 0:
            return ts
    except Exception:
        pass
    # Fallback: st_ctime is wrong on Linux for this purpose, but better than nothing
    try:
        return os.stat(path).st_ctime
    except OSError:
        return None


def get_session_time_range(jsonl_path: Path) -> tuple[float | None, float | None]:
    """Return (start_ms, end_ms) timestamps from a session JSONL file."""
    timestamps: list[float] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = rec.get("timestamp")
                if not ts:
                    continue
                try:
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        timestamps.append(dt.timestamp() * 1000)
                    elif isinstance(ts, (int, float)):
                        timestamps.append(ts)
                except (ValueError, TypeError):
                    pass
    except FileNotFoundError:
        pass
    if timestamps:
        return min(timestamps), max(timestamps)
    return None, None


def parse_round_timing(output_dir: str) -> dict[str, Any]:
    """Parse round directories to extract timing information.

    Returns:
        Dictionary with round timing details
    """
    round_timing = {
        "baseline": {"start": None, "end": None, "duration_minutes": 0},
        "rounds": {},
        "total_evolution_minutes": 0,
    }

    def get_dir_times(path: Path) -> tuple[float, float]:
        birth = _get_dir_birthtime(path)
        mtime = os.stat(path).st_mtime
        return birth if birth is not None else mtime, mtime

    # Baseline timing
    baseline_dir = Path(output_dir) / "baseline"
    if baseline_dir.exists():
        create_time, modify_time = get_dir_times(baseline_dir)
        round_timing["baseline"]["start"] = create_time
        round_timing["baseline"]["end"] = modify_time
        round_timing["baseline"]["duration_minutes"] = max(0, (modify_time - create_time)) / 60

    # Round timing - use variant directory birthtimes for accurate round starts
    round_dirs = sorted(
        [d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith("round_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    for rd in round_dirs:
        round_num = int(rd.name.split("_")[1])

        variants = []
        parallel_dirs = sorted(
            [p for p in rd.iterdir() if p.is_dir() and p.name.startswith("parallel_")],
            key=lambda p: int(p.name.split("_")[1]),
        )
        for pd in parallel_dirs:
            par_num = int(pd.name.split("_")[1])
            # Use evaluation_results.json timestamp for end time (when eval finished)
            eval_json = pd / "evaluation_results.json"
            if eval_json.exists():
                var_end = eval_json.stat().st_mtime
            else:
                _, var_end = get_dir_times(pd)
            # Try to get a more accurate build-end time from the evolved directory
            evolved_dir = pd / "evolved"
            if evolved_dir.exists():
                build_end = evolved_dir.stat().st_mtime
            else:
                build_end = var_end
            variants.append({
                "parallel": par_num,
                "build_end": build_end,
                "end": var_end,
                "duration_minutes": max(0, (var_end - build_end)) / 60,
            })

        # Calculate round timing
        if variants:
            round_end = max(v["end"] for v in variants)
            # Use earliest variant dir BIRTH time as round start.
            # Birth time reflects when the directory was actually created (subagent launch),
            # unlike ctime which updates whenever files are written into the directory.
            variant_starts = []
            for v in variants:
                pd = rd / f"parallel_{v['parallel']}"
                if pd.exists():
                    birth = _get_dir_birthtime(pd)
                    if birth is not None:
                        variant_starts.append(birth)
            if variant_starts:
                round_start = min(variant_starts)
            else:
                round_start = rd.stat().st_ctime
            round_timing["rounds"][round_num] = {
                "start": round_start,
                "end": round_end,
                "duration_minutes": max(0, (round_end - round_start)) / 60,
                "variants": variants,
            }
        else:
            # Fallback to directory times
            round_create, round_modify = get_dir_times(rd)
            round_timing["rounds"][round_num] = {
                "start": round_create,
                "end": round_modify,
                "duration_minutes": max(0, (round_modify - round_create)) / 60,
                "variants": [],
            }

    # Fix pre-created round directories:
    # If a round's start time is earlier than or too close to the previous round's end,
    # the directory was pre-created. Shift its start to after the previous round ends.
    sorted_round_nums = sorted(round_timing["rounds"].keys())
    for i in range(1, len(sorted_round_nums)):
        prev_rn = sorted_round_nums[i - 1]
        curr_rn = sorted_round_nums[i]
        prev_end = round_timing["rounds"][prev_rn]["end"]
        curr_start = round_timing["rounds"][curr_rn]["start"]
        # If current round appears to start before previous round ended,
        # or if multiple rounds were created within a very short window (< 2 min),
        # treat as pre-created and shift start time.
        if curr_start is not None and prev_end is not None and curr_start < prev_end + 120:
            round_timing["rounds"][curr_rn]["start"] = prev_end + 60
            # Recalculate duration, ensure non-negative
            new_duration = max(0, round_timing["rounds"][curr_rn]["end"] - round_timing["rounds"][curr_rn]["start"])
            round_timing["rounds"][curr_rn]["duration_minutes"] = new_duration / 60

    # Calculate total evolution time
    if round_timing["rounds"]:
        first_round_start = min(r["start"] for r in round_timing["rounds"].values())
        last_round_end = max(r["end"] for r in round_timing["rounds"].values())
        round_timing["total_evolution_minutes"] = max(0, (last_round_end - first_round_start)) / 60

    return round_timing


def build_resource_stats_section(session_stats: dict, round_timing: dict) -> str:
    """Build HTML section for resource statistics (timing and tokens)."""
    timing = session_stats.get("timing", {})
    tokens = session_stats.get("tokens", {})

    # Format duration
    def format_duration(minutes: float) -> str:
        if minutes < 60:
            return f"{minutes:.1f} 分钟"
        total_mins = round(minutes)
        hours = total_mins // 60
        mins = total_mins % 60
        if mins == 0:
            return f"{hours} 小时"
        return f"{hours} 小时 {mins} 分钟"

    # Format tokens
    def format_tokens(count: int) -> str:
        if count >= 1_000_000:
            return f"{count / 1_000_000:.2f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.1f}K"
        return str(count)

    # Build timing table
    timing_rows = []

    # Helper to format time range (handles cross-day sessions)
    def format_time_range(start_ms: float | None, end_ms: float | None) -> str:
        if not start_ms or not end_ms:
            return "- ~ -"
        start_dt = datetime.fromtimestamp(start_ms / 1000)
        end_dt = datetime.fromtimestamp(end_ms / 1000)
        if start_dt.date() == end_dt.date():
            # Same day: HH:MM:SS ~ HH:MM:SS
            return f"{start_dt.strftime('%H:%M:%S')} ~ {end_dt.strftime('%H:%M:%S')}"
        else:
            # Different days: MM-DD HH:MM ~ MM-DD HH:MM
            return f"{start_dt.strftime('%m-%d %H:%M')} ~ {end_dt.strftime('%m-%d %H:%M')}"

    # Main session
    main_timing = timing.get("main_session", {})
    if main_timing.get("duration_minutes", 0) > 0:
        time_range = format_time_range(main_timing.get("start"), main_timing.get("end"))
        timing_rows.append(
            f"<tr><td>主会话</td><td>{time_range}</td>"
            f"<td>{format_duration(main_timing['duration_minutes'])}</td><td>串行</td></tr>"
        )

    # Evo agent
    evo_timing = timing.get("evo_agent", {})
    if evo_timing.get("duration_minutes", 0) > 0:
        time_range = format_time_range(evo_timing.get("start"), evo_timing.get("end"))
        timing_rows.append(
            f"<tr><td>ops-evo 代理</td><td>{time_range}</td>"
            f"<td>{format_duration(evo_timing['duration_minutes'])}</td><td>主控</td></tr>"
        )

    # Rounds
    for round_num, round_data in sorted(round_timing.get("rounds", {}).items()):
        start_str = datetime.fromtimestamp(round_data["start"]).strftime("%H:%M:%S") if round_data.get("start") else "-"
        end_str = datetime.fromtimestamp(round_data["end"]).strftime("%H:%M:%S") if round_data.get("end") else "-"
        variant_count = len(round_data.get("variants", []))
        timing_rows.append(
            f"<tr><td>轮次 {round_num}</td><td>{start_str} ~ {end_str}</td>"
            f"<td>{format_duration(round_data['duration_minutes'])}</td>"
            f"<td>{variant_count} 变体并行</td></tr>"
        )

    timing_table = "\n".join(timing_rows) if timing_rows else "<tr><td colspan='4'>无时间数据</td></tr>"

    # Build token table
    total_tokens = tokens.get("total", {})
    token_rows = []

    def _token_sum(t: dict) -> int:
        return t.get("input", 0) + t.get("output", 0) + t.get("cache_read", 0) + t.get("cache_creation", 0)

    # Main session
    main_tokens = tokens.get("main_session", {})
    if main_tokens.get("input", 0) > 0 or main_tokens.get("output", 0) > 0:
        token_rows.append(
            f"<tr><td>主会话</td>"
            f"<td>{format_tokens(main_tokens.get('input', 0))}</td>"
            f"<td>{format_tokens(main_tokens.get('output', 0))}</td>"
            f"<td>{format_tokens(main_tokens.get('cache_read', 0))}</td>"
            f"<td>{format_tokens(main_tokens.get('cache_creation', 0))}</td>"
            f"<td>{format_tokens(_token_sum(main_tokens))}</td></tr>"
        )

    # Evo agent
    evo_tokens = tokens.get("evo_agent", {})
    if evo_tokens.get("input", 0) > 0 or evo_tokens.get("output", 0) > 0:
        token_rows.append(
            f"<tr><td>ops-evo 代理</td>"
            f"<td>{format_tokens(evo_tokens.get('input', 0))}</td>"
            f"<td>{format_tokens(evo_tokens.get('output', 0))}</td>"
            f"<td>{format_tokens(evo_tokens.get('cache_read', 0))}</td>"
            f"<td>{format_tokens(evo_tokens.get('cache_creation', 0))}</td>"
            f"<td>{format_tokens(_token_sum(evo_tokens))}</td></tr>"
        )

    # Aside agents
    for aside in tokens.get("aside_agents", []):
        if aside.get("input", 0) > 0 or aside.get("output", 0) > 0:
            token_rows.append(
                f"<tr><td>{aside['name'][:20]}...</td>"
                f"<td>{format_tokens(aside.get('input', 0))}</td>"
                f"<td>{format_tokens(aside.get('output', 0))}</td>"
                f"<td>{format_tokens(aside.get('cache_read', 0))}</td>"
                f"<td>{format_tokens(aside.get('cache_creation', 0))}</td>"
                f"<td>{format_tokens(_token_sum(aside))}</td></tr>"
            )

    token_table = "\n".join(token_rows) if token_rows else "<tr><td colspan='6'>无 词元 数据</td></tr>"

    # Calculate percentages
    total_all = (
        total_tokens.get("input", 0)
        + total_tokens.get("output", 0)
        + total_tokens.get("cache_read", 0)
        + total_tokens.get("cache_creation", 0)
    )
    cache_pct = (total_tokens.get("cache_read", 0) / total_all * 100) if total_all > 0 else 0
    cache_creation_pct = (total_tokens.get("cache_creation", 0) / total_all * 100) if total_all > 0 else 0
    input_pct = (total_tokens.get("input", 0) / total_all * 100) if total_all > 0 else 0
    output_pct = (total_tokens.get("output", 0) / total_all * 100) if total_all > 0 else 0

    html = f"""
<h2>资源消耗统计</h2>
<div class="two-col">
<div class="card">
<h3 style="margin-top:0">耗时统计</h3>
<table>
<tr><th>阶段</th><th>时间范围</th><th>持续时间</th><th>执行方式</th></tr>
{timing_table}
<tr style="font-weight:600;border-top:2px solid var(--border);">
<td>总计</td><td>-</td>
<td>{format_duration(timing.get('total_duration_minutes', 0))}</td><td>-</td>
</tr>
</table>
</div>
<div class="card">
<h3 style="margin-top:0">词元用量统计</h3>
<table>
<tr><th>组件</th><th>Input</th><th>Output</th><th>Cache Read</th><th>Cache Creation</th><th>总计</th></tr>
{token_table}
<tr style="font-weight:600;border-top:2px solid var(--border);">
<td>总计</td>
<td>{format_tokens(total_tokens.get('input', 0))}</td>
<td>{format_tokens(total_tokens.get('output', 0))}</td>
<td>{format_tokens(total_tokens.get('cache_read', 0))}</td>
<td>{format_tokens(total_tokens.get('cache_creation', 0))}</td>
<td>{format_tokens(total_all)}</td>
</tr>
</table>
<div style="margin-top:1rem;font-size:0.85rem;color:var(--text-muted);">
<strong>词元分布:</strong> Cache Read {cache_pct:.1f}% | Cache Creation {cache_creation_pct:.1f}% | Input {input_pct:.1f}% | Output {output_pct:.1f}%
</div>
</div>
</div>
"""
    return html


def _parse_test_cases_csv(csv_path: str) -> dict:
    """Parse the first data row of test_cases.csv into a dict of param->value."""
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) < 2:
            return {}
        headers = [h.strip() for h in lines[0].split(",")]
        values = [v.strip() for v in lines[1].split(",")]
        return dict(zip(headers, values))
    except Exception:
        return {}


def collect_rounds(output_dir: str) -> list[dict]:
    """Collect all round/parallel evaluation results."""
    results = []
    round_dirs = sorted(
        [d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith("round_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    for rd in round_dirs:
        round_num = int(rd.name.split("_")[1])
        parallel_dirs = sorted(
            [p for p in rd.iterdir() if p.is_dir() and p.name.startswith("parallel_")],
            key=lambda p: int(p.name.split("_")[1]),
        )
        for pd in parallel_dirs:
            par_num = int(pd.name.split("_")[1])
            eval_json = load_json(str(pd / "evaluation_results.json"))
            if not eval_json:
                eval_json = load_json(str(pd / "eval.json"))
            eval_json = normalize_eval_json(eval_json)
            impl_note = load_text(str(pd / "implementation_note.txt"))
            modified_files_dir = pd / "modified_files"
            modified_files = []
            if modified_files_dir.is_dir():
                for root, _, files in os.walk(modified_files_dir):
                    for fn in files:
                        rel = os.path.relpath(os.path.join(root, fn), modified_files_dir)
                        if not _should_include_modified_file(rel):
                            continue
                        modified_files.append(rel)
            results.append({
                "round": round_num,
                "parallel": par_num,
                "eval": eval_json,
                "impl_note": impl_note,
                "modified_files": modified_files,
                "modified_files_dir": str(modified_files_dir),
                "path": str(pd),
            })
    return results


def enrich_rounds_from_world_model(rounds: list[dict], wm: dict | None, baseline_time: float) -> None:
    """Supplement rounds missing evaluation_results.json with world model node data.

    When eval JSON is absent but the world model has a node with a matching
    solution_ref and a positive score, synthesize eval data so the variant
    is not silently dropped.
    """
    if not wm or baseline_time <= 0:
        return
    nodes = wm.get("decision_tree", {}).get("nodes", {})
    ref_map: dict[str, dict] = {}
    for node in nodes.values():
        ref = node.get("solution_ref")
        if ref:
            ref_map[ref] = node

    for r in rounds:
        if r.get("eval"):
            continue
        ref = f"round_{r['round']}/parallel_{r['parallel']}"
        node = ref_map.get(ref)
        if not node:
            continue
        score = node.get("score")
        if not score or score <= 0:
            continue
        # Synthesize evaluation data from world model score
        evolved_time = baseline_time / score
        r["eval"] = {
            "compilation_success": True,
            "precision_passed": True,
            "evolved": {"time_us": round(evolved_time, 2)},
            "_source": "world_model",
        }


def find_best_variant(rounds: list[dict], baseline_time: float) -> dict | None:
    """Find the variant with the highest speedup."""
    best = None
    for r in rounds:
        ev = r.get("eval")
        if not ev:
            continue
        # Check compilation_success and precision_passed at top level or in comparison
        compilation_ok = ev.get("compilation_success")
        precision_ok = ev.get("precision_passed")
        if compilation_ok is None and "comparison" in ev:
            compilation_ok = ev["comparison"].get("compilation_success")
            precision_ok = ev["comparison"].get("precision_passed")
        if not compilation_ok or not precision_ok:
            continue
        evolved_time = ev.get("evolved", {}).get("time_us")
        if evolved_time and evolved_time > 0:
            speedup = baseline_time / evolved_time
            if best is None or speedup > best["speedup"]:
                best = {**r, "speedup": speedup, "time_us": evolved_time}
    return best


def get_node_for_variant(wm: dict, round_num: int, par_num: int) -> dict | None:
    """Find the world model node matching a round/parallel variant."""
    if not wm:
        return None
    nodes = wm.get("decision_tree", {}).get("nodes", {})
    ref = f"round_{round_num}/parallel_{par_num}"
    for nid, node in nodes.items():
        if node.get("solution_ref") == ref:
            return node
    return None


def find_best_path(wm: dict, best_variant: dict | None = None, baseline_time: float = 0) -> tuple[list[str], float]:
    """Trace the best path from root to the best leaf.

    Priority:
    1. Path to the node matching the actual best variant (if found in WM)
    2. Path to the highest-score node in the world model

    Returns (path, endpoint_score)
    """
    if not wm:
        return [], 0.0

    nodes = wm.get("decision_tree", {}).get("nodes", {})
    root_children = wm.get("decision_tree", {}).get("root", {}).get("children", [])
    if not root_children and "root" in nodes:
        root_children = nodes["root"].get("children", [])
    # Fallback: infer root children from parent_id="root"
    if not root_children:
        root_children = [nid for nid, n in nodes.items() if n.get("parent_id") == "root"]
    # Also include any node with parent_id="root" that is not already in root_children
    for nid, n in nodes.items():
        if n.get("parent_id") == "root" and nid not in root_children:
            root_children.append(nid)

    # Build parent lookup for traceback
    parent_of = {}
    for nid, node in nodes.items():
        for child_id in node.get("children", []):
            parent_of[child_id] = nid

    # Strategy 1: Try to find path to the node matching actual best variant
    if best_variant:
        target_ref = f"round_{best_variant['round']}/parallel_{best_variant['parallel']}"
        target_node = None
        for nid, node in nodes.items():
            if node.get("solution_ref") == target_ref:
                target_node = nid
                break

        if target_node:
            # Trace back from target_node to root
            path = [target_node]
            current = target_node
            while current in parent_of:
                current = parent_of[current]
                path.insert(0, current)
            endpoint_score = nodes.get(target_node, {}).get("score", 0)
            if endpoint_score is None:
                endpoint_score = 0.0
            return ["root"] + path, endpoint_score

    # Strategy 2: Find the node with the highest score
    best_node_id = None
    best_node_score = -1.0
    for nid, node in nodes.items():
        score = node.get("score")
        if score is not None and score > best_node_score:
            best_node_score = score
            best_node_id = nid

    if best_node_id and best_node_score > 0:
        # Trace back via parent_of
        path = [best_node_id]
        current = best_node_id
        while current in parent_of:
            current = parent_of[current]
            path.insert(0, current)
        return ["root"] + path, best_node_score

    return ["root"], 1.0


def _extract_shape_groups(call_spec: dict) -> list[dict]:
    """Normalize call_spec into a list of shape groups.

    Supports two formats:
    - Legacy single-shape: top-level ``inputs`` / ``scalar_args`` / ``tensor_kwargs``.
    - Multi-shape: ``target_shapes`` / ``generalization_shapes`` arrays, each item
      carrying its own ``inputs`` / ``scalar_args`` / ``tensor_kwargs``.

    Each returned group has: ``{role, name, inputs, scalar_args, tensor_kwargs}``.
    """
    groups: list[dict] = []
    targets = call_spec.get("target_shapes") or []
    gens = call_spec.get("generalization_shapes") or []
    if targets or gens:
        for i, sh in enumerate(targets):
            groups.append({
                "role": "target",
                "name": sh.get("name") or f"T{i+1}",
                "inputs": sh.get("inputs", []),
                "scalar_args": sh.get("scalar_args", {}),
                "tensor_kwargs": sh.get("tensor_kwargs", {}),
            })
        for i, sh in enumerate(gens):
            groups.append({
                "role": "generalization",
                "name": sh.get("name") or f"G{i+1}",
                "inputs": sh.get("inputs", []),
                "scalar_args": sh.get("scalar_args", {}),
                "tensor_kwargs": sh.get("tensor_kwargs", {}),
            })
        return groups

    # Legacy single-shape format
    if call_spec.get("inputs") or call_spec.get("scalar_args") or call_spec.get("tensor_kwargs"):
        groups.append({
            "role": "single",
            "name": "",
            "inputs": call_spec.get("inputs", []),
            "scalar_args": call_spec.get("scalar_args", {}),
            "tensor_kwargs": call_spec.get("tensor_kwargs", {}),
        })
    return groups


def _render_group_rows(group: dict, with_header: bool) -> list[str]:
    """Render rows for one shape group: optional colspan=4 header + 2-col params + tensor remark."""
    rows: list[str] = []
    if with_header:
        role_tag = {"target": "Target", "generalization": "泛化", "single": ""}.get(group["role"], "")
        label_parts = [p for p in (role_tag, group["name"]) if p]
        label = " · ".join(label_parts) if label_parts else "参数"
        rows.append(
            f'<tr><td colspan="4" style="background:var(--bg-subtle);font-weight:600;">'
            f'Shape: {_escape_html(label)}</td></tr>'
        )

    structured: dict[str, str] = {}
    for inp in group.get("inputs", []) or []:
        name = inp.get("name", "")
        shape = inp.get("shape", [])
        dtype = inp.get("dtype", "unknown")
        structured[name] = f"shape={shape}, dtype={dtype}"
    for k, v in (group.get("scalar_args") or {}).items():
        structured[k] = str(v)

    tensor_kwargs_info: list[str] = []
    for k, v in (group.get("tensor_kwargs") or {}).items():
        if isinstance(v, dict):
            tensor_kwargs_info.append(
                f"{k}: shape={v.get('shape', [])}, dtype={v.get('dtype', 'unknown')}"
            )
        else:
            tensor_kwargs_info.append(f"{k}: {v}")

    items = list(structured.items())
    for i in range(0, len(items), 2):
        k1, v1 = items[i]
        if i + 1 < len(items):
            k2, v2 = items[i + 1]
            rows.append(f"<tr><td>{k1}</td><td>{v1}</td><td>{k2}</td><td>{v2}</td></tr>")
        else:
            rows.append(f"<tr><td>{k1}</td><td>{v1}</td><td></td><td></td></tr>")

    if tensor_kwargs_info:
        remark = "; ".join(tensor_kwargs_info)
        rows.append(
            f'<tr><td colspan="4" style="color:var(--text-muted);font-size:0.8rem;">'
            f'<strong>辅助张量:</strong> {_escape_html(remark)}</td></tr>'
        )
    return rows


def build_test_case_rows(test_case: dict, output_dir: str = "") -> str:
    """Build HTML table rows for test case parameters.

    Multi-shape call_spec → emit one section per shape group with a header row.
    Legacy single-shape call_spec → emit a flat 2-col table (no header).
    """
    call_spec = None
    if output_dir:
        call_spec = load_json(os.path.join(output_dir, "shared", "call_spec.json"))

    groups: list[dict] = []
    if call_spec:
        groups = _extract_shape_groups(call_spec)

    # Fallback: synthesize one group from the test_case dict if call_spec missing/empty
    if not groups and test_case:
        # Treat each kv as a scalar arg so they still show up
        groups = [{
            "role": "single",
            "name": "",
            "inputs": [],
            "scalar_args": {k: str(v) for k, v in test_case.items()},
            "tensor_kwargs": {},
        }]

    if not groups:
        return '<tr><td colspan="4">参数信息不可用</td></tr>'

    multi = len(groups) > 1 or (groups and groups[0]["role"] != "single")
    all_rows: list[str] = []
    for g in groups:
        all_rows.extend(_render_group_rows(g, with_header=multi))

    if not all_rows:
        return '<tr><td colspan="4">参数信息不可用</td></tr>'
    return "\n".join(all_rows)


def build_hardware_rows(hw: dict, eval_info: dict) -> str:
    """Build HTML table rows for hardware info."""
    chip = hw.get("chip_model", "Unknown")
    cube = hw.get("core_num_cube") or hw.get("core_num", "?")
    # Fallback: for 910B series, cube and vector core counts are identical
    vector = hw.get("core_num_vector") or hw.get("vector_core_num")
    if vector is None:
        vector = cube
    rows = [
        f"<tr><td>芯片</td><td>Ascend {chip}</td></tr>",
        f"<tr><td>CubeCore / VectorCore</td><td>{cube} / {vector}</td></tr>",
    ]
    # Add a note when vector core count was inferred from cube core count
    if hw.get("core_num_vector") is None and hw.get("vector_core_num") is None:
        rows.append(
            f'<tr><td colspan="2" style="color:var(--text-muted);font-size:0.8rem;">'
            f'注：VectorCore 数量未单独提供，按 Ascend {chip} 架构推断与 CubeCore 相同（每 AI Core 含 1 Cube + 1 Vector）'
            f"</td></tr>"
        )
    if hw.get("peak_bw_gbps"):
        rows.append(f"<tr><td>峰值带宽</td><td>{hw['peak_bw_gbps']} GB/s</td></tr>")
    backend = eval_info.get("eval_backend", "forge")
    profiling_method = eval_info.get("profiling_method", "msprof")
    # Normalize: strip trailing " profiling" or "op profiling" suffixes
    for suffix in (" op profiling", " profiling"):
        if profiling_method.endswith(suffix):
            profiling_method = profiling_method[:-len(suffix)]
    rows.append(f"<tr><td>评估后端</td><td>{backend}</td></tr>")
    rows.append(f"<tr><td>性能采集</td><td>{profiling_method}</td></tr>")
    return "\n".join(rows)


def build_evolution_table_rows(rounds: list[dict], wm: dict, baseline_time: float, best_variant: dict | None) -> str:
    """Build HTML table rows for the full evolution trajectory."""
    rows = []
    best_round = best_variant["round"] if best_variant else -1
    best_par = best_variant["parallel"] if best_variant else -1

    for r in rounds:
        rn, pn = r["round"], r["parallel"]
        vid = f"{rn}-V{pn}"
        node = get_node_for_variant(wm, rn, pn)
        node_id = node.get("id", "N/A") if node else "N/A"
        strategy = "+".join(node.get("strategy_combination", [])) if node else "—"
        if node and node.get("mode") == "open_exploration":
            strategy = "open"
        # When world model node is missing, label as heuristic sampling
        if not node:
            strategy = "分层采样"
        desc_raw = node.get("description", "") if node else r.get("impl_note", "")[:60]
        desc = desc_raw[:40] if len(desc_raw) > 40 else desc_raw

        ev = r.get("eval")
        is_best = rn == best_round and pn == best_par

        # Check compilation_success and precision_passed at top level or in comparison
        compilation_ok = ev.get("compilation_success") if ev else None
        precision_ok = ev.get("precision_passed") if ev else None
        if compilation_ok is None and ev and "comparison" in ev:
            compilation_ok = ev["comparison"].get("compilation_success")
            precision_ok = ev["comparison"].get("precision_passed")

        if not ev or not compilation_ok:
            reason = ""
            if node and node.get("failure_reason"):
                reason = node["failure_reason"][:30]
            rows.append(
                f'<tr class="invalid"><td><span class="vid">{vid}</span></td>'
                f'<td><span class="tag tag-node">{node_id}</span></td>'
                f'<td><span class="sid">{strategy}</span></td>'
                f"<td>{desc}</td>"
                f'<td colspan="3"><span class="badge badge-skip">无效 — {reason}</span></td></tr>'
            )
            continue

        evolved_time = ev.get("evolved", {}).get("time_us", 0)
        if not precision_ok:
            rows.append(
                f'<tr class="invalid"><td><span class="vid">{vid}</span></td>'
                f'<td><span class="tag tag-node">{node_id}</span></td>'
                f'<td><span class="sid">{strategy}</span></td>'
                f"<td>{desc}</td>"
                f'<td>{evolved_time:.2f}</td><td>—</td>'
                f'<td><span class="badge badge-fail">精度失败</span></td></tr>'
            )
            continue

        speedup = baseline_time / evolved_time if evolved_time > 0 else 0
        if is_best:
            tr_class = ' class="best"'
            badge = '<span class="badge badge-best">最优</span>'
        elif speedup >= 1.0:
            tr_class = ""
            badge = '<span class="badge badge-pass">有效</span>'
        else:
            tr_class = ""
            badge = '<span class="badge badge-fail">退化</span>'

        rows.append(
            f"<tr{tr_class}><td><span class=\"vid\">{vid}</span></td>"
            f'<td><span class="tag tag-node">{node_id}</span></td>'
            f'<td><span class="sid">{strategy}</span></td>'
            f"<td>{desc}</td>"
            f"<td>{evolved_time:.2f}</td><td>{speedup:.3f}x</td>"
            f"<td>{badge}</td></tr>"
        )
    return "\n".join(rows)


def build_chart_script(rounds: list[dict], baseline_time: float) -> str:
    """Build Chart.js initialization script."""
    labels = []
    data = []
    for r in rounds:
        ev = r.get("eval")
        if not ev:
            continue
        # Check compilation_success and precision_passed at top level or in comparison
        compilation_ok = ev.get("compilation_success")
        precision_ok = ev.get("precision_passed")
        if compilation_ok is None and "comparison" in ev:
            compilation_ok = ev["comparison"].get("compilation_success")
            precision_ok = ev["comparison"].get("precision_passed")
        if not compilation_ok or not precision_ok:
            continue
        evolved_time = ev.get("evolved", {}).get("time_us")
        if evolved_time and evolved_time > 0:
            # Skip extreme outliers that would distort the chart
            if baseline_time > 0 and evolved_time > baseline_time * 5:
                continue
            labels.append(f"{r['round']}-V{r['parallel']}")
            data.append(round(evolved_time, 2))

    if not data:
        return "// No valid data for chart"

    best_idx = data.index(min(data))
    labels_json = json.dumps(labels)
    data_json = json.dumps(data)

    return f"""
var evoCtx = document.getElementById('evolutionChart');
if (evoCtx) {{
    new Chart(evoCtx, {{
        type: 'line',
        data: {{
            labels: {labels_json},
            datasets: [{{
                label: '耗时 (us)',
                data: {data_json},
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88,166,255,0.1)',
                fill: true, tension: 0.3, pointRadius: 4,
                pointBackgroundColor: function(ctx) {{ return ctx.dataIndex === {best_idx} ? '#3fb950' : '#58a6ff'; }},
                pointBorderColor: function(ctx) {{ return ctx.dataIndex === {best_idx} ? '#3fb950' : '#58a6ff'; }},
                pointRadius: function(ctx) {{ return ctx.dataIndex === {best_idx} ? 8 : 4; }}
            }}, {{
                label: 'Baseline ({baseline_time} us)',
                data: Array({len(data)}).fill({baseline_time}),
                borderColor: 'rgba(248,81,73,0.5)',
                borderDash: [6, 4], pointRadius: 0, fill: false
            }}]
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{
                legend: {{ labels: {{ color: '#c9d1d9' }} }},
                tooltip: {{ callbacks: {{ afterLabel: function(ctx) {{
                    if (ctx.datasetIndex === 0) return '加速比: ' + ({baseline_time} / ctx.raw).toFixed(3) + 'x';
                }} }} }}
            }},
            scales: {{
                x: {{ ticks: {{ color: '#8b949e', maxRotation: 45 }}, grid: {{ color: 'rgba(48,54,61,0.5)' }} }},
                y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: 'rgba(48,54,61,0.5)' }}, title: {{ display: true, text: '耗时 (us)', color: '#8b949e' }} }}
            }}
        }}
    }});
}}"""


# Excluded directories and extensions for modified files filtering
_EXCLUDED_MOD_DIRS = {"tests", "docs", "__pycache__", ".git"}
_EXCLUDED_MOD_EXTS = {".bak", ".tmp", ".swp", ".swo", ".orig", ".rej", ".o", ".so", ".a", ".pyc", ".pyo"}


def _should_include_modified_file(rel_path: str) -> bool:
    """Filter out non-core and temporary files from modified_files listing."""
    parts = rel_path.split(os.sep)
    if any(p in _EXCLUDED_MOD_DIRS for p in parts):
        return False
    ext = os.path.splitext(rel_path)[1].lower()
    if ext in _EXCLUDED_MOD_EXTS:
        return False
    basename = os.path.basename(rel_path)
    if basename.startswith(".") or basename.endswith("~"):
        return False
    return True


def _detect_modified_files_prefix(modified_files: list[str], op_name: str) -> str | None:
    """Detect if modified_files have a common op-name prefix directory.

    ops-partial sometimes copies files into modified_files/<op_name>/...
    instead of modified_files/... directly. This detects such prefixes.
    """
    if not modified_files:
        return None
    first_dirs = set()
    for f in modified_files:
        parts = f.split(os.sep)
        if len(parts) > 1:
            first_dirs.add(parts[0])
    if len(first_dirs) == 1:
        prefix = list(first_dirs)[0]
        op_flat = op_name.replace("_", "").lower()
        prefix_flat = prefix.replace("_", "").lower()
        if op_flat in prefix_flat or prefix_flat in op_flat:
            return prefix
    return None


def _strip_prefix(rel_path: str, prefix: str | None) -> str:
    """Strip leading directory prefix from a relative path."""
    if not prefix:
        return rel_path
    prefix_with_sep = prefix + os.sep
    if rel_path.startswith(prefix_with_sep):
        return rel_path[len(prefix_with_sep):]
    return rel_path


def _is_source_file(rel_path: str) -> bool:
    """Check if a file is a source code file that should appear in diff sections."""
    ext = os.path.splitext(rel_path)[1].lower()
    return ext in (".cpp", ".h", ".hpp", ".c", ".py", ".cc", ".cu", ".cuh")


def build_code_diff_sections(best_variant: dict, baseline_source: str | None) -> str:
    """Build HTML code diff sections for the best variant's modified files."""
    if not best_variant:
        return '<div class="card"><p style="color:var(--text-muted);">无最优变体数据</p></div>'

    sections = []
    mod_dir = best_variant.get("modified_files_dir", "")
    prefix = best_variant.get("modified_files_prefix", "")
    copy_id_counter = 0

    for rel_path in best_variant.get("modified_files", []):
        # Skip metadata files that are not actual source code
        if not _is_source_file(rel_path):
            continue
        mod_file = os.path.join(mod_dir, rel_path)
        if not os.path.isfile(mod_file):
            continue

        # Normalize path for baseline comparison (strip op-name prefix if present)
        norm_path = _strip_prefix(rel_path, prefix)

        copy_id_counter += 1
        code_id = f"diff-{copy_id_counter}"

        if not baseline_source:
            # No baseline source — cannot determine if file is new or modified, skip
            copy_id_counter -= 1
            continue

        base_file = os.path.join(baseline_source, norm_path)
        if not os.path.isfile(base_file):
            # No baseline counterpart — this is a NEW file, skip per policy
            # (report should only show diffs of existing files)
            copy_id_counter -= 1
            continue

        with open(base_file, "r", encoding="utf-8", errors="replace") as f:
            base_lines = f.readlines()
        with open(mod_file, "r", encoding="utf-8", errors="replace") as f:
            mod_lines = f.readlines()
        diff = list(difflib.unified_diff(
            base_lines, mod_lines,
            fromfile=f"a/{norm_path}", tofile=f"b/{norm_path}", lineterm=""
        ))
        if not diff:
            # Files are identical — skip entirely
            copy_id_counter -= 1
            continue
        diff_html = _format_diff_html(diff)
        sections.append(
            f'<div class="card"><h3 style="margin-top:0">{norm_path}</h3>'
            f'<div class="copy-wrapper">'
            f'<button class="copy-btn" onclick="copyCode(this,\'{code_id}\')">复制</button>'
            f'<pre><code id="{code_id}">{diff_html}</code></pre>'
            f"</div></div>"
        )

    if not sections:
        return '<div class="card"><p style="color:var(--text-muted);">无代码修改（所有文件与基线一致）</p></div>'
    return "\n".join(sections)


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_diff_html(diff_lines: list[str]) -> str:
    """Format unified diff lines as colored HTML."""
    parts = []
    for line in diff_lines:
        line_escaped = _escape_html(line.rstrip("\n"))
        if line.startswith("@@"):
            parts.append(f'<span class="diff-hunk">{line_escaped}</span>')
        elif line.startswith("+") and not line.startswith("+++"):
            parts.append(f'<span class="diff-add">{line_escaped}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            parts.append(f'<span class="diff-del">{line_escaped}</span>')
        else:
            parts.append(line_escaped)
    return "\n".join(parts)


def build_decision_tree_html(wm: dict, best_path: list[str], baseline_time: float, best_score: float = 0, rounds: list[dict] | None = None) -> str:
    """Build vertical decision tree HTML from world model."""
    if not wm:
        return '<p style="color:var(--text-muted);">无世界模型数据</p>'

    nodes = wm.get("decision_tree", {}).get("nodes", {})
    root_children = wm.get("decision_tree", {}).get("root", {}).get("children", [])
    if not root_children and "root" in nodes:
        root_children = nodes["root"].get("children", [])

    best_score = best_score or wm.get("best_speedup") or wm.get("best_score", 0)

    # Build a lookup from solution_ref -> evolved_time so we can show speedups
    # computed with the *unified* baseline_time instead of the per-variant
    # baseline that world-model scores are based on (avoids small mismatches).
    ref_to_time: dict[str, float] = {}
    if rounds:
        for r in rounds:
            ref = f"round_{r['round']}/parallel_{r['parallel']}"
            ev = r.get("eval")
            if ev:
                t = ev.get("evolved", {}).get("time_us")
                if t and t > 0:
                    ref_to_time[ref] = t

    def _node_speedup(node: dict) -> float | None:
        """Return speedup using unified baseline if eval data exists, else world-model score."""
        ref = node.get("solution_ref")
        if ref and ref in ref_to_time and baseline_time > 0:
            return baseline_time / ref_to_time[ref]
        return node.get("score")

    # Build unified parent→children mapping from BOTH sources:
    # 1. node["children"] arrays
    # 2. node["parent_id"] fields (reverse mapping)
    # This prevents omissions when one source is incomplete.
    children_map: dict[str, list[str]] = {}  # parent_id → [child_ids]
    for nid, n in nodes.items():
        if nid == "root":
            continue
        # Source 1: parent's children list (already handled via root_children above)
        # Source 2: node's parent_id field
        pid = n.get("parent_id")
        if pid:
            children_map.setdefault(pid, [])
            if nid not in children_map[pid]:
                children_map[pid].append(nid)
    # Merge root_children into the map
    children_map.setdefault("root", [])
    for cid in root_children:
        if cid not in children_map["root"]:
            children_map["root"].append(cid)
    # Also merge each node's own children list
    for nid, n in nodes.items():
        for cid in n.get("children", []):
            children_map.setdefault(nid, [])
            if cid not in children_map[nid]:
                children_map[nid].append(cid)

    def get_children(nid: str) -> list[str]:
        """Get unified children list for a node."""
        return children_map.get(nid, [])

    # BFS from root using unified children
    levels = []  # Each level: list of (node_id, node_data)
    # Level 0: root
    levels.append([("root", {"id": "root", "score": 1.0, "strategy_combination": [],
                              "description": f"Baseline {baseline_time:.2f}us",
                              "children": get_children("root"),
                              "status": "passed", "mode": "root"})])
    current_ids = get_children("root")
    visited = {"root"}
    while current_ids:
        level = []
        next_ids = []
        for nid in current_ids:
            if nid in visited:
                continue
            visited.add(nid)
            node = nodes.get(nid)
            if node:
                level.append((nid, node))
                for child_id in get_children(nid):
                    if child_id not in visited:
                        next_ids.append(child_id)
        if level:
            levels.append(level)
        current_ids = next_ids

    # Append any orphaned nodes (not reachable via BFS) as an extra level.
    # These nodes exist in the tree but have broken parent references.
    orphaned = [(nid, node) for nid, node in nodes.items() if nid not in visited]
    if orphaned:
        levels.append(orphaned)

    html_parts = []
    for lvl_idx, level in enumerate(levels):
        # Render level nodes
        html_parts.append('<div class="vtree-level">')
        for nid, node in level:
            css_class = _get_node_css_class(nid, node, best_path, best_score)
            strategy = "+".join(node.get("strategy_combination", []))
            if node.get("mode") == "open_exploration":
                strategy = "open"
            desc = node.get("description", "")[:25]
            score = _node_speedup(node)
            speedup_class = ""
            speedup_text = ""
            if nid == "root":
                speedup_text = f"{baseline_time:.2f}us"
            elif score is not None:
                speedup_text = f"{score:.3f}x"
                if score and abs(score - best_score) < 0.001:
                    speedup_class = " best"
                    speedup_text += " 最优"
                elif score >= 1.0:
                    speedup_class = " good"
                else:
                    speedup_class = " bad"
            else:
                speedup_text = "无效"
                speedup_class = " bad"

            time_text = ""
            if score and score > 0 and nid != "root":
                time_text = f"{baseline_time / score:.2f}us"

            html_parts.append(f'<div class="vtree-node {css_class}">')
            html_parts.append(f'    <div class="vnode-title">{nid}</div>')
            if strategy:
                html_parts.append(f'    <div class="vnode-strategy">{strategy}</div>')
            if desc:
                html_parts.append(f'    <div class="vnode-desc">{_escape_html(desc)}</div>')
            if time_text:
                html_parts.append(f'    <div class="vnode-perf">{time_text}</div>')
            html_parts.append(f'    <div class="vnode-speedup{speedup_class}">{speedup_text}</div>')
            html_parts.append("</div>")
        html_parts.append("</div>")

        # Render connectors to next level using unified children_map
        if lvl_idx < len(levels) - 1:
            next_level = levels[lvl_idx + 1]
            # Build parent→children mapping for this level pair
            connections = []
            for i, (nid, node) in enumerate(level):
                children = get_children(nid)
                child_indices = []
                best_child_idx = -1
                for j, (cnid, _) in enumerate(next_level):
                    if cnid in children:
                        child_indices.append(j)
                        if cnid in best_path:
                            best_child_idx = j
                if child_indices:
                    connections.append({"pi": i, "ci": child_indices, "bi": best_child_idx})

            # Also handle orphan nodes: connect them to their parent_id if
            # the parent is in the current level (handles broken children lists)
            if lvl_idx == len(levels) - 2 and orphaned:
                for j, (cnid, cnode) in enumerate(next_level):
                    pid = cnode.get("parent_id")
                    if pid:
                        # Check if this child is already connected
                        already_connected = any(j in c["ci"] for c in connections)
                        if not already_connected:
                            # Find parent in current level
                            for i, (pnid, _) in enumerate(level):
                                if pnid == pid:
                                    # Add to existing connection or create new
                                    existing = next((c for c in connections if c["pi"] == i), None)
                                    if existing:
                                        existing["ci"].append(j)
                                    else:
                                        connections.append({"pi": i, "ci": [j], "bi": -1})
                                    break

            conn_json = json.dumps(connections)
            html_parts.append(
                f'<div class="vtree-connectors" data-parent-level="{lvl_idx}" '
                f"data-connections='{conn_json}'>"
                f'<svg class="vtree-svg"></svg></div>'
            )

    return "\n".join(html_parts)


def _get_node_css_class(nid: str, node: dict, best_path: list[str], best_score: float) -> str:
    if nid == "root":
        return "node-best-path node-root"
    score = node.get("score")
    status = node.get("status", "")
    if score and abs(score - best_score) < 0.001:
        return "node-best-leaf"
    if nid in best_path:
        return "node-best-path"
    if status == "open" and node.get("failure_reason"):
        return "node-invalid"
    if score is not None and score < 1.0:
        return "node-regress"
    return "node-minor"


def build_best_strategy_section(best_variant: dict | None, wm: dict | None, baseline_time: float = 0.0) -> str:
    """Build the best strategy analysis section with auto-filled content from world model."""
    if not best_variant:
        return ""

    node = get_node_for_variant(wm, best_variant["round"], best_variant["parallel"]) if wm else None
    desc = node.get("description", "") if node else best_variant.get("impl_note", "")
    strategies = node.get("strategy_combination", []) if node else []
    impl_note = best_variant.get("impl_note", "")

    # Profiling insight from node or eval
    profiling = {}
    if node and node.get("profiling_insight"):
        profiling = node["profiling_insight"]
    elif best_variant.get("eval") and best_variant["eval"].get("evolved", {}).get("pipeline"):
        profiling = best_variant["eval"]["evolved"]["pipeline"]

    # Build analysis text from available data
    analysis_parts = []
    if desc:
        analysis_parts.append(desc)
    if impl_note and impl_note != desc:
        analysis_parts.append(impl_note)

    full_analysis = "\n\n".join(analysis_parts) if analysis_parts else "暂无详细分析"

    # Performance data
    best_time = best_variant.get("time_us", 0)
    speedup = baseline_time / best_time if baseline_time > 0 and best_time > 0 else 1.0

    # Strategy tags
    strategy_tags_html = ""
    if strategies:
        strategy_tags_html = '<div style="margin-top:0.8rem;">' + "".join(
            f'<span class="tag tag-strategy">{s}</span>' for s in strategies
        ) + '</div>'

    # Profiling one-liner
    one_liner = profiling.get("profiling_one_liner", "") if isinstance(profiling, dict) else ""

    # World model coverage note
    wm_coverage_note = ""
    if wm:
        wm_best_variant = wm.get("best_variant", "")
        wm_best_speedup = wm.get("best_speedup", 0)
        actual_round = best_variant.get("round", 0)
        actual_par = best_variant.get("parallel", 0)
        wm_rounds_covered = set()
        for nid, n in wm.get("decision_tree", {}).get("nodes", {}).items():
            sol = n.get("solution_ref") or ""
            if sol.startswith("round_"):
                try:
                    wm_rounds_covered.add(int(sol.split("/")[0].split("_")[1]))
                except ValueError:
                    pass
        total_rounds = actual_round  # best_variant is from the latest round
        if total_rounds > 0 and len(wm_rounds_covered) < total_rounds:
            missing = total_rounds - len(wm_rounds_covered)
            wm_coverage_note = (
                f'<li><strong>世界模型数据</strong>: '
                f'仅覆盖前 {len(wm_rounds_covered)} 轮（共 {total_rounds} 轮），'
                f'缺少后续 {missing} 轮的决策节点。'
                f'世界模型记录最优为 {wm_best_variant} ({wm_best_speedup:.3f}x)，'
                f'而实际评估最优为 round_{actual_round}/parallel_{actual_par} ({speedup:.3f}x)。</li>'
            )

    html = f'''<h2>最优策略分析</h2>
<div class="card">
<h3 style="margin-top:0">优化概述</h3>
<p style="color:var(--text-muted);font-size:0.88rem;margin-bottom:0.8rem;">{_escape_html(full_analysis[:300])}</p>

<h3>关键数据</h3>
<ul style="color:var(--text);font-size:0.88rem;line-height:1.8;">
<li><strong>Baseline</strong>: {baseline_time:.2f} us → <strong>最优</strong>: {best_time:.2f} us ({speedup:.3f}x)</li>
<li><strong>策略组合</strong>: {" + ".join(strategies) if strategies else "开放探索"}</li>
{f'<li><strong>Profiling</strong>: {_escape_html(one_liner)}</li>' if one_liner else ''}
{wm_coverage_note}
</ul>
{strategy_tags_html}
</div>'''
    return html


def build_failure_analysis(rounds: list[dict], wm: dict | None) -> str:
    """Build failure analysis section if there are failed rounds."""
    failures = []
    for r in rounds:
        ev = r.get("eval")
        # Check compilation_success at top level or in comparison (same logic as table rows)
        compilation_ok = ev.get("compilation_success") if ev else None
        precision_ok = ev.get("precision_passed") if ev else None
        if compilation_ok is None and ev and "comparison" in ev:
            compilation_ok = ev["comparison"].get("compilation_success")
            precision_ok = ev["comparison"].get("precision_passed")

        if not ev or not compilation_ok or not precision_ok:
            node = get_node_for_variant(wm, r["round"], r["parallel"]) if wm else None
            reason = ""
            if node and node.get("failure_reason"):
                reason = node["failure_reason"]
            elif not ev:
                reason = "evaluation_results.json 缺失"
            elif not compilation_ok:
                reason = "编译失败"
            elif not precision_ok:
                reason = "精度失败"
            else:
                reason = "未知错误"
            failures.append((r["round"], r["parallel"], reason))

    if not failures:
        return ""

    rows = []
    for rn, pn, reason in failures:
        rows.append(f"<tr><td>{rn}-V{pn}</td><td>{_escape_html(reason)}</td></tr>")

    return (
        '<h2>失败/无效轮次分析</h2>\n<div class="card">\n'
        "<table>\n<tr><th>变体</th><th>原因</th></tr>\n"
        + "\n".join(rows)
        + "\n</table>\n</div>"
    )


def build_apply_cmd(best_variant: dict | None, output_dir: str, baseline_source: str | None) -> str:
    """Build the apply command for the best variant.

    Only includes files that:
    1. Are source files (not docs, tests, configs)
    2. Actually differ from baseline (when baseline is available)
    3. Exist in baseline (new files are excluded — only modifications)

    Output is a concise shell snippet: variable definitions + cp commands only.
    No shebang, comments, or set -e — just the actionable commands.
    """
    if not best_variant:
        return "# 无最优变体"

    best_path = best_variant["path"]
    mod_dir = os.path.join(best_path, "modified_files")
    if not os.path.isdir(mod_dir):
        return "# 未找到修改文件目录"

    # Determine repo path
    if baseline_source:
        repo_path = baseline_source
    else:
        repo_path = "<your_ops_repo_path>"

    prefix = best_variant.get("modified_files_prefix", "")

    # Only include source files that actually differ from baseline
    changed_files: list[str] = []
    seen: set[str] = set()  # deduplication
    for root, _, files in os.walk(mod_dir):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), mod_dir)
            if not _is_source_file(rel):
                continue
            # Normalize path for baseline comparison
            norm_rel = _strip_prefix(rel, prefix)
            # Deduplicate (same normalized path from different prefix paths)
            if norm_rel in seen:
                continue
            seen.add(norm_rel)
            if baseline_source:
                base_file = os.path.join(baseline_source, norm_rel)
                if not os.path.isfile(base_file):
                    continue  # New file — skip, only include modifications
                with open(base_file, "r", encoding="utf-8", errors="replace") as f:
                    base_lines = f.readlines()
                with open(os.path.join(mod_dir, rel), "r", encoding="utf-8", errors="replace") as f:
                    mod_lines = f.readlines()
                diff = list(difflib.unified_diff(base_lines, mod_lines, lineterm=""))
                if not diff:
                    continue  # identical to baseline
            changed_files.append(norm_rel)

    if not changed_files:
        return "# 无源代码修改（所有文件与基线一致）"

    # Use forward slash consistently for path separator in prefix
    prefix_path = prefix.replace(os.sep, "/") + "/" if prefix else ""

    # Build concise command snippet: variables + cp commands only
    lines: list[str] = [
        f'REPO_ROOT="{repo_path}"',
        f'OUTPUT_DIR="{os.path.abspath(output_dir)}"',
        f'BEST_VARIANT="{os.path.relpath(best_path, output_dir)}"',
        "",
    ]

    # Group by directory for cleaner output and merge cp commands
    dirs: dict[str, list[str]] = {}
    for rel in sorted(changed_files):
        parent = os.path.dirname(rel) or "."
        dirs.setdefault(parent, []).append(rel)

    for parent, files in sorted(dirs.items()):
        for f in files:
            src = f'"$OUTPUT_DIR/$BEST_VARIANT/modified_files/{prefix_path}{f}"'
            dst = f'"$REPO_ROOT/{f}"'
            lines.append(f"cp {src} {dst}")

    return "\n".join(lines)


def self_check_report(html: str, rounds: list[dict], wm: dict | None, baseline_time: float, round_timing: dict | None = None) -> list[str]:
    """Run self-check on the generated report and return a list of warnings."""
    warnings = []

    # 1. Check for placeholder residues
    if "LLM_FILL" in html:
        warnings.append("报告仍含未填充的 LLM_FILL 占位符")

    # 2. Check best variant validity
    best = find_best_variant(rounds, baseline_time)
    if best:
        ev = best.get("eval", {})
        compilation_ok = ev.get("compilation_success")
        if compilation_ok is None and "comparison" in ev:
            compilation_ok = ev["comparison"].get("compilation_success")
        precision_ok = ev.get("precision_passed")
        if precision_ok is None and "comparison" in ev:
            precision_ok = ev["comparison"].get("precision_passed")
        evolved_time = ev.get("evolved", {}).get("time_us")

        if not compilation_ok:
            warnings.append(f"最优变体 R{best['round']}-P{best['parallel']} compilation_success=False，请确认")
        if not precision_ok:
            warnings.append(f"最优变体 R{best['round']}-P{best['parallel']} precision_passed=False，请确认")
        if evolved_time is None or evolved_time <= 0:
            warnings.append(f"最优变体 R{best['round']}-P{best['parallel']} time_us 无效 ({evolved_time})，请确认")

        # 2a. Check for redundant/unmodified files in diff section
        mod_files = best.get("modified_files", [])
        prefix = best.get("modified_files_prefix", "")
        if len(mod_files) > 20:
            warnings.append(f"代码修改部分文件数异常多 ({len(mod_files)} 个)，可能存在冗余未修改文件")
        # Check if path prefix was detected but not stripped properly
        if prefix:
            if f'modified_files/{prefix}/' not in html:
                warnings.append(f"modified_files 路径前缀 '{prefix}' 可能未正确处理，diff 或 apply-cmd 路径可能不匹配")

    # 3. Check for suspicious all-failed pattern
    failed_count = 0
    for r in rounds:
        ev = r.get("eval", {})
        compilation_ok = ev.get("compilation_success")
        if compilation_ok is None and "comparison" in ev:
            compilation_ok = ev["comparison"].get("compilation_success")
        if not compilation_ok:
            failed_count += 1
    if failed_count == len(rounds) and len(rounds) > 0:
        warnings.append(f"所有 {len(rounds)} 个变体均标记为编译失败，请检查 evaluation_results.json 格式或 ops-partial 子agent 输出")

    # 4. Check world model data consistency
    if wm:
        nodes = wm.get("decision_tree", {}).get("nodes", {})
        missing_nodes = []
        for r in rounds:
            ref = f"round_{r['round']}/parallel_{r['parallel']}"
            node = None
            for n in nodes.values():
                if n.get("solution_ref") == ref:
                    node = n
                    break
            if not node:
                missing_nodes.append(f"R{r['round']}-P{r['parallel']}")
                continue
            node_score = node.get("score")
            ev = r.get("eval", {})
            evolved_time = ev.get("evolved", {}).get("time_us")
            if node_score and node_score > 0 and evolved_time and evolved_time > 0:
                computed_speedup = baseline_time / evolved_time
                if abs(computed_speedup - node_score) > 0.05:
                    warnings.append(
                        f"R{r['round']}-P{r['parallel']} speedup 不一致: "
                        f"world_model={node_score:.3f}x, eval={computed_speedup:.3f}x"
                    )
        if missing_nodes:
            warnings.append(
                f"以下变体在世界模型中无对应节点（可能 refine 被跳过）: {', '.join(missing_nodes)}"
            )

    # 5. Check test case rows
    if '<td colspan="4">参数信息不可用</td>' in html:
        warnings.append("测试用例参数未获取到，请检查 shared/call_spec.json 是否存在")

    # 6. Check model info (accept plain model name in subtitle or explicit tag)
    _known_models = ["Kimi", "Claude", "GPT", "Gemini", "Qwen", "DeepSeek", "Llama"]
    has_model_info = any(m in html for m in _known_models) or "Model:" in html
    if not has_model_info:
        warnings.append("未包含模型信息，请检查 session JSONL 是否存在或模型字段是否可用")

    # 7. Check subtitle quality (reject placeholder-like values)
    subtitle_match = re.search(r'<p class="subtitle">(.+?)</p>', html)
    if subtitle_match:
        subtitle_text = subtitle_match.group(1).strip()
        # Strip HTML tags for plain text check
        plain_subtitle = re.sub(r'<[^>]+>', '', subtitle_text).strip()
        bad_subtitles = {"test", "report", "title", "default", ""}
        if plain_subtitle.lower() in bad_subtitles:
            warnings.append(f"报告副标题不规范: '{plain_subtitle}'，请检查 --title 参数")

    # 8. Check apply command for redundancy / invalid paths
    apply_match = re.search(r'id="apply-cmd">(.*?)</code>', html, re.DOTALL)
    if apply_match:
        cmd_text = apply_match.group(1)
        cp_lines = [l for l in cmd_text.split('\n') if l.strip().startswith('cp ')]
        if len(cp_lines) > 20:
            warnings.append(f"应用最优变体命令数量过多 ({len(cp_lines)} 条)，可能存在冗余或路径前缀未剥离")
        # Check for potential path mismatch (op-name prefix in destination)
        if best:
            prefix = best.get("modified_files_prefix", "")
            if prefix and any(f'$REPO_ROOT/{prefix}/' in l for l in cp_lines):
                warnings.append(f"apply-cmd 目标路径包含算子名前缀 '{prefix}/'，可能与仓库实际结构不匹配")
        # Check for old-style boilerplate that should have been removed
        if '#!/bin/bash' in cmd_text or 'set -e' in cmd_text:
            warnings.append("apply-cmd 含冗余指令（shebang / set -e），应仅保留变量定义和 cp 命令")

    # 9. Check round timing for anomalies
    if round_timing and round_timing.get("rounds"):
        for rn, rd in round_timing["rounds"].items():
            duration = rd.get("duration_minutes", 0)
            if duration <= 0:
                warnings.append(f"轮次 {rn} 耗时异常 (<=0 分钟: {duration:.2f})，请检查目录时间戳数据")
            elif duration < 0.5:
                warnings.append(f"轮次 {rn} 耗时过短 ({duration:.2f} 分钟)，可能存在预创建目录时间戳干扰")

    # 10. Check token table completeness (cache_creation column)
    if '词元用量统计' in html:
        if 'Cache Creation' not in html and 'cache_creation' not in html:
            warnings.append("词元用量统计表格缺少 Cache Creation 列，统计可能不完整")

    # 11. Decision tree topology validation
    if wm:
        nodes = wm.get("decision_tree", {}).get("nodes", {})
        if nodes:
            # Check all variant rounds have corresponding tree nodes
            variant_refs = {f"round_{r['round']}/parallel_{r['parallel']}" for r in rounds}
            node_refs = {n.get("solution_ref") for n in nodes.values() if n.get("solution_ref")}
            missing_in_tree = variant_refs - node_refs
            if missing_in_tree:
                warnings.append(
                    f"决策树缺少 {len(missing_in_tree)} 个变体节点: "
                    f"{', '.join(sorted(missing_in_tree)[:5])}"
                )
            # Check for orphan nodes (no parent and not root children)
            root_children_set = set(wm.get("decision_tree", {}).get("root", {}).get("children", []))
            if "root" in nodes:
                root_children_set.update(nodes["root"].get("children", []))
            all_children = set()
            for n in nodes.values():
                all_children.update(n.get("children", []))
            all_children.update(root_children_set)
            orphans = [nid for nid in nodes if nid != "root" and nid not in all_children
                       and not nodes[nid].get("parent_id")]
            if orphans:
                warnings.append(
                    f"决策树存在 {len(orphans)} 个孤立节点（无父引用）: "
                    f"{', '.join(orphans[:5])}"
                )

    # 12. Code diff new-file leak detection (Issue 2)
    # Diffs should only show modifications to existing files, not full new-file content
    diff_sections = re.findall(r'<pre><code class="diff">(.*?)</code></pre>', html, re.DOTALL)
    for i, diff_content in enumerate(diff_sections):
        lines = diff_content.strip().split('\n')
        add_lines = [l for l in lines if l.startswith('+') and not l.startswith('+++')]
        del_lines = [l for l in lines if l.startswith('-') and not l.startswith('---')]
        # If a diff has only additions and zero deletions, it's likely a new file leak
        if len(add_lines) > 10 and len(del_lines) == 0:
            # Check for the "new file" pattern in hunk headers
            has_dev_null = any('--- /dev/null' in l or '--- a/dev/null' in l for l in lines)
            if has_dev_null:
                warnings.append(
                    f"代码修改第 {i+1} 段 diff 为新增文件（非已有文件修改），不应出现在报告中"
                )

    # 13. Apply command duplicate target detection (Issue 5)
    if apply_match:
        cmd_text = apply_match.group(1)
        cp_lines = [l for l in cmd_text.split('\n') if l.strip().startswith('cp ')]
        # Extract destination paths and check for duplicates
        destinations = []
        for cp_line in cp_lines:
            parts = cp_line.strip().split()
            if len(parts) >= 3:
                destinations.append(parts[-1])
        dup_dests = [d for d in destinations if destinations.count(d) > 1]
        if dup_dests:
            unique_dups = sorted(set(dup_dests))
            warnings.append(
                f"apply-cmd 存在 {len(unique_dups)} 个重复目标路径: "
                f"{', '.join(unique_dups[:3])}"
            )

    # 14. Total duration reasonableness (Issue 4)
    if round_timing:
        total_dur = round_timing.get("total_evolution_minutes", 0)
        rounds_dur = sum(
            rd.get("duration_minutes", 0)
            for rd in round_timing.get("rounds", {}).values()
        )
        # Total should not exceed sum of rounds by more than 3x (indicates idle inflation)
        if rounds_dur > 0 and total_dur > rounds_dur * 3:
            warnings.append(
                f"总耗时 ({total_dur:.1f}min) 远超各轮次之和 ({rounds_dur:.1f}min)，"
                f"可能包含了空闲等待时间"
            )
        # Total should be positive if we have rounds
        if total_dur <= 0 and len(round_timing.get("rounds", {})) > 0:
            warnings.append("总耗时为 0 或负值，资源统计可能异常")

    # 15. No source code modification detection (all files identical to baseline)
    if best:
        mod_files = best.get("modified_files", [])
        if len(mod_files) == 0 and best.get("eval", {}).get("evolved", {}).get("time_us", 0) > 0:
            warnings.append(
                "无源代码修改（所有文件与基线一致），但最优变体有有效耗时 — "
                "请检查 ops-partial 子agent 是否正确保存了修改文件到 modified_files/"
            )

    return warnings


def build_strategy_legend(wm: dict | None) -> str:
    """Build strategy code legend from world model nodes."""
    if not wm:
        return ""
    nodes = wm.get("decision_tree", {}).get("nodes", {})
    strategies = set()
    for node in nodes.values():
        for s in node.get("strategy_combination", []):
            strategies.add(s)
    if not strategies:
        return ""
    # Known strategy names
    known = {
        "P1": "双缓冲", "P2": "自适应Tiling", "P3": "多核均衡", "P4": "向量化DMA",
        "P5": "标量优化", "P6": "内存复用", "P7": "循环展开", "P8": "流水线优化",
        "P9": "确定性输出", "P10": "预取", "P11": "数据对齐", "P12": "融合计算",
        "P13": "核数调整",
    }
    parts = []
    for s in sorted(strategies):
        name = known.get(s, "")
        parts.append(f"{s}={name}" if name else s)
    return "策略编号: " + " ".join(parts) + " | open=开放探索模式"


def main():
    parser = argparse.ArgumentParser(description="Generate evolution optimization HTML report")
    parser.add_argument("output_dir", help="Evolution output directory path")
    parser.add_argument("--baseline-source", default=None, help="Baseline source directory for diff")
    parser.add_argument("--title", default=None, help="Custom report title")
    parser.add_argument("--pipeline", default="ops-evo", help="Pipeline type (default: ops-evo)")
    parser.add_argument("--session-jsonl", default=None, help="Explicit path to the session JSONL file for token/timing stats")
    args = parser.parse_args()

    output_dir = args.output_dir.rstrip("/")
    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # Load data
    op_name, timestamp = parse_output_dir_name(output_dir)
    baseline_eval = load_json(os.path.join(output_dir, "baseline_evaluation.json"))
    wm = load_json(os.path.join(output_dir, "world_model_final.json"))
    if not wm:
        wm = load_json(os.path.join(output_dir, "world_model.json"))

    if not baseline_eval and not wm:
        print("Error: neither baseline_evaluation.json nor world_model_final.json found", file=sys.stderr)
        sys.exit(2)

    # Extract baseline info
    baseline_time = 0.0
    test_case = {}
    hw_params = {}
    eval_info = {}

    if baseline_eval:
        baseline_time = baseline_eval.get("baseline", {}).get("time_us", 0)
        # Forge mode: baseline tested against itself stores actual time in evolved.time_us
        if baseline_time <= 0:
            baseline_time = baseline_eval.get("evolved", {}).get("time_us", 0)
        # Flat format fallback (forge raw result or standalone files)
        if baseline_time <= 0:
            baseline_time = baseline_eval.get("baseline_time_us", 0)
        test_case = baseline_eval.get("test_case", {})
        if not test_case:
            test_case = baseline_eval.get("test_config", {})
        eval_info = baseline_eval
    if wm:
        if baseline_time <= 0:
            baseline_time = wm.get("baseline", {}).get("time_us", 0)
        if baseline_time <= 0:
            baseline_time = wm.get("baseline_time_us", 0)
        hw_params = wm.get("hw_params", {})
        if not test_case:
            test_case = wm.get("test_case", {})

    # Fallback: parse test_cases.csv for test case params
    if not test_case:
        csv_path = os.path.join(output_dir, "shared", "test_cases.csv")
        if os.path.isfile(csv_path):
            test_case = _parse_test_cases_csv(csv_path)

    # Collect rounds and find best
    rounds = collect_rounds(output_dir)
    # Detect common prefix in modified_files for path normalization
    all_mod_files = []
    for r in rounds:
        all_mod_files.extend(r.get("modified_files", []))
    prefix = _detect_modified_files_prefix(all_mod_files, op_name)
    for r in rounds:
        r["modified_files_prefix"] = prefix
    enrich_rounds_from_world_model(rounds, wm, baseline_time)
    best_variant = find_best_variant(rounds, baseline_time) if baseline_time > 0 else None
    best_path, wm_best_score = find_best_path(wm, best_variant, baseline_time) if wm else ([], 0.0)

    # Compute metrics
    best_time = best_variant["time_us"] if best_variant else baseline_time
    speedup = baseline_time / best_time if best_time > 0 else 1.0
    # Time change: negative = faster (time decreased), positive = slower (time increased)
    time_reduction = (best_time / baseline_time - 1) * 100 if baseline_time > 0 else 0

    # Build subtitle
    chip = hw_params.get("chip_model", "Unknown")
    num_rounds = max((r["round"] for r in rounds), default=0) if rounds else 0
    num_parallels = max((r["parallel"] for r in rounds), default=0) + 1 if rounds else 0
    default_subtitle = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} | Ascend {chip} | {num_rounds}轮 x {num_parallels}变体"
    # Reject placeholder-like titles (e.g., "test", "report", empty)
    _bad_titles = {"test", "report", "", "title", "default"}
    subtitle = default_subtitle if (args.title or "").strip().lower() in _bad_titles else (args.title or default_subtitle)

    # Load template
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "..", "templates", "evolution-report.html")
    with open(template_path, "r", encoding="utf-8") as f:
        template_str = f.read()

    # Auto-detect baseline source from shared/original/ if not provided
    baseline_source = args.baseline_source
    if not baseline_source:
        auto_baseline = os.path.join(output_dir, "shared", "original")
        if os.path.isdir(auto_baseline):
            baseline_source = auto_baseline

    # Build all sections
    if wm:
        node_count = len(wm.get('decision_tree', {}).get('nodes', {}))
        # Detect world-model coverage gaps
        wm_rounds_covered = set()
        for nid, n in wm.get("decision_tree", {}).get("nodes", {}).items():
            sol = n.get("solution_ref") or ""
            if sol.startswith("round_"):
                try:
                    wm_rounds_covered.add(int(sol.split("/")[0].split("_")[1]))
                except ValueError:
                    pass
        missing_rounds = num_rounds - len(wm_rounds_covered) if num_rounds > 0 else 0
        coverage_warn = ""
        if missing_rounds > 0:
            coverage_warn = (
                f' | <span style="color:var(--orange);font-weight:600;">'
                f'警告: 仅覆盖 {len(wm_rounds_covered)}/{num_rounds} 轮，缺少 {missing_rounds} 轮节点'
                f'</span>'
            )
        wm_score_note = ""
        if wm_best_score and abs(wm_best_score - speedup) > 0.01:
            wm_score_note = f" | 世界模型记录: {wm_best_score:.3f}x"
        tree_summary = (
            f"共 {node_count} 个节点 | "
            f'最优路径: <span style="color:var(--green);font-weight:600;">'
            f"{' → '.join(best_path)}</span> ({speedup:.3f}x)"
            f"{wm_score_note}{coverage_warn}"
        )
    else:
        tree_summary = "无世界模型数据"

    # Parse resource statistics (timing and tokens)
    # Try to find session directory from output_dir
    session_dir = None
    session_jsonl = None  # The specific JSONL file for this session

    # Explicit override via CLI
    if args.session_jsonl:
        jsonl_path = Path(args.session_jsonl)
        if jsonl_path.exists():
            session_jsonl = jsonl_path
            session_dir = jsonl_path.parent
        else:
            print(f"Warning: --session-jsonl not found: {args.session_jsonl}", file=sys.stderr)

    if not session_jsonl:
        # Search all possible .claude/projects directories (not just Path.home())
        # because the script may run as a different user than the session owner
        search_roots = []
        # Add common home directories and any extra paths from environment
        home_candidates = [Path.home()]
        extra_homes_env = os.environ.get("EVO_EXTRA_HOME_DIRS", "")
        for extra in extra_homes_env.split(":"):
            extra = extra.strip()
            if extra:
                home_candidates.append(Path(extra))
        for home_dir in home_candidates:
            if home_dir.exists():
                projects_dir = home_dir / ".claude" / "projects"
                if projects_dir.exists():
                    search_roots.append(projects_dir)
        # Also check current user's home if not already included
        current_home = Path.home()
        if not any(str(current_home) in str(r) for r in search_roots):
            projects_dir = current_home / ".claude" / "projects"
            if projects_dir.exists():
                search_roots.append(projects_dir)
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for r in search_roots:
            if str(r) not in seen:
                seen.add(str(r))
                deduped.append(r)
        search_roots = deduped

        # Collect all candidate session files across ALL search roots
        candidate_sessions = []
        output_path = Path(output_dir).resolve()
        output_str = str(output_path)
        output_marker = "/output/"
        if output_marker in output_str:
            output_as_proj = "-" + output_str[:output_str.index(output_marker)].replace("/", "-").replace("_", "-").lstrip("-")
        else:
            output_as_proj = "-" + output_str.replace("/", "-").replace("_", "-").lstrip("-")

        for projects_dir in search_roots:
            if not projects_dir.exists():
                continue
            for proj_dir in projects_dir.iterdir():
                if not proj_dir.is_dir():
                    continue
                if proj_dir.name != output_as_proj:
                    continue
                session_files = list(proj_dir.glob("*.jsonl"))
                for sf in session_files:
                    subagent_dir = sf.parent / sf.stem / "subagents"
                    has_subagents = (
                        subagent_dir.exists() and len(list(subagent_dir.glob("*.jsonl"))) > 0
                    )
                    candidate_sessions.append({
                        "proj_dir": proj_dir,
                        "session_file": sf,
                        "has_subagents": has_subagents,
                        "mtime": sf.stat().st_mtime,
                    })

        # Use round_1 birthtime as an anchor to find the matching session
        round1_dir = Path(output_dir) / "round_1"
        round1_birth = None
        if round1_dir.exists():
            round1_birth = _get_dir_birthtime(round1_dir)
            if round1_birth:
                # Convert seconds -> milliseconds for comparison
                round1_birth_ms = round1_birth * 1000 if round1_birth < 1e12 else round1_birth
                filtered = []
                for cand in candidate_sessions:
                    start_ms, end_ms = get_session_time_range(cand["session_file"])
                    if start_ms is not None and end_ms is not None:
                        if start_ms <= round1_birth_ms <= end_ms:
                            filtered.append(cand)
                if filtered:
                    candidate_sessions = filtered

        if candidate_sessions:
            best = max(candidate_sessions, key=lambda c: (c["has_subagents"], c["mtime"]))
            session_dir = best["proj_dir"]
            session_jsonl = best["session_file"]

    session_stats = {}
    round_timing = {}
    if session_dir and session_jsonl:
        try:
            session_stats = parse_session_stats(str(session_dir), str(session_jsonl))
        except Exception as e:
            print(f"Warning: Failed to parse session stats: {e}", file=sys.stderr)
            session_stats = {}

    try:
        round_timing = parse_round_timing(output_dir)
    except Exception as e:
        print(f"Warning: Failed to parse round timing: {e}", file=sys.stderr)
        round_timing = {}

    # Build resource stats section
    resource_stats_section = ""
    if session_stats or round_timing:
        try:
            resource_stats_section = build_resource_stats_section(session_stats, round_timing)
        except Exception as e:
            print(f"Warning: Failed to build resource stats section: {e}", file=sys.stderr)
            resource_stats_section = ""

    # Model info: prefer config file, fallback to session JSONL
    model_name = get_model_from_config()
    if not model_name and session_stats:
        model_name = session_stats.get("model")
    # Append model info to subtitle for a cleaner header
    if model_name:
        subtitle = f"{model_name} | {subtitle}"

    replacements = {
        "OP_NAME": op_name,
        "SUBTITLE": subtitle,
        "BASELINE_TIME": f"{baseline_time:.2f}",
        "BEST_TIME": f"{best_time:.2f}",
        "SPEEDUP": f"{speedup:.3f}",
        "TIME_REDUCTION": f"{time_reduction:+.1f}",
        "TEST_CASE_ROWS": build_test_case_rows(test_case, output_dir),
        "HARDWARE_ROWS": build_hardware_rows(hw_params, eval_info),
        "EVOLUTION_TABLE_ROWS": build_evolution_table_rows(rounds, wm, baseline_time, best_variant),
        "STRATEGY_LEGEND": build_strategy_legend(wm),
        "BEST_STRATEGY_SECTION": build_best_strategy_section(best_variant, wm, baseline_time),
        "CODE_DIFF_SECTIONS": build_code_diff_sections(best_variant, baseline_source),
        "TREE_SUMMARY": tree_summary,
        "DECISION_TREE_HTML": build_decision_tree_html(wm, best_path, baseline_time, speedup, rounds),
        "FAILURE_ANALYSIS_SECTION": build_failure_analysis(rounds, wm),
        "RESOURCE_STATS_SECTION": resource_stats_section,
        "APPLY_CMD": build_apply_cmd(best_variant, output_dir, baseline_source),
        "CHART_INIT_SCRIPT": build_chart_script(rounds, baseline_time),
        "MODEL_INFO": "",
    }

    # Render template
    tmpl = Template(template_str)
    html = tmpl.safe_substitute(replacements)

    # Self-check
    warnings = self_check_report(html, rounds, wm, baseline_time, round_timing)
    if warnings:
        print("\n[报告自检警告]", file=sys.stderr)
        for w in warnings:
            print(f"  [WARN] {w}", file=sys.stderr)
    else:
        print("\n[报告自检通过] 无警告", file=sys.stderr)

    # Write output
    out_filename = f"evolution-report_{op_name}_{timestamp}.html"
    out_path = os.path.join(output_dir, out_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(out_path)


if __name__ == "__main__":
    main()
