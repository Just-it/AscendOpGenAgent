#!/usr/bin/env python3
"""
Resume an existing autoresearch task.

Usage:
    python .autoresearch/scripts/resume.py [task_dir]

If task_dir is omitted, auto-detects the most recently active task.
Validates state files. Prints the task_dir on success, exits with error if incompatible.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_machine import (
    ALL_PHASES, load_progress,
    progress_path, plan_path, state_path, edit_marker_path,
    has_pending_items,
)


def _find_latest_task() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # 1. Try .active_task
    active_file = os.path.join(project_root, ".autoresearch", ".active_task")
    if os.path.exists(active_file):
        with open(active_file, "r") as f:
            td = f.read().strip()
        if td and os.path.isdir(td):
            return td
        # Stale pointer (task dir was deleted) — clean it so future calls skip straight to scan
        try:
            os.remove(active_file)
        except OSError:
            pass

    # 2. Latest mtime from ar_tasks/
    tasks_dir = os.path.join(project_root, "ar_tasks")
    if not os.path.isdir(tasks_dir):
        return ""

    candidates = []
    for d in os.listdir(tasks_dir):
        full = os.path.join(tasks_dir, d)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "task.yaml")):
            pfile = progress_path(full)
            mtime = os.path.getmtime(pfile) if os.path.exists(pfile) else os.path.getmtime(full)
            candidates.append((full, mtime))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def _validate(task_dir: str) -> tuple[bool, str]:
    """Check task state is resumable. Returns (ok, error_message)."""
    if not os.path.isdir(task_dir):
        return False, f"Not a directory: {task_dir}"

    # Required files
    for rel in ("task.yaml",):
        if not os.path.exists(os.path.join(task_dir, rel)):
            return False, f"Missing required file: {rel}"

    progress = load_progress(task_dir)
    if progress is None:
        return False, "Missing or corrupt .ar_state/progress.json — task was never initialized"

    required_fields = {"task", "eval_rounds", "max_rounds", "status"}
    missing = required_fields - set(progress.keys())
    if missing:
        return False, f"progress.json missing fields: {missing} (incompatible version)"

    # Validate .phase if present
    phase_file = state_path(task_dir, ".phase")
    if os.path.exists(phase_file):
        with open(phase_file, "r") as f:
            phase = f.read().strip()
        if phase not in ALL_PHASES:
            return False, f"Unknown phase '{phase}' in .phase file (incompatible version)"

    # Validate plan.md if present. A fully-consumed plan (0 pending) is legal —
    # compute_resume_phase routes it to REPLAN. validate_plan would reject it
    # for lacking an ACTIVE item, so only validate when pending items exist.
    if os.path.exists(plan_path(task_dir)) and has_pending_items(task_dir):
        from phase_machine import validate_plan
        ok, err = validate_plan(task_dir)
        if not ok:
            return False, f"plan.md invalid: {err}"

    return True, ""


def _check_active_lock(task_dir: str, force: bool) -> None:
    """Check if another Claude Code instance is actively running this task.

    Uses .ar_state/.heartbeat file mtime — if updated in last 3 minutes, warn.
    """
    heartbeat = state_path(task_dir, ".heartbeat")
    if not os.path.exists(heartbeat):
        return

    import time
    age = time.time() - os.path.getmtime(heartbeat)
    if age < 180:  # 3 minutes
        if force:
            print(f"[resume] WARNING: Task was active {age:.0f}s ago. Forcing takeover (--force).",
                  file=sys.stderr)
            return
        print(f"[resume] ERROR: Task is currently active (heartbeat updated {age:.0f}s ago).",
              file=sys.stderr)
        print(f"[resume] Another Claude Code session may be running it.", file=sys.stderr)
        print(f"[resume] If you're sure no other session is running, add --force:", file=sys.stderr)
        print(f"[resume]   /autoresearch --resume --force", file=sys.stderr)
        sys.exit(1)


def main():
    args = sys.argv[1:]
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    task_dir = args[0] if args else None

    if task_dir:
        task_dir = os.path.abspath(task_dir)
    else:
        task_dir = _find_latest_task()
        if not task_dir:
            print("[resume] ERROR: No existing task found in ar_tasks/", file=sys.stderr)
            sys.exit(1)

    ok, err = _validate(task_dir)
    if not ok:
        print(f"[resume] ERROR: Cannot resume {task_dir}", file=sys.stderr)
        print(f"[resume] {err}", file=sys.stderr)
        print("[resume] This task may be from an incompatible version. Start fresh.", file=sys.stderr)
        sys.exit(1)

    _check_active_lock(task_dir, force)

    # Clean stale edit marker (git clean means marker is stale)
    marker = edit_marker_path(task_dir)
    if os.path.exists(marker):
        import subprocess
        try:
            diff = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=task_dir, capture_output=True, text=True, timeout=5,
            )
            if not diff.stdout.strip():
                os.remove(marker)
                print("[resume] Cleaned stale edit marker.", file=sys.stderr)
        except Exception:
            pass

    progress = load_progress(task_dir) or {}
    print(f"[resume] Task: {progress.get('task')}")
    print(f"[resume] Round: {progress.get('eval_rounds')}/{progress.get('max_rounds')}")
    print(f"[resume] Best: {progress.get('best_metric')} | Baseline: {progress.get('baseline_metric')}")
    phase_file = state_path(task_dir, ".phase")
    if os.path.exists(phase_file):
        with open(phase_file) as f:
            print(f"[resume] Phase: {f.read().strip()}")

    # Print task_dir on last line (for easy parsing)
    print(task_dir)


if __name__ == "__main__":
    main()
