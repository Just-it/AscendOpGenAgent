"""State storage layer.

Single owner of `<task_dir>/.ar_state/` and `.autoresearch/.active_task`.
No other module reads/writes these files directly — go through the helpers
here.

What lives in this module:
  - Phase enum constants (used as keys / values throughout).
  - Canonical file basenames inside `.ar_state/` (PHASE_FILE, etc.).
  - Path builders (`state_path`, `plan_path`, `progress_path`, …).
  - Phase I/O (`read_phase`, `write_phase`).
  - Progress I/O (`load_progress` -> Progress, `save_progress`,
    `update_progress`). Progress is a typed dataclass (see models.py)
    so writers construct full objects and the field set is validated.
  - History append (`append_history`).
  - Active-task pointer (`get_task_dir`, `set_task_dir`).
  - Heartbeat touch.
  - JSON-tail parser used by every subprocess output.

Why phase constants live here and not in phase_policy: `read_phase` needs
`ALL_PHASES` to validate; phase_policy in turn needs `compute_next_phase`
to read progress, which lives here. Putting the constants at the bottom
of the dependency stack avoids the cycle.
"""
import json
import os
import sys
from typing import Optional, Union

from .models import Progress


# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

INIT = "INIT"
GENERATE_REF = "GENERATE_REF"
GENERATE_KERNEL = "GENERATE_KERNEL"
BASELINE = "BASELINE"
PLAN = "PLAN"
EDIT = "EDIT"
DIAGNOSE = "DIAGNOSE"
REPLAN = "REPLAN"
FINISH = "FINISH"

ALL_PHASES = {INIT, GENERATE_REF, GENERATE_KERNEL, BASELINE, PLAN, EDIT,
              DIAGNOSE, REPLAN, FINISH}


# ---------------------------------------------------------------------------
# Canonical filenames inside <task_dir>/.ar_state/
# ---------------------------------------------------------------------------

PHASE_FILE = ".phase"
PROGRESS_FILE = "progress.json"
HISTORY_FILE = "history.jsonl"
PLAN_FILE = "plan.md"
PLAN_ITEMS_FILE = "plan_items.xml"  # canonical XML payload path under .ar_state/
EDIT_MARKER_FILE = ".edit_started"
PENDING_SETTLE_FILE = ".pending_settle.json"  # kd_json saved when settle.py fails
HEARTBEAT_FILE = ".heartbeat"
ACTIVE_TASK_FILE = ".active_task"  # under .autoresearch/, not .ar_state/

# DIAGNOSE artifact contract — see CLAUDE.md invariant #10.
# The DIAGNOSE phase is gated on a structured report at this path before
# create_plan.py / Stop become legal. The ar-diagnosis subagent is the
# intended writer (per its prompt + read-only tool isolation), but hook
# payloads do NOT distinguish main agent from subagent — provenance is
# not enforced. Only the artifact's CONTENT is validated. The marker is
# plan-version-aware so a stale prior diagnose can't be replayed across
# REPLAN boundaries.
DIAGNOSE_ARTIFACT_TEMPLATE = "diagnose_v{}.md"
DIAGNOSE_MARKER_TEMPLATE = "[AR DIAGNOSE COMPLETE marker_v{}]"
DIAGNOSE_ATTEMPTS_CAP = 5


# ---------------------------------------------------------------------------
# Project root resolution + active-task pointer
# ---------------------------------------------------------------------------

def _find_project_root() -> str:
    """Walk up from this file to find the dir that has `.autoresearch/`."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, ".autoresearch")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))


_PROJECT_ROOT = _find_project_root()
_ACTIVE_TASK_FILE = os.path.join(_PROJECT_ROOT, ".autoresearch", ACTIVE_TASK_FILE)
_TASK_DIR_POINTERS = os.path.join(_PROJECT_ROOT, ".autoresearch",
                                  ".task_dir_pointers")


def task_dir_pointer_path(op_name: str) -> str:
    """Filesystem path of the per-op task_dir pointer.

    scaffold writes this immediately after creating <repo>/ar_tasks/
    <op>_<ts>_<rand>; batch/run.py reads it instead of mtime-scanning.
    The mtime scan still works as a fallback (tasks scaffolded before
    this change have no pointer), but the pointer is the authoritative
    answer when present.
    """
    safe = op_name.replace("/", "_").replace("\\", "_")
    return os.path.join(_TASK_DIR_POINTERS, safe)


def write_task_dir_pointer(op_name: str, task_dir: str) -> None:
    """Atomic write of the per-op pointer. Tmp + os.replace so a racing
    reader sees either nothing or the full path - never a half line."""
    path = task_dir_pointer_path(op_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(task_dir) + "\n")
    os.replace(tmp, path)


def read_task_dir_pointer(op_name: str) -> Optional[str]:
    """Return the absolute task_dir for op_name, or None when no pointer
    exists / contents are stale (the dir no longer exists)."""
    path = task_dir_pointer_path(op_name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            td = f.read().strip()
    except OSError:
        return None
    if td and os.path.isdir(td):
        return td
    return None


def get_task_dir() -> str:
    """Get active task_dir. Reads from .autoresearch/.active_task file.

    Falls back to AR_TASK_DIR env var for backward compat.
    Returns "" if no active task.
    """
    if os.path.exists(_ACTIVE_TASK_FILE):
        with open(_ACTIVE_TASK_FILE, "r") as f:
            td = f.read().strip()
        if td and os.path.isdir(td):
            return td
    return os.environ.get("AR_TASK_DIR", "")


def set_task_dir(task_dir: str):
    """Write active task_dir to .autoresearch/.active_task."""
    os.makedirs(os.path.dirname(_ACTIVE_TASK_FILE), exist_ok=True)
    with open(_ACTIVE_TASK_FILE, "w") as f:
        f.write(os.path.abspath(task_dir))
    touch_heartbeat(task_dir)


def touch_heartbeat(task_dir: str):
    """Update .ar_state/.heartbeat file to signal this task is active.

    Called from every hook invocation. resume.py checks mtime to detect
    conflicting concurrent Claude Code sessions. A failed touch is reported
    to stderr — silently swallowing it would make the session look dead in
    a way that's nearly impossible to debug.
    """
    try:
        heartbeat = state_path(task_dir, HEARTBEAT_FILE)
        os.makedirs(os.path.dirname(heartbeat), exist_ok=True)
        import time
        with open(heartbeat, "w") as f:
            f.write(f"{int(time.time())}\n")
    except Exception as e:
        print(f"[AR] WARNING: heartbeat write failed ({e}); resume.py may "
              f"misreport this task as inactive.", file=sys.stderr)


# ---------------------------------------------------------------------------
# State file path builders
# ---------------------------------------------------------------------------

def state_path(task_dir: str, name: str) -> str:
    """Path to a file under <task_dir>/.ar_state/. Centralized so no module
    hand-builds state paths."""
    return os.path.join(task_dir, ".ar_state", name)


def plan_path(task_dir: str) -> str:
    return state_path(task_dir, PLAN_FILE)


def progress_path(task_dir: str) -> str:
    return state_path(task_dir, PROGRESS_FILE)


def history_path(task_dir: str) -> str:
    return state_path(task_dir, HISTORY_FILE)


def edit_marker_path(task_dir: str) -> str:
    return state_path(task_dir, EDIT_MARKER_FILE)


def pending_settle_path(task_dir: str) -> str:
    """Sidecar holding the kd_json from a settle.py invocation that failed.

    pipeline.py persists the kd_json here when settle returns non-zero, then
    its NEXT invocation detects this file and retries settle ONLY (skipping
    quick_check/eval/keep_or_discard). Without this replay-only path, a
    re-run of pipeline.py would double-mutate progress.json (eval_rounds++)
    and history.jsonl (duplicate row) before the original settle even gets
    a second chance.

    Removed by pipeline.py on successful settle.
    """
    return state_path(task_dir, PENDING_SETTLE_FILE)


def diagnose_artifact_path(task_dir: str, plan_version: int) -> str:
    """Path to the DIAGNOSE artifact for a given plan_version. The subagent
    Writes to this exact path; the validator reads from it. Plan-version
    suffix prevents stale artifacts from satisfying a later DIAGNOSE round."""
    return state_path(task_dir, DIAGNOSE_ARTIFACT_TEMPLATE.format(plan_version))


def diagnose_marker(plan_version: int) -> str:
    return DIAGNOSE_MARKER_TEMPLATE.format(plan_version)


# ---------------------------------------------------------------------------
# Phase file I/O
# ---------------------------------------------------------------------------

def read_phase(task_dir: str) -> str:
    """Read current phase. Returns INIT if no phase file."""
    path = state_path(task_dir, PHASE_FILE)
    if not os.path.exists(path):
        return INIT
    with open(path, "r") as f:
        phase = f.read().strip()
    return phase if phase in ALL_PHASES else INIT


def write_phase(task_dir: str, phase: str):
    """Write phase to .ar_state/.phase."""
    assert phase in ALL_PHASES, f"Invalid phase: {phase}"
    path = state_path(task_dir, PHASE_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(phase)


# ---------------------------------------------------------------------------
# Progress + history I/O
# ---------------------------------------------------------------------------

def load_progress(task_dir: str) -> Optional[Progress]:
    """Read .ar_state/progress.json into a typed Progress, or None if
    absent/corrupt. Single canonical reader.

    Existing read sites use `progress.get("X", default)`; Progress.get
    mirrors dict.get so they keep working without any rewrite. New code
    should prefer attribute access (`progress.eval_rounds`).
    """
    path = progress_path(task_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return Progress.from_dict(data)


def save_progress(task_dir: str, progress: Union[Progress, dict],
                  *, stamp: bool = True):
    """Write progress to .ar_state/progress.json atomically. Accepts
    Progress or a plain dict (the dict path stays for batch/manifest.py
    which has its own schema and predates the dataclass).

    Atomicity: tmp + os.replace. Earlier non-atomic rewrites occasionally
    let `load_progress` see an empty file mid-write and `compute_next_
    phase` then short-circuit to FINISH well before max_rounds.
    """
    from datetime import datetime, timezone
    path = progress_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(progress, Progress):
        if stamp:
            progress = progress.apply(
                last_updated=datetime.now(timezone.utc).isoformat())
        payload = progress.to_dict()
    else:
        payload = dict(progress)
        if stamp:
            payload["last_updated"] = datetime.now(timezone.utc).isoformat()
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def append_history(task_dir: str, record: dict):
    """Append one JSON record to history.jsonl. Single canonical writer
    used by keep_or_discard and _baseline_init."""
    path = history_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_progress(task_dir: str, **fields) -> Optional[Progress]:
    """Load Progress, .apply(**fields), save. Returns the new Progress.

    Field-name validation is delegated to Progress.apply, so a typo here
    becomes TypeError instead of a silently-dropped attribute (which is
    what `progress["typo"] = ...` produced in the dict-era code).

    Silently no-ops if progress.json does not exist.
    """
    progress = load_progress(task_dir)
    if progress is None:
        return None
    new_progress = progress.apply(**fields)
    try:
        save_progress(task_dir, new_progress, stamp=False)
    except Exception:
        return None
    return new_progress


# ---------------------------------------------------------------------------
# Subprocess output parser (every script tail-emits a JSON line)
# ---------------------------------------------------------------------------

def parse_last_json_line(text: str) -> Optional[dict]:
    """Scan `text` from the bottom up and return the last standalone JSON
    object. Our pipeline/baseline/local-eval scripts all follow the protocol
    "stdout last line is JSON"; this is the single place that reads it.
    """
    if not text:
        return None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None
