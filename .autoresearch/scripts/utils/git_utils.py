"""Shared git helpers. `commit_in_task` returns (True, "<hash>"|"noop")
on success and (False, "<reason>") on real failure; both callers
(scaffold + workflow.round) route through it for consistent
error-handling. `git config user.name/email` is set every call for
fresh-box safety; the writes are idempotent."""
import os
import subprocess
import sys
from typing import Optional

_GIT_USER_NAME = "autoresearch"
_GIT_USER_EMAIL = "auto@research"


def _run(cmd: list, cwd: str, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def ensure_git_identity(task_dir: str) -> None:
    """Idempotent local user.name/email — closes 'undefined identity'
    on fresh CI boxes."""
    _run(["git", "config", "user.name", _GIT_USER_NAME], cwd=task_dir)
    _run(["git", "config", "user.email", _GIT_USER_EMAIL], cwd=task_dir)


# git commit messages that mean "nothing committed but not an error".
# All three are treated as noop so a KEEP doesn't get demoted to FAIL
# just because some sidecar file (.ar_state, profiling output, ...)
# is untracked.
_NOOP_COMMIT_MARKERS = (
    "nothing to commit",
    "no changes added",
    "nothing added to commit",
)


def commit_in_task(task_dir: str, paths, message: str) -> tuple:
    """Stage `paths` under `task_dir` and commit. Missing files in
    `paths` are skipped. Returns (True, "<short hash>"), (True, "noop"),
    or (False, "<reason>")."""
    try:
        ensure_git_identity(task_dir)
        for p in paths:
            if p != "." and not os.path.exists(os.path.join(task_dir, p)):
                continue
            r = _run(["git", "add", "--", p], cwd=task_dir)
            if r.returncode != 0:
                return False, f"git add {p!r} failed: {(r.stderr or r.stdout).strip()[-300:]}"
        r = _run(["git", "commit", "-m", message], cwd=task_dir)
        if r.returncode != 0:
            blob = (r.stdout or "") + (r.stderr or "")
            if any(m in blob for m in _NOOP_COMMIT_MARKERS):
                return True, "noop"
            return False, blob.strip()[-400:] or "git commit returned non-zero with no output"
        h = _run(["git", "rev-parse", "--short", "HEAD"], cwd=task_dir)
        return True, h.stdout.strip() if h.returncode == 0 else "ok"
    except subprocess.TimeoutExpired:
        return False, "git operation timed out (>10s) — check for index lock or fs contention"
    except Exception as e:
        return False, f"unexpected error: {e}"


def is_working_tree_clean(task_dir: str) -> bool:
    """True iff `git status --porcelain` is empty. Errors → False
    (better to leave a stale marker than falsely declare clean)."""
    try:
        r = _run(["git", "status", "--porcelain"], cwd=task_dir, timeout=5)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0 and not r.stdout.strip()


def current_head_short(task_dir: str) -> Optional[str]:
    """Short hash of HEAD, or None if rev-parse fails. round.py uses
    this to keep best_commit pointing at HEAD when a KEEP commit was a
    no-op."""
    try:
        r = _run(["git", "rev-parse", "--short", "HEAD"], cwd=task_dir)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def auto_rollback(task_dir: str):
    """Revert editable_files to HEAD via `git checkout HEAD --`. Silent
    on git failures — rollback is a recovery path."""
    try:
        # __file__ is scripts/utils/git_utils.py — climb two to reach scripts/.
        _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from task_config import load_task_config
        config = load_task_config(task_dir)
        if config is None:
            return
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=task_dir, capture_output=True, text=True,
        ).stdout.strip()
        for f in config.editable_files:
            fpath = os.path.relpath(os.path.join(task_dir, f), repo_root)
            subprocess.run(["git", "checkout", "HEAD", "--", fpath],
                           cwd=repo_root, capture_output=True)
    except Exception as e:
        print(f"[AR] Rollback failed: {e}", file=sys.stderr)
