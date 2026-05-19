"""Shared git helpers for autoresearch lifecycle code.

scaffold and hook_post_edit both need to commit files inside a task's git
repo. Historically each path open-coded its own `subprocess.run(["git", ...])`
sequence, with subtly different error-handling — scaffold used `check=True`
(crash loud on any git failure) while hook_post_edit's `_commit_seed`
swallowed every error as a stderr WARNING and let phase advance anyway.

The latter caused a class of bugs where the seed kernel never made it into
HEAD, baseline ran fine on the in-tree code, phase walked forward, and the
problem only surfaced two phases later as a misleading "uncommitted changes
from previous round" block in `_edit_phase_git_gate`.

This module is the single canonical implementation of "stage these files
and commit". Both scaffold and hook_post_edit now route through it. The
contract:

    ok, info = commit_in_task(task_dir, paths, message)
        ok   == True  → either created a commit (info=short hash) or there
                        was nothing to commit (info="noop"); both safe to
                        proceed.
        ok   == False → commit really failed (info = human-readable cause).
                        Caller decides whether to abort, hold phase, or
                        propagate.

Defensive `git config user.name/email` runs every call so we don't depend
on `.git/config` having been set by an earlier path. Repeated config sets
are idempotent.
"""
import os
import subprocess
import sys

_GIT_USER_NAME = "autoresearch"
_GIT_USER_EMAIL = "auto@research"


def _run(cmd: list, cwd: str, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def ensure_git_identity(task_dir: str) -> None:
    """Set local user.name/email if missing.

    Idempotent: writing identity that's already there is a no-op. Doing
    this in every commit path closes the "user.name=undefined on a fresh
    box" failure mode that hit Mode-2 GENERATE_KERNEL commits.
    """
    _run(["git", "config", "user.name", _GIT_USER_NAME], cwd=task_dir)
    _run(["git", "config", "user.email", _GIT_USER_EMAIL], cwd=task_dir)


def commit_in_task(task_dir: str, paths, message: str) -> tuple:
    """Stage `paths` under `task_dir` and create a commit.

    `paths` are task-dir-relative ("kernel.py", "reference.py", ...) or the
    literal "." for an "add everything" first commit. Missing files are
    skipped silently — caller decides whether that's an error.

    Returns (True, "<short hash>") on success, (True, "noop") if there was
    nothing staged worth committing, (False, "<reason>") on any other
    failure.
    """
    try:
        # Identity bootstrap inside the try so a missing/invalid task_dir
        # surfaces as (False, "...") instead of leaking NotADirectoryError /
        # FileNotFoundError out of the function. Callers expect a tuple.
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
            if "nothing to commit" in blob or "no changes added" in blob:
                return True, "noop"
            return False, blob.strip()[-400:] or "git commit returned non-zero with no output"

        h = _run(["git", "rev-parse", "--short", "HEAD"], cwd=task_dir)
        return True, h.stdout.strip() if h.returncode == 0 else "ok"

    except subprocess.TimeoutExpired:
        return False, "git operation timed out (>10s) — check for index lock or fs contention"
    except Exception as e:
        return False, f"unexpected error: {e}"


def auto_rollback(task_dir: str):
    """Revert editable_files to HEAD via `git checkout HEAD --`.

    Used by keep_or_discard (DISCARD/FAIL paths) and by the pipeline's
    quick-check failure branch. Reads `task.yaml.editable_files` to know
    which files to revert. Silent on git failures: rollback is a recovery
    path, swallowing here is appropriate (caller has already decided to
    abandon the round).
    """
    try:
        _scripts_dir = os.path.dirname(os.path.abspath(__file__))
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
