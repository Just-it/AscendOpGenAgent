#!/usr/bin/env python3
"""
PostToolUse hook for Edit/Write — advances phase after code edits.

- reference.py in GENERATE_REF → GENERATE_KERNEL or BASELINE (depending on
  whether kernel.py is still a placeholder)
- editable file in GENERATE_KERNEL → BASELINE
- editable file in EDIT → no phase change; Claude runs pipeline.py when done

plan.md is never a legal target for Edit/Write — hook_guard_edit blocks it
at every phase and directs Claude to create_plan.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input, emit_status, norm_abs_fwd_slash, extract_target_path
from workflow import PhaseController
from phase_machine import (
    read_phase, get_guidance, _load_config_safe,
    get_task_dir, touch_heartbeat,
    validate_reference, validate_kernel, is_placeholder_file,
    EDIT, BASELINE, GENERATE_REF, GENERATE_KERNEL,
)
from git_utils import commit_in_task


def _same_path(a: str, b: str) -> bool:
    return norm_abs_fwd_slash(a) == norm_abs_fwd_slash(b)


_WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}


def _seed_and_advance(task_dir: str, current_phase: str, kind: str,
                      validator, files_to_commit, next_phase: str) -> bool:
    """Validate, git-commit the seed, then advance phase.

    `kind` is the human-readable file label ("reference.py" / "kernel.py")
    used in the status messages; `validator` is the validate_* function;
    `files_to_commit` is the list passed to git; `next_phase` is the phase
    to write on success. Returns True iff phase advanced; False (and
    emit_status already called) otherwise.
    """
    ok, err = validator(task_dir)
    if not ok:
        emit_status(
            f"[AR] {kind} invalid — phase stays at {current_phase}.\n"
            f"     {err}\n"
            f"     Re-Edit {kind} to fix; downstream phases will not "
            f"advance until it passes."
        )
        return False
    commit_ok, info = commit_in_task(
        task_dir, files_to_commit,
        f"autoresearch: seed {kind} ({current_phase})",
    )
    if not commit_ok:
        emit_status(
            f"[AR] {kind} validated but seed commit FAILED — phase stays "
            f"at {current_phase}.\n"
            f"     git error: {info}\n"
            f"     Resolve the git issue (e.g. clear .git/index.lock, "
            f"check disk space, fix .git/config), then re-Edit {kind} to "
            f"retry the commit."
        )
        return False
    PhaseController(task_dir).on_seed_validated(next_phase)
    emit_status(f"[AR] {kind} validated. Phase -> {next_phase}. "
                f"{get_guidance(task_dir)}")
    return True


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") not in _WRITE_TOOLS:
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    file_path = extract_target_path(hook_input)
    if not file_path:
        sys.exit(0)

    phase = read_phase(task_dir)
    is_ref = _same_path(file_path, os.path.join(task_dir, "reference.py"))

    config = _load_config_safe(task_dir)
    is_editable = False
    if config:
        try:
            rel = os.path.relpath(file_path, task_dir).replace("\\", "/")
            is_editable = rel in set(config.editable_files)
        except ValueError:
            is_editable = False

    if is_ref and phase == GENERATE_REF:
        # Route to GENERATE_KERNEL if kernel.py is still the scaffold
        # placeholder, else straight to BASELINE.
        next_phase = GENERATE_KERNEL if is_placeholder_file(
            os.path.join(task_dir, "kernel.py")
        ) else BASELINE
        _seed_and_advance(task_dir, GENERATE_REF, "reference.py",
                          validate_reference, ["reference.py"], next_phase)

    elif is_editable and phase == GENERATE_KERNEL:
        editable_files = list(config.editable_files) if config else ["kernel.py"]
        _seed_and_advance(task_dir, GENERATE_KERNEL, "kernel.py",
                          validate_kernel, editable_files, BASELINE)

    elif is_editable and phase == EDIT:
        emit_status(
            f"[AR] Code edited. Continue editing OR run: "
            f"python .autoresearch/scripts/pipeline.py \"{task_dir}\""
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
