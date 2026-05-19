"""phase_machine package — facade over four single-concern submodules.

Dependency direction (top depends on lower):
    guidance, phase_policy
        → validators
            → state_store

This `__init__.py` re-exports the public surface; new code may import
directly from the submodule (`from phase_machine.state_store import ...`).
`auto_rollback` lives in git_utils and is re-exported here for callers
that still import it from phase_machine.
"""
# fmt: off
from .models import Progress
from .state_store import (
    # Phase constants
    INIT, GENERATE_REF, GENERATE_KERNEL, BASELINE, PLAN, EDIT,
    DIAGNOSE, REPLAN, FINISH, ALL_PHASES,
    # File constants
    PHASE_FILE, PROGRESS_FILE, HISTORY_FILE, PLAN_FILE, PLAN_ITEMS_FILE,
    EDIT_MARKER_FILE, PENDING_SETTLE_FILE, HEARTBEAT_FILE, ACTIVE_TASK_FILE,
    DIAGNOSE_ARTIFACT_TEMPLATE, DIAGNOSE_MARKER_TEMPLATE, DIAGNOSE_ATTEMPTS_CAP,
    # Path builders
    state_path, plan_path, progress_path, history_path, edit_marker_path,
    pending_settle_path,
    diagnose_artifact_path, diagnose_marker,
    # Phase I/O
    read_phase, write_phase,
    # Progress + history I/O
    load_progress, save_progress, append_history, update_progress,
    # Active-task pointer
    get_task_dir, set_task_dir, touch_heartbeat,
    # Per-op task_dir pointer (scaffold -> batch.run.py handoff)
    task_dir_pointer_path, write_task_dir_pointer, read_task_dir_pointer,
    # Helpers
    parse_last_json_line,
)
from .validators import (
    KERNEL_PLACEHOLDER, REFERENCE_PLACEHOLDER_PREFIX,
    is_placeholder_file,
    validate_reference, validate_kernel, validate_plan, validate_diagnose,
    DiagnoseState, diagnose_state,
    DIAGNOSE_NEED_DIAGNOSIS, DIAGNOSE_READY, DIAGNOSE_MANUAL_FALLBACK,
    get_plan_items, parse_plan_text, has_pending_items, get_active_item,
    is_settled_table_header,
    # Internal — re-exported so debug / extension scripts that previously
    # reached into phase_machine can still find them at the old name.
    _PLAN_ITEM_RE, _PLAN_TAG_RE, _REF_RUNCHECK_SCRIPT,
)
from .phase_policy import (
    # Layer 1: classifier (pure function, command shape only)
    classify, CommandShape,
    parse_canonical_ar, parse_script_names, parse_invoked_ar_script,
    is_single_foreground_ar_invocation,
    # Layer 3: predicates hooks call
    check_bash, check_edit,
    # Phase transitions
    compute_next_phase, compute_resume_phase,
    # Layer 2: phase tables — public-ish because tests / dashboards
    # reference them. Underscore-prefixed for a "do not mutate at
    # runtime" hint rather than for true privacy.
    _AR_ALLOWED_BY_PHASE, _OTHER_ALLOWED_BY_PHASE, _LIFECYCLE_SCRIPTS,
    _EDIT_RULES, _SUBPROCESS_ONLY_AR_SCRIPTS,
    _CANONICAL_AR_RE,
)
from .guidance import (
    get_guidance,
    # _load_config_safe is consumed by hook_post_edit (only external user
    # of the helper today). Re-exported even though it's underscore-prefixed
    # because the call site predates the package split.
    _load_config_safe,
    # XML schema and field rules — referenced by name in create_plan.py's
    # docstring and used by tests / dashboards that want to render the
    # canonical example. Re-exported to keep the old `phase_machine.
    # _PLAN_XML_EXAMPLE` reference resolvable.
    _PLAN_XML_EXAMPLE, _PLAN_FIELD_RULES,
)

# auto_rollback used to live in phase_machine; the implementation moved to
# git_utils alongside commit_in_task / ensure_git_identity. Re-export so
# stale `from phase_machine import auto_rollback` still resolves.
import os
import sys
_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from git_utils import auto_rollback  # noqa: E402
# fmt: on
