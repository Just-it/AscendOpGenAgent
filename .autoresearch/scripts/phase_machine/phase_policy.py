"""Phase rules + bash/edit gates + transition logic.

Bash gate is three layers, each with one job:

  1. CLASSIFIER (`classify`) — pure function: command string → CommandShape.
       AR(name)   canonical `.autoresearch/scripts/<name>.py` invocation
       LIFECYCLE  AR but lifecycle (dashboard / parse_args / scaffold /
                  resume), allowed in every phase
       READONLY   every chain segment is read-only (ls / cat / git read /
                  echo / pwd / ...). NO file redirects, NO mutations.
       OTHER      anything else (ad-hoc bash, malformed AR, writes)
     Consults command text only — no phase, no filesystem.

  2. PHASE TABLE — `_AR_ALLOWED_BY_PHASE` / `_OTHER_ALLOWED_BY_PHASE`.
     Static dicts. LIFECYCLE / READONLY are implicitly allowed everywhere.
     A rule change is a one-row edit.

  3. `check_bash` — global string bans, classify(), table lookup.

The Edit/Write gate (`check_edit`) and phase-transition logic
(`compute_next_phase` / `compute_resume_phase`) live below.
"""
import os
import re
import shlex
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .state_store import (
    INIT, GENERATE_REF, GENERATE_KERNEL, BASELINE, PLAN, EDIT,
    DIAGNOSE, REPLAN, FINISH,
    PLAN_FILE, PLAN_ITEMS_FILE,
    load_progress, plan_path,
)
from .validators import get_active_item, has_pending_items


# === LAYER 1: CLASSIFIER ===============================================

# Canonical AR invocation. Anchored full-command. Group 1 = basename.
#
# Flag whitelist is INTENTIONALLY restrictive — only Python flags that
# really run the script. Excluded: `--version` / `-V` / `--help` / `-h`
# (short-circuit; print and exit), `-c` (runs inline code instead),
# `-m` (runs module instead). Earlier rounds had a generic `--[\w-]+`
# fallback; the result was that `python --version
# .autoresearch/scripts/X.py` falsely classified as AR(X.py), and
# hook_post_bash thought X.py had run.
#
# Optional `(?:\S*?/)?` before `.autoresearch/scripts/` accepts both
# relative invocations (`python .autoresearch/scripts/X.py`) and
# absolute prefixes (`python /repo/.autoresearch/scripts/X.py`,
# `python C:/repo/.autoresearch/scripts/X.py` after backslash
# normalization).
# Common Python flags allowed by both `py` and `python*` (don't affect
# script execution semantics; just runtime tweaks).
_COMMON_PY_FLAGS = (
    r'-[OuBdESIqs]+'                              # combinable singles
    r'|-X(?:\s+\S+|\S+)'                          # -X dev or -Xdev
    r'|-W(?:\s+\S+|\S+)'                          # -W default or -Wdefault
)

_CANONICAL_AR_RE = re.compile(
    r'\A\s*'
    r'(?:[A-Za-z_]\w*=\S+\s+)*'                  # env-var prefixes
    r'(?:'
    # py launcher (Windows): accepts version flags `-3`/`-3.10` AND
    # the common Python flags. The launcher then forwards everything
    # past the version to the actual interpreter.
    r'py(?:\s+(?:-\d+(?:\.\d+)?|' + _COMMON_PY_FLAGS + r'))*'
    r'|'
    # python / python3 / python3.10: NO version flag (Python rejects
    # `-3`/`-3.10` with "Unknown option"). Only the common flags.
    r'python(?:\d+(?:\.\d+)?)?(?:\s+(?:' + _COMMON_PY_FLAGS + r'))*'
    r')'
    r'\s+(?:\S*?/)?\.autoresearch/scripts/'      # script (with optional path prefix)
    r'([A-Za-z_]\w*\.py)\b'                      #   group 1 = basename
    r'(?:\s+(?:'                                 # script args
    # Quoted strings: backtick and `$(` are forbidden (caught by the
    # pre-check in `classify`); `$VAR`/`${VAR}` are still fine.
    r'"[^"]*"|\'[^\']*\'|[^\s&|;()<>`][^\s&|;()<>`]*'
    r'))*'
    r'(?:\s+(?:'                                 # FD redirections
    r'\d?>>?\s*\S+|\d?<\s*\S+|2>&1|1>&2|&>>?\s*\S+'
    r'))*'
    r'\s*\Z'
)

# READONLY check: a small tokenizer + per-command argspec.
#
# Earlier rounds tried "anchored regex with negative lookaheads" and
# kept growing patches — quoted args bypassed the lookahead, then
# `\b`-anchored lookaheads got the boundary wrong (`--output-format`
# vs `--output`). The tokenizer collapses both: shlex strips quotes,
# then the same `args[i] == "--output"` rule catches `--output`,
# `"--output"`, and `'--output'` uniformly.
#
# Allowed shapes:
#   ls/cat/head/tail/wc/grep/echo/pwd ARGS
#   git log/diff/status/show/rev-parse/blame ARGS
#   git branch [--list | --show-current | -a | -r | -v | -vv]*
#   export AR_TASK_DIR=...
#
# ARGS are any tokens EXCEPT `--output` and `--output=...` (so
# `git diff --output=patch.diff` can't smuggle a write). `--output-format`
# and other hyphen-extended flags are NOT blocked.
#
# `find` is intentionally absent (`-delete` / `-exec rm` are too easy a
# smuggle). Agent has Glob / Read tools.

_READONLY_HEAD_SINGLE = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "echo", "pwd",
})
_READONLY_GIT_OPS = frozenset({
    "log", "diff", "status", "show", "rev-parse", "blame",
})
_GIT_BRANCH_LISTING_FLAGS = frozenset({
    "--list", "--show-current", "-a", "-r", "-v", "-vv",
})


def _is_safe_readonly_arg(arg: str) -> bool:
    """Reject `--output` and `--output=...` (file-writing flag).
    Accepts `--output-format`, `--output-something`, etc. — those are
    different flags."""
    return arg != "--output" and not arg.startswith("--output=")


def _has_unquoted_redirect(s: str) -> bool:
    """True iff `s` contains an unquoted `<` or `>`. shlex tokenization
    can't see these as redirection — `cat foo > log` becomes
    `["cat", "foo", ">", "log"]` with all tokens looking like normal
    args. Pre-scan with quote tracking so the readonly check can
    reject the whole segment before tokenizing."""
    in_s = in_d = False
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and not in_s and i + 1 < n:
            i += 2; continue
        if c == "'" and not in_d:
            in_s = not in_s; i += 1; continue
        if c == '"' and not in_s:
            in_d = not in_d; i += 1; continue
        if not in_s and not in_d and c in "<>":
            return True
        i += 1
    return False


def _is_readonly_segment(seg_text: str) -> bool:
    """True iff `seg_text` is a single read-only command. Tokenizes
    via shlex so quoted args go through the same checks as bare args:
    `git diff "--output=patch"` and `git diff --output=patch` both
    fail by the same rule.

    shlex parse error (unbalanced quotes etc.) → False; the caller's
    READONLY claim cannot be made about a malformed segment."""
    s = seg_text.strip()
    if not s:
        return False
    if _has_unquoted_redirect(s):
        return False  # `cat foo > log`, `cat foo 2>&1`, etc.
    try:
        tokens = shlex.split(s, posix=True, comments=False)
    except ValueError:
        return False
    if not tokens:
        return False

    head, args = tokens[0], tokens[1:]

    if head == "git":
        if not args:
            return False
        sub, rest = args[0], args[1:]
        if sub == "branch":
            return all(a in _GIT_BRANCH_LISTING_FLAGS for a in rest)
        if sub in _READONLY_GIT_OPS:
            return all(_is_safe_readonly_arg(a) for a in rest)
        return False

    if head == "export" and args:
        # `export AR_TASK_DIR=<value>` — shlex unquotes the value into
        # the same token, so `AR_TASK_DIR="..."` arrives as one piece.
        return args[0].startswith("AR_TASK_DIR=") and len(args) == 1

    if head in _READONLY_HEAD_SINGLE:
        return all(_is_safe_readonly_arg(a) for a in args)

    return False


_LIFECYCLE_SCRIPTS = frozenset({
    "dashboard.py", "parse_args.py", "scaffold.py", "resume.py",
})


@dataclass(frozen=True)
class CommandShape:
    """Output of `classify`. Pure data — no methods, no phase awareness."""
    klass: str                     # "AR" | "LIFECYCLE" | "READONLY" | "OTHER"
    name: Optional[str] = None     # for AR/LIFECYCLE: the .py basename


def _normalize(command: str) -> str:
    """Forward-slash all backslashes so Windows path forms hit the same
    grammar as POSIX forms."""
    return command.replace("\\", "/")


def _split_chain(command: str) -> List[str]:
    """Split on bash chain operators (&& || ; | bare-&) outside quotes,
    keeping `&` adjacent to `>`/`<` literal as FD redirection. Used by
    `classify` to walk segments for the READONLY check.

    Quote tracking and the redirection-aware `&` rule are inlined here
    because this is the only consumer; AR shapes are vetted in one pass
    by `_CANONICAL_AR_RE` and don't need segmenting."""
    segments: List[str] = []
    cur: List[str] = []
    in_s = in_d = False
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if c == "\\" and not in_s and i + 1 < n:
            cur.append(c); cur.append(command[i + 1]); i += 2; continue
        if c == "'" and not in_d:
            in_s = not in_s; cur.append(c); i += 1; continue
        if c == '"' and not in_s:
            in_d = not in_d; cur.append(c); i += 1; continue
        if in_s or in_d:
            cur.append(c); i += 1; continue
        if i + 1 < n and command[i:i + 2] in ("&&", "||"):
            segments.append("".join(cur)); cur = []; i += 2; continue
        if c in (";", "|"):
            segments.append("".join(cur)); cur = []; i += 1; continue
        if c == "&":
            prev = next((ch for ch in reversed(cur) if not ch.isspace()), None)
            nxt = command[i + 1] if i + 1 < n else None
            if prev in (">", "<") or nxt == ">":
                cur.append(c); i += 1; continue   # FD redirect, not chain
            segments.append("".join(cur)); cur = []; i += 1; continue
        cur.append(c); i += 1
    segments.append("".join(cur))
    return segments


def classify(command: str) -> CommandShape:
    """Sole authority on 'what shape is this command?'. Pure function;
    consults only `command`. Does NOT know phase, does NOT advance state.

    Decision order:
      1. Command-substitution pre-check (`$(...)` and backticks). These
         execute arbitrary commands at parse time — never canonical AR,
         never READONLY. Caught here so neither the AR regex's quoted-
         arg branch nor the READONLY segment grammar has to defend
         individually. `$VAR` / `${VAR}` (variable expansion) is fine.
      2. Canonical AR regex on the normalized command.
      3. Per-segment READONLY check. Read-only heads (cat/git diff/...)
         don't execute their args, so AR script paths in args
         (`cat .autoresearch/scripts/X.py`,
         `git diff -- .autoresearch/scripts/X.py`) are pure references,
         not invocations — safe to allow. Pipes / chains / redirects
         to non-readonly heads still fail the grammar so smuggle attempts
         like `cat .../X.py | bash` are rejected at the second segment.
      4. AR-mention non-canonical and not readonly → malformed AR shape
         (wrappers like `nohup python .../X.py &`, `bash -lc "..."`, etc.).
         Returns OTHER; check_bash rejects in every phase.
      5. Else OTHER.
    """
    if "$(" in command or "`" in command:
        return CommandShape("OTHER")

    normalized = _normalize(command)

    m = _CANONICAL_AR_RE.match(normalized)
    if m:
        name = m.group(1)
        klass = "LIFECYCLE" if name in _LIFECYCLE_SCRIPTS else "AR"
        return CommandShape(klass, name)

    segments = [s for s in _split_chain(command) if s.strip()]
    if segments and all(_is_readonly_segment(s) for s in segments):
        return CommandShape("READONLY")

    if ".autoresearch/scripts/" in normalized:
        return CommandShape("OTHER")  # malformed AR shape

    return CommandShape("OTHER")


# Thin views over `classify` — kept for hook callers that want a
# one-liner answer to a specific question.

def parse_canonical_ar(command: str) -> Optional[str]:
    """Return the AR script basename if `command` classifies as AR or
    LIFECYCLE, else None."""
    shape = classify(command)
    return shape.name if shape.klass in ("AR", "LIFECYCLE") else None


def parse_invoked_ar_script(command: str) -> Optional[str]:
    """Basename of the AR script invocation, or None. Used by
    hook_post_bash for routing on baseline.py / pipeline.py /
    create_plan.py."""
    return parse_canonical_ar(command)


def parse_script_names(command: str) -> List[Tuple[str, str]]:
    """Return [(path, basename)] for the AR script in `command`, or [].
    Used by hook_guard_bash's hallucinated-name pre-check."""
    invoked = parse_canonical_ar(command)
    return [(f".autoresearch/scripts/{invoked}", invoked)] if invoked else []


def is_single_foreground_ar_invocation(command: str, *, script: str) -> tuple:
    """Recovery-gate predicate: is `command` exactly one foreground
    invocation of `<.autoresearch/scripts/script>`?"""
    invoked = parse_canonical_ar(command)
    if invoked is None:
        return False, _CANONICAL_FORM_REJECTION
    if invoked != script:
        return False, f"expected {script!r}, got {invoked!r}"
    return True, ""


# === LAYER 2: PHASE TABLE ==============================================

# Subprocess-only AR scripts: never user-callable. The check fires
# only when classify() returns AR(name) with name in this set — i.e.,
# someone tried `python .autoresearch/scripts/<x>.py task` directly.
# An earlier version did substring-banning on the raw command, but
# that falsely blocked harmless mentions like `echo quick_check.py`
# or `git diff -- .autoresearch/scripts/settle.py` (READONLY shapes
# that mention the name in args, not invocations).
_SUBPROCESS_ONLY_AR_SCRIPTS = {
    "eval_wrapper.py":    "subprocess-only (invoked by pipeline.py)",
    "keep_or_discard.py": "subprocess-only (invoked by pipeline.py)",
    "quick_check.py":     "subprocess-only (invoked by pipeline.py)",
    "settle.py":          "subprocess-only (invoked by pipeline.py)",
    "_baseline_init.py":  "subprocess-only (invoked by baseline.py)",
}

# Per-phase: which AR script names may run.
# LIFECYCLE scripts are always allowed (handled separately, not duplicated).
# Subprocess-only scripts above are blocked everywhere.
# EDIT-recovery (create_plan.py while .pending_settle.json exists) is
# layered on top by hook_guard_bash; this table reflects the normal path.
_AR_ALLOWED_BY_PHASE = {
    INIT:            frozenset(),
    GENERATE_REF:    frozenset(),
    GENERATE_KERNEL: frozenset(),
    BASELINE:        frozenset({"baseline.py"}),
    DIAGNOSE:        frozenset({"create_plan.py"}),
    PLAN:            frozenset({"create_plan.py"}),
    REPLAN:          frozenset({"create_plan.py"}),
    EDIT:            frozenset({"pipeline.py"}),
    FINISH:          frozenset(),
}

# Per-phase: do we accept OTHER (ad-hoc shell, anything not classified
# as AR/LIFECYCLE/READONLY)? Strict phases reject; permissive phases
# accept. AR-mention-but-not-canonical is rejected in EVERY phase
# (handled in check_bash, not via this table).
_OTHER_ALLOWED_BY_PHASE = {
    INIT:            False,
    GENERATE_REF:    False,
    GENERATE_KERNEL: False,
    BASELINE:        False,
    DIAGNOSE:        False,
    PLAN:            True,
    REPLAN:          True,
    EDIT:            True,
    FINISH:          True,
}

# Edit/Write rules: which file classes may be written per phase.
#   "ref"      — reference.py
#   "editable" — anything in task.yaml:editable_files
# plan.md is never in any set — it's machine-generated.
_EDIT_RULES = {
    GENERATE_REF:    {"ref"},
    GENERATE_KERNEL: {"editable"},
    EDIT:            {"editable"},
}


_CANONICAL_FORM_REJECTION = (
    "AR scripts must be invoked directly: "
    "`python .autoresearch/scripts/<name>.py <task_dir> [args...]`. "
    "Allowed alongside: env-var assignments, real Python flags "
    "(`-O`, `-u`, `-X dev`, ...), and FD redirection (`> log`, `2>&1`). "
    "Not supported: short-circuit flags (`--version`, `-c`, `-m`), "
    "wrappers (`nohup`, `bash -lc`, subshells, `$(…)`), chains "
    "(`&&`, `||`, `;`, `|`), and backgrounding (`&`). Run multiple AR "
    "scripts in separate Bash calls; use the Read tool to inspect "
    "script source. (This is an LLM workflow guardrail, not a Bash "
    "sandbox.)"
)


# === LAYER 3: check_bash + check_edit ==================================

def check_bash(phase: str, command: str) -> tuple:
    """Return (allowed: bool, reason: str) for a Bash command at `phase`.

    Three layers, in order:
      1. `git commit` substring ban (must never run raw, even inside
         a permissive phase).
      2. classify() — pure command-shape decision.
      3. Phase table lookup — (phase, class) → allowed/blocked. The
         subprocess-only AR-script check fires only when classify
         returns AR(name) for one of those names — substring banning
         was retired because it falsely blocked READONLY mentions.

    AR-mention-but-not-canonical is rejected in EVERY phase to keep
    the canonical-form contract crisp; without that, permissive phases
    would accept shapes like `nohup python .autoresearch/scripts/X.py
    &` as 'ad-hoc shell'.
    """
    if "git commit" in command:
        return False, ("manual 'git commit' forbidden — commits are "
                       "produced by pipeline.py via keep_or_discard")

    shape = classify(command)

    if shape.klass == "LIFECYCLE":
        return True, ""

    if shape.klass == "READONLY":
        return True, ""

    if shape.klass == "AR":
        if shape.name in _SUBPROCESS_ONLY_AR_SCRIPTS:
            return False, (f"'{shape.name}' — "
                           f"{_SUBPROCESS_ONLY_AR_SCRIPTS[shape.name]}")
        allowed = _AR_ALLOWED_BY_PHASE.get(phase)
        if allowed is None:
            return False, f"unknown phase {phase!r}"
        if shape.name in allowed:
            return True, ""
        allowed_txt = sorted(allowed) or "(no AR scripts allowed in this phase)"
        return False, (f"phase {phase}: AR script {shape.name!r} not "
                       f"allowed; allowed = {allowed_txt}")

    # OTHER. AR-mention here means a malformed AR shape (chain, wrapper,
    # backgrounded, --version, etc.) — reject everywhere.
    if ".autoresearch/scripts/" in _normalize(command):
        return False, _CANONICAL_FORM_REJECTION

    if _OTHER_ALLOWED_BY_PHASE.get(phase, False):
        return True, ""
    return False, (f"phase {phase}: only AR scripts, lifecycle scripts, "
                   f"and read-only commands are allowed")


_DIAGNOSE_ARTIFACT_RE = re.compile(r"^\.ar_state/diagnose_v\d+\.md$")


def check_edit(phase: str, rel_path: str, editable_files,
               *, diagnose_action: Optional[str] = None) -> tuple:
    """Return (allowed: bool, reason: str) for an Edit/Write on `rel_path`
    (task-dir-relative, forward-slash form) at `phase`.

    Writes under .ar_state/ are restricted to a precise allowlist. Phase,
    progress, history, plan.md, report.md, heartbeat, and markers are all
    machine-maintained — letting Claude Edit them would let the model skip
    phases, rewrite counters, or forge history. Two paths are agent-writable:
      - .ar_state/plan_items.xml: the XML input file /autoresearch hands to
        create_plan.py. In DIAGNOSE this write is gated on
        `diagnose_action` so the agent can't pre-stage stale plan input
        before the subagent has produced (or the cap has retired) the
        diagnosis artifact. Caller hooks are expected to compute the
        action via `diagnose_state(...).action` and pass it; if not
        provided, the gate is open (matches pre-DIAGNOSE behaviour).
      - .ar_state/diagnose_v<N>.md: the DIAGNOSE-phase artifact. The
        ar-diagnosis subagent is the intended writer (per the prompt
        contract), but hook payloads do NOT distinguish main agent from
        subagent — provenance is not enforced. Only the artifact's
        CONTENT (sections + marker) is validated, and only writable while
        phase=DIAGNOSE.

    The FINISH-phase report (.ar_state/report.md) is generated by
    pipeline.py via report.py — Claude does not write it.
    """
    if rel_path.startswith(".ar_state/"):
        if rel_path == f".ar_state/{PLAN_ITEMS_FILE}":
            # In DIAGNOSE the write is gated on the three-state action
            # (see validators.diagnose_state). NEED_DIAGNOSIS = subagent
            # hasn't validated yet; allowing plan_items.xml here lets the
            # agent skip the diagnosis. READY / MANUAL_FALLBACK both
            # legitimately want the plan input written.
            if (phase == DIAGNOSE
                    and diagnose_action == "NEED_DIAGNOSIS"):
                return False, (
                    f".ar_state/{PLAN_ITEMS_FILE} is locked while DIAGNOSE "
                    f"awaits a valid diagnosis artifact. Issue Task with "
                    f"subagent_type='ar-diagnosis' first; once the artifact "
                    f"validates (or the attempt cap relaxes the gate) "
                    f"plan_items.xml becomes writable."
                )
            return True, ""
        if _DIAGNOSE_ARTIFACT_RE.match(rel_path):
            if phase == DIAGNOSE:
                return True, ""
            return False, (
                f"{rel_path!r} is the DIAGNOSE artifact and is only "
                f"writable while phase=DIAGNOSE."
            )
        if rel_path == f".ar_state/{PLAN_FILE}":
            return False, (
                f"plan.md is machine-generated — never hand-edit it. Write "
                f"your <items>...</items> XML to .ar_state/{PLAN_ITEMS_FILE} "
                f"with the Write tool, then run "
                f"`python .autoresearch/scripts/create_plan.py \"<task_dir>\"`."
            )
        return False, (
            f"{rel_path!r} is machine-maintained state. Only "
            f".ar_state/{PLAN_ITEMS_FILE} (plan input) and "
            f".ar_state/diagnose_v<N>.md (DIAGNOSE artifact) are writable "
            f"under .ar_state/; everything else (including the FINISH "
            f"report.md) is owned by hooks and scripts."
        )

    allowed_classes = _EDIT_RULES.get(phase, set())
    if "ref" in allowed_classes and rel_path == "reference.py":
        return True, ""
    if "editable" in allowed_classes and rel_path in set(editable_files or ()):
        return True, ""

    return False, f"phase {phase} does not allow writing {rel_path!r}"


# === Phase transitions =================================================

def compute_next_phase(task_dir: str) -> str:
    """After a pipeline round finishes, mechanically determine the next phase.

    `eval_rounds >= max_rounds` is the only legitimate FINISH trigger; the
    `not progress` branch is an error fallback for unrecoverable state.
    """
    progress = load_progress(task_dir)
    if not progress:
        return FINISH  # error fallback: corrupt/missing progress.json

    consecutive_failures = progress.get("consecutive_failures", 0)
    eval_rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", 999)

    if eval_rounds >= max_rounds:
        return FINISH
    if consecutive_failures >= 3:
        return DIAGNOSE
    if has_pending_items(task_dir):
        return EDIT
    return REPLAN


def compute_resume_phase(task_dir: str) -> str:
    """Determine phase for resuming after interruption."""
    progress = load_progress(task_dir)
    if not progress:
        return BASELINE

    status = progress.get("status", "no_plan")
    eval_rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", 999)

    if eval_rounds >= max_rounds:
        return FINISH

    # Baseline didn't settle cleanly → demote to GENERATE_KERNEL so Edit on
    # kernel.py is permitted again (BASELINE phase blocks editable_files
    # writes). Mirrors hook_post_bash's live-session demotion: both
    # seed_metric=None (no timing) and baseline_correctness=False (wrong
    # output) count as failure. A seed that profiled but didn't verify is
    # still a broken seed — letting resume enter PLAN would build a plan
    # against a reference that the kernel doesn't actually match.
    if (progress.get("seed_metric") is None
            or progress.get("baseline_correctness") is False):
        return GENERATE_KERNEL

    if not os.path.exists(plan_path(task_dir)) or status == "no_plan":
        return PLAN

    if get_active_item(task_dir) is not None or has_pending_items(task_dir):
        return EDIT
    return REPLAN
