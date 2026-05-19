"""Phase-specific guidance - what the LLM should do next.

`get_guidance(task_dir)` is the only public API; it reads phase + progress
+ task config + plan, then returns the `[AR Phase: …]` message that hooks
inject into Claude's context after every state-changing event.

The XML schema example for plan creation (`_PLAN_XML_EXAMPLE`) and the
field-rules tail (`_PLAN_FIELD_RULES`) live here - they're prompt content
shared between PLAN, DIAGNOSE, and REPLAN guidance.
"""
import json
import os
import sys
from typing import Optional

from .state_store import (
    INIT, GENERATE_REF, GENERATE_KERNEL, BASELINE, PLAN, EDIT,
    DIAGNOSE, REPLAN, FINISH,
    PLAN_ITEMS_FILE, DIAGNOSE_ATTEMPTS_CAP,
    diagnose_artifact_path, diagnose_marker,
    history_path, load_progress, read_phase, state_path,
    _PROJECT_ROOT,
)
from .validators import (
    get_active_item, diagnose_state,
    DIAGNOSE_READY, DIAGNOSE_MANUAL_FALLBACK,
)


def _format_fail_record(rec: dict,
                        progress: Optional[dict] = None) -> str:
    """Compact per-FAIL block for the DIAGNOSE subagent prompt.

    Surfaces what kept-or-discard now persists per FAIL:
    `failure_signals` (kind + extracted params + hint) and a trimmed
    `raw_output_tail` when the pattern matchers found nothing. The
    earlier prompt only listed `R<n>: <description>`, so the subagent
    had to Read history.jsonl to learn what actually broke - which
    didn't help anyway because the structured signals weren't in
    history before this change.

    `progress` is passed through to `_failed_shapes_block` so per-shape
    correctness failures (eval_client's `correctness_failed_cases` etc.)
    get rendered with their describe_case() strings inline.
    """
    rnd = rec.get("round", "?")
    desc = (rec.get("description") or "")[:80]
    out = [f"  R{rnd}: {desc}"]

    sig = rec.get("failure_signals") or {}
    signals = sig.get("signals") or []
    primary = sig.get("primary")
    python_error = sig.get("python_error")

    for s in signals[:2]:  # at most two distinct kinds, the rest is noise
        kind = s.get("kind", "?")
        params = ", ".join(
            f"{k}={v}"
            for k, v in s.items()
            if k not in ("kind", "excerpt", "hint") and v is not None
        )
        marker = f"      → {kind}"
        if params:
            marker += f"  [{params}]"
        out.append(marker)
        if s.get("hint"):
            out.append(f"        hint: {s['hint']}")
    if python_error:
        out.append(f"      python_error: {python_error[:200]}")
    if not signals and not python_error:
        # Legacy FAIL row (recorded before failure_signals were
        # persisted) or one whose log matched no pattern. Surface the
        # raw error message and a trimmed tail if either is available.
        err = (rec.get("error") or "").strip()
        if err and "verify failed (kernel broken)" not in err:
            out.append(f"      error: {err[:160]}")
        tail = (rec.get("raw_output_tail") or "").strip()
        if tail:
            # Strip noisy whitespace-only lines from the tail head, keep
            # the last couple of lines - that's where Python tracebacks
            # and ACL runtime errors land.
            keep = [l for l in tail.splitlines()[-4:] if l.strip()]
            if keep:
                out.append("      tail:")
                for line in keep:
                    out.append(f"        {line[:160]}")

    # Per-shape correctness failure detail. Empty for compile/profile-style
    # failures (no per-shape signal) and for single-shape ops; see
    # _failed_shapes_block for the gating.
    fs = _failed_shapes_block(rec.get("metrics"), progress)
    if fs:
        out.append(fs)

    return "\n".join(out)


# Shared plan-item scaffolding shown in PLAN / DIAGNOSE / REPLAN guidance.
# The example is deliberately a short SENTENCE (not a snake_case identifier) -
# dashboards surface `desc` directly in the history and plan tables, so
# "Fuse SwiGLU into the matmul epilogue to avoid a second launch" reads far
# better than "fuse_swiglu_epilogue". create_plan.py enforces the prose form.
#
# XML is the required format - tag-delimited text is structurally harder for
# LLMs to hallucinate than JSON (no stray commas / quote escaping / brace
# balance to track).
# Inline XML comments inside the example double as schema reminders. The
# model is far more likely to obey rules embedded in the structure it's
# mimicking than rules sitting in a separate paragraph it has to remember
# to apply. Anti-drift hints are placed where each drift tends to land:
# attributes on <item>, extra child elements, missing fields.
_PLAN_XML_EXAMPLE = (
    '<items>'
    '<!-- Provide >= 3 <item> elements. No attributes or extra tags on <items>. -->'
    '<item>'
    '<!-- An <item> has NO attributes and EXACTLY two child elements: '
    '<desc> and <rationale>. Do NOT add <id>, <pid>, <keywords>, '
    '<priority>, <reactivate_pid>, or id="..." / pid="..." attributes. '
    'Pids are auto-assigned by create_plan.py from a monotonic counter; '
    'the model never supplies them - supplying one is rejected. -->'
    '<desc>Fuse SwiGLU into the matmul epilogue to avoid a second launch</desc>'
    '<!-- <desc>: short SENTENCE (>=12 chars, has spaces). Not a '
    'snake_case label; the dashboard shows desc verbatim. -->'
    '<rationale>Separate SwiGLU kernel re-reads the matmul output from DRAM; '
    'fusing it into the epilogue cuts one round-trip and a launch.</rationale>'
    '<!-- <rationale>: 30-400 chars, explains WHY this should help. -->'
    '</item>'
    '<!-- Repeat <item> blocks for >= 3 total items. Same two-child rule '
    'each time; nothing per-item is optional and nothing extra is allowed. -->'
    '</items>'
)
_PLAN_FIELD_RULES = (
    "Schema reminders are embedded as <!-- comments --> inside the XML "
    "example above; read them - each comment marks the spot where a "
    "field rule applies. Beyond schema: escape '&', '<', '>' in text as "
    "'&amp;', '&lt;', '&gt;' (or wrap the offending field in "
    "<![CDATA[...]]>)."
)


def _create_plan_instruction(task_dir: str) -> str:
    """Common 'how to invoke create_plan.py' block used by PLAN, DIAGNOSE,
    and REPLAN guidance. Emits the canonical two-step flow:

      1. Write XML to the FIXED path .ar_state/plan_items.xml.
      2. Run create_plan.py with just <task_dir> - it reads from that path.

    The fixed path eliminates the LLM-drift class where the model wrote
    to one path and then passed a different `@<path>` to create_plan
    (most often a hallucinated /tmp/... or a typoed task subdir).
    """
    xml_path = state_path(task_dir, PLAN_ITEMS_FILE)
    return (
        f"To create the plan, do EXACTLY these two steps:\n"
        f"  1. Use the Write tool to write your <items>...</items> XML to:\n"
        f"       {xml_path}\n"
        f"     (Path is fixed - do NOT invent a different path, do NOT use "
        f"/tmp/, do NOT pass it as a CLI arg later. The Write tool is the "
        f"only thing that touches this path.)\n"
        f"  2. Run:\n"
        f"       python .autoresearch/scripts/create_plan.py \"{task_dir}\"\n"
        f"     (No second argument. The script reads .ar_state/{PLAN_ITEMS_FILE} "
        f"automatically. Adding `@/some/path` reintroduces the drift this "
        f"two-step form exists to prevent.)\n"
        f"\n"
        f"XML schema (write this exact shape to the file in step 1):\n"
        f"{_PLAN_XML_EXAMPLE}\n"
        f"{_PLAN_FIELD_RULES}\n"
    )


def _load_config_safe(task_dir: str):
    """Load TaskConfig, return None on any failure.

    task_config lives in scripts/ root (one level up from this package);
    insert the parent dir into sys.path so the import resolves no matter
    who's importing us.
    """
    try:
        _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from task_config import load_task_config
        return load_task_config(task_dir)
    except Exception:
        return None


# Single skills reference source: `skills/triton/latency-optimizer/
# references/*.md` (flat perf-tuning markdown — autotune, scalar_to_
# vector, tiling_optimization, dimension-merge, vector_core_partition,
# libdevice-usage, load-order, loop-invariant-hoisting, pass-merge,
# discrete_memory_access, constexpr_parameters, checklist,
# avoid_scalar_lowering). Returns "" when the directory is missing so
# the prompt doesn't dispatch the agent to a dead Glob.
_LATENCY_REFS_DIR = "triton/latency-optimizer/references"


def _skills_hint() -> str:
    """Recommend reading the latency-optimizer reference docs when
    authoring plan items.

    Used by PLAN and REPLAN (parent-voice — the parent agent reads
    skills directly and writes the plan). DIAGNOSE has its own inline
    skills section because the subagent's framing differs: it's
    diagnosing failures, not opening a plan, and the prompt wording
    reflects that.
    """
    if not os.path.isdir(os.path.join(_PROJECT_ROOT, "skills", _LATENCY_REFS_DIR)):
        return ""
    return (
        f"\nSkills: Glob skills/{_LATENCY_REFS_DIR}/*.md (flat perf-tuning "
        f"refs — autotune, scalar_to_vector, tiling_optimization, "
        f"dimension-merge, vector_core_partition, etc.; pick by filename "
        f"topic), Read 1-3 most relevant to a candidate plan-item direction. "
        f"Citing the filename in the rationale is recommended for "
        f"traceability but not enforced."
    )


def _multi_shape_plan_note(progress: Optional[dict],
                           task_dir: Optional[str] = None) -> str:
    """One-line note for the PLAN phase: say the op is multi-shape and point
    at the actual file(s) holding the shape spec. Deliberately does NOT
    list individual shapes - plan items are coarse-grained decisions, a
    30-line case dump in the planning prompt makes the agent over-engineer
    for shape generality at the expense of writing good plan items.

    NPUKernelBench-style refs read shapes from a sidecar JSON (the ref's
    `get_input_groups()` opens a same-directory `<basename>.json`). When
    that JSON is present in `task_dir`, this note names it explicitly -
    pointing at reference.py alone is not enough because the .py file is
    just a loader; the actual shape list lives in the JSON.

    Returns "" for single-shape ops (progress.num_cases <= 1) and when
    progress.json hasn't been written yet (pre-BASELINE).
    """
    if not progress:
        return ""
    n = progress.get("num_cases")
    if not isinstance(n, int) or n <= 1:
        return ""

    # Locate sidecar JSONs that scaffold copied into task_dir at startup.
    # Skip task.yaml / *.lock / dotfiles and only surface JSONs at the
    # task_dir root (sidecars live there per scaffold.py:106-116).
    sidecar_names: list[str] = []
    if task_dir and os.path.isdir(task_dir):
        try:
            for fname in sorted(os.listdir(task_dir)):
                if not fname.endswith(".json"):
                    continue
                if fname.startswith("."):
                    continue
                if os.path.isfile(os.path.join(task_dir, fname)):
                    sidecar_names.append(fname)
        except OSError:
            pass

    # Fully qualify sidecar paths with task_dir so the agent can Read
    # them without first having to glob / guess where they live.
    if sidecar_names:
        full_paths = [f"{task_dir}/{name}" for name in sidecar_names]
        if len(full_paths) == 1:
            where = f"shape list: {full_paths[0]}"
        else:
            where = "shape lists:\n  - " + "\n  - ".join(full_paths)
    else:
        # Inline get_input_groups() - shapes are constructed in reference.py
        # itself, not a sidecar. Point at the loader function in the ref.
        where = (
            f"shape list: {task_dir}/reference.py "
            f"(in the get_input_groups() body)"
        )

    return (
        f"Note: multi-shape op - reference exposes {n} input groups "
        f"via get_input_groups(). {where}\n"
        f"Plan items must hold across all shapes; rely on shape-aware "
        f"logic (read shape at runtime, dispatch on dtype/rank, adapt "
        f"tile size) rather than constants pinned to one shape."
    )


def _last_failure_metrics(task_dir: str) -> Optional[dict]:
    """Return the metrics dict of the most recent FAIL/SEED record in
    history.jsonl whose `correctness` is False. Used by GENERATE_KERNEL
    retry guidance (the seed kernel's BASELINE just failed correctness;
    the round-0 SEED record carries the per-shape detail) and could be
    reused elsewhere. Returns None when no failed record exists or
    history is missing.
    """
    hpath = history_path(task_dir)
    if not os.path.exists(hpath):
        return None
    last = None
    try:
        with open(hpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("correctness") is False:
                    last = rec
    except OSError:
        return None
    if not last:
        return None
    metrics = last.get("metrics")
    return metrics if isinstance(metrics, dict) else None


def _failed_shapes_block(metrics: Optional[dict],
                         progress: Optional[dict],
                         *, max_listed: int = 5) -> str:
    """Render the per-shape failure detail used in DIAGNOSE's FAIL records.

    Pulls from the metrics block of a single FAIL history record (which
    eval_client populates from the verify subprocess's verify_json):
      - correctness_failed_cases: list of failing case indices
      - correctness_total_cases:  total case count at FAIL time
      - correctness_worst_case:   index with the largest max_abs_diff
      - correctness_worst_max_abs: that diff

    Resolves indices to describe_case() strings via progress.per_shape_descs
    so the agent sees both the index and the actual shape it fouled up.
    Returns "" when none of those fields are present (e.g. compile-error
    failure with no per-case detail, or single-shape op where the failure
    block is redundant).
    """
    if not metrics:
        return ""
    failed = metrics.get("correctness_failed_cases")
    total = metrics.get("correctness_total_cases")
    if not isinstance(failed, list) or not failed:
        return ""
    if not isinstance(total, int) or total <= 1:
        # Single-shape FAIL: the message that says "kernel broken"
        # already conveys this; an extra "1/1 shapes failed" block is noise.
        return ""

    descs = (progress or {}).get("per_shape_descs") or []
    parts = [f"      failed shapes: {len(failed)}/{total}"]
    for idx in failed[:max_listed]:
        if isinstance(idx, int) and 0 <= idx < len(descs):
            parts.append(f"        [{idx}] {descs[idx]}")
        else:
            parts.append(f"        [{idx}] (desc unavailable)")
    if len(failed) > max_listed:
        parts.append(f"        ... ({len(failed)} failures total)")
    worst_idx = metrics.get("correctness_worst_case")
    worst_max = metrics.get("correctness_worst_max_abs")
    if isinstance(worst_idx, int) and isinstance(worst_max, (int, float)):
        parts.append(
            f"      worst: case [{worst_idx}] max_abs={worst_max:.3e}"
        )
    return "\n".join(parts)


def _diagnose_plan_next_step(task_dir: str, *,
                             artifact_path: Optional[str] = None,
                             fallback: bool = False) -> str:
    """Guidance text for the post-DIAGNOSE create_plan step.

    Two callers in get_guidance: action == DIAGNOSE_READY passes the
    artifact path; action == DIAGNOSE_MANUAL_FALLBACK passes nothing
    (artifact_path is unused in fallback mode - the diagnosis context
    is history.jsonl + plan.md).
    """
    if fallback:
        header = "[AR Phase: DIAGNOSE - manual planning fallback]"
        source = "history.jsonl + plan.md (subagent route exhausted)"
    else:
        header = "[AR Phase: DIAGNOSE - diagnosis ready]"
        source = artifact_path or "(diagnosis artifact)"
    return (
        f"{header}\n"
        f"Create a NEW plan with >= 3 diverse items using {source}.\n"
        f"Max 1 parameter-tuning item; the rest must be structural changes "
        f"(algorithmic / fusion / memory layout / data movement).\n\n"
        f"{_create_plan_instruction(task_dir)}"
        f"\nAfter create_plan.py validates, the hook advances phase to EDIT "
        f"and emits the TodoWrite payload."
    )


def get_guidance(task_dir: str) -> str:
    """Return a context-aware instruction for Claude based on current phase.

    Reads task.yaml to inject dynamic info (editable files, skills path)
    so the .md slash command doesn't need to hardcode anything.
    """
    phase = read_phase(task_dir)
    active = get_active_item(task_dir)
    progress = load_progress(task_dir)
    config = _load_config_safe(task_dir)

    # Extract config fields
    editable = config.editable_files if config else []
    primary_metric = config.primary_metric if config else "score"

    if phase == INIT:
        return f"[AR Phase: INIT] Run: export AR_TASK_DIR=\"{task_dir}\""

    if phase == GENERATE_REF:
        description = config.description if config else "(no description)"
        return (f"[AR Phase: GENERATE_REF] Write reference.py for: {description}\n"
                f"Write to: {task_dir}/reference.py\n"
                f"Must contain: class Model(nn.Module) with forward(), "
                f"get_init_inputs(), and one of get_inputs() (single-shape) "
                f"or get_input_groups() (multi-shape, returns List[List]).\n"
                f"This is the BASELINE implementation - no optimization, just correct.")

    if phase == GENERATE_KERNEL:
        # Retry detection: progress.json only exists once _baseline_init.py
        # has run, and hook_post_bash only demotes back to GENERATE_KERNEL
        # when seed_metric is None (compile/profile failed) or
        # baseline_correctness is False (numerical mismatch). On the first
        # entry progress is None, so this is a clean signal.
        is_retry = bool(progress) and (
            progress.get("seed_metric") is None
            or progress.get("baseline_correctness") is False
        )
        if is_retry:
            retry_reason = (
                "seed kernel produced no timing (compile/profile failed)"
                if progress.get("seed_metric") is None
                else "seed kernel ran but failed correctness vs reference"
            )
            header = f"[AR Phase: GENERATE_KERNEL - retry, prior seed failed: {retry_reason}]"
            verb = "Generate a corrected"
        else:
            header = "[AR Phase: GENERATE_KERNEL]"
            verb = "Generate an initial"

        description = config.description if config else "(no description)"
        target_file = editable[0] if editable else "kernel.py"
        editable_line = (
            f"Editable files: {', '.join(editable)}\n" if editable else ""
        )
        constraints_part = ""
        if config and getattr(config, "constraints", None):
            # constraints is {metric: (op_str, threshold)} - render compactly
            constraint_strs = [
                f"{m}{op}{thr}" for m, (op, thr) in config.constraints.items()
            ]
            constraints_part = f" | constraints: {', '.join(constraint_strs)}"

        retry_block = ""
        if is_retry:
            retry_block = (
                "\nThis is a retry. baseline.py just printed structured failure "
                "signals above (UB overflow / aivec trap / OOM / correctness "
                f"mismatch / ...). Read that output, then read the current "
                f"{task_dir}/{target_file} to see what failed. Use the skills "
                "Glob above to find a SKILL.md whose description matches the "
                "failure kind before rewriting. Do NOT rewrite from scratch "
                "unless the failure is structural - incremental fixes converge "
                "faster.\n"
            )

        # Failed-shape detail: only on retry, only when BASELINE failed
        # because some shapes mismatched (not when seed_metric is None,
        # i.e. compile/profile failed - that has no per-shape signal).
        # Read the round-0 SEED record from history; eval_client surfaced
        # `correctness_failed_cases` etc. into its metrics dict.
        failed_shapes_section = ""
        if is_retry and progress.get("baseline_correctness") is False:
            fail_metrics = _last_failure_metrics(task_dir)
            block = _failed_shapes_block(fail_metrics, progress)
            if block:
                failed_shapes_section = (
                    f"\n\nThe BASELINE correctness failure had per-shape "
                    f"detail (from the round-0 SEED record):\n{block}\n"
                    f"Fix the kernel for those shape(s) specifically; "
                    f"common pitfalls: block size hardcoded for a "
                    f"different shape, dtype dispatch missing, broadcast "
                    f"assumed but not present.\n"
                )

        return (
            f"{header} {verb} kernel from reference.\n"
            f"Task: {description}\n"
            f"Primary metric: {primary_metric}{constraints_part}\n"
            f"{editable_line}"
            f"\n"
            f"Read {task_dir}/reference.py, then write to {task_dir}/{target_file}.\n"
            f"Must contain: class ModelNew (can inherit from Model)."
            f"{_skills_hint()}\n"
            f"{retry_block}"
            f"{failed_shapes_section}"
            f"\n"
            f"Start simple - the autoresearch loop will iterate from here."
        )

    if phase == BASELINE:
        return (f"[AR Phase: BASELINE] Run: "
                f"python .autoresearch/scripts/baseline.py \"{task_dir}\"")

    if phase == PLAN:
        metric_hint = ""
        if progress:
            baseline = progress.get("baseline_metric")
            if baseline is not None:
                metric_hint = f" Baseline {primary_metric}: {baseline}."
        # PLAN gets a short multi-shape note (count + pointer to the
        # sidecar JSON / ref). The detailed case list stays out-of-band in
        # the actual file - clogging the planning context with 30 lines of
        # `case i: tensor[...]` makes the agent over-engineer for shape
        # generality rather than write good plan items. Returns "" for
        # single-shape ops, so most tasks see no extra prompt content.
        plan_note = _multi_shape_plan_note(progress, task_dir=task_dir)
        plan_note_section = f"\n\n{plan_note}" if plan_note else ""

        return (f"[AR Phase: PLAN] "
                f"Read task.yaml, editable files ({editable}), and reference.py.{_skills_hint()}{metric_hint}{plan_note_section}\n"
                f"\n"
                f"{_create_plan_instruction(task_dir)}"
                f"\n"
                f"The script writes plan.md in the correct format. Hook validates and advances to EDIT.\n"
                f"(After validation the hook emits a TodoWrite payload - call "
                f"it verbatim; do not pre-emptively craft one here.)")

    if phase == EDIT:
        desc = active["description"] if active else "(no active item)"
        item_id = active["id"] if active else "?"
        files_hint = f" (files: {', '.join(editable)})" if editable else ""
        # No multi-shape note here. EDIT prompts already carry the active
        # plan item's description and the agent has the file list; piling
        # the full shape spec on top every EDIT phase (10-20+ times per
        # task) bloats context and biases edits toward over-generalisation.
        # If a multi-shape regression surfaces, it lands as a FAIL and the
        # next DIAGNOSE / GENERATE_KERNEL retry will surface the offending
        # shapes via _failed_shapes_block.
        return (f"[AR Phase: EDIT] ACTIVE item: **{item_id}** - {desc}\n"
                f"{files_hint}\n"
                f"CRITICAL: Implement ONLY {item_id}'s idea. Do NOT implement other plan items.\n"
                f"The pipeline will settle {item_id} with this round's metric.\n"
                f"Make your edit(s), then: python .autoresearch/scripts/pipeline.py \"{task_dir}\"\n"
                f"(TodoWrite payloads are delivered by the hook after each "
                f"settle / create_plan - call them verbatim when emitted; "
                f"do not synthesize TodoWrite calls from this hint.)")

    if phase == DIAGNOSE:
        # Pre-bake the recent-rounds summary INTO the subagent prompt so the
        # subagent has it without spending a tool call re-reading
        # history.jsonl. The full file stays in the read list for deeper
        # digs (full traces / older rounds).
        # Single pass through history.jsonl: build both the high-level
        # rhythm (last 5 records, any decision) and the FAIL detail block
        # (last 3 FAILs with structured failure_signals, courtesy of
        # keep_or_discard which now persists them per FAIL).
        hpath = history_path(task_dir)
        recent_summary = ""
        fail_details = ""
        if os.path.exists(hpath):
            all_recs = []
            with open(hpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_recs.append(json.loads(line))
                    except Exception:
                        continue
            for rec in all_recs[-5:]:
                _r = rec.get("round")
                _r = "?" if _r is None else _r
                recent_summary += f"  R{_r}: {rec.get('decision','?')} - {rec.get('description','')[:60]}\n"
            last_3_fails = [
                r for r in all_recs
                if r.get("decision") == "FAIL" and r.get("round") is not None
            ][-3:]
            if last_3_fails:
                fail_details = "\n".join(
                    _format_fail_record(r, progress) for r in last_3_fails
                ) + "\n"

        # Compact metric snapshot - saves the subagent from reading
        # history.jsonl just to answer "how big a delta do we need?".
        metric_line = ""
        if progress:
            seed = progress.get("seed_metric")
            base = progress.get("baseline_metric")
            best = progress.get("best_metric")
            if any(v is not None for v in (seed, base, best)):
                metric_line = (
                    f"\nMetrics ({primary_metric}): "
                    f"seed={seed} | ref_baseline={base} | current_best={best}"
                )
        # Single source of plan_version + per-pv attempt counter (also
        # validates the artifact, but the result is unused here - accepting
        # the small extra read so all callers go through the same helper).
        ds = diagnose_state(task_dir, progress=progress) if progress else None
        plan_version = ds.plan_version if ds else 0
        attempts = ds.attempts if ds else 0
        artifact_path = diagnose_artifact_path(task_dir, plan_version)
        if ds and ds.action == DIAGNOSE_READY:
            return _diagnose_plan_next_step(
                task_dir, artifact_path=artifact_path)
        if ds and ds.action == DIAGNOSE_MANUAL_FALLBACK:
            return _diagnose_plan_next_step(task_dir, fallback=True)

        arch = (config.arch if config and config.arch else "<unknown>")
        editable_paths = "\n".join(
            f"  - {task_dir}/{name}" for name in (editable or ["kernel.py"])
        )
        # Skills section is conditional on the references dir being present.
        # Without the dir-existence check we'd hand the agent a Glob
        # pattern that returns zero matches, and they'd silently skip
        # the skill-reading step.
        skills_present = os.path.isdir(
            os.path.join(_PROJECT_ROOT, "skills", _LATENCY_REFS_DIR))
        if skills_present:
            skills_block = (
                f"Read curated perf-tuning references (use them to ground "
                f"fix directions in known-good patterns for this "
                f"hardware):\n"
                f"  - Glob skills/{_LATENCY_REFS_DIR}/*.md (flat .md, no "
                f"frontmatter; pick by filename topic — autotune, "
                f"scalar_to_vector, tiling_optimization, dimension-merge, "
                f"etc.) and Read what matches the fix direction.\n"
                f"  - Cite filename in the rationale of items you propose.\n\n"
            )
            scope_constraint = (
                f"  - Glob / Grep ONLY under skills/{_LATENCY_REFS_DIR}/. "
                f"The 4 task files plus that references subtree are the "
                f"entire scope.\n"
            )
            cite_clause = " Cite reference filenames where relevant."
        else:
            skills_block = ""
            scope_constraint = (
                "  - Do NOT Glob / Grep the wider codebase. The 4 task "
                "files are the entire scope.\n"
            )
            cite_clause = ""

        # Artifact contract - the host validates these literals after the
        # Task call returns. See validators.validate_diagnose.
        marker = diagnose_marker(plan_version)

        # Pre-baked subagent prompt. Parent passes this verbatim to the Agent
        # tool so the subagent doesn't improvise (an earlier open-ended brief
        # sent it grepping git log for 100+ tool calls before timing out).
        #
        # Two-section history view: the recent-5 block gives rhythm
        # (KEEP/DISCARD/FAIL pattern), the FAIL-detail block gives the
        # structured signals (UB overflow / aivec trap / OOM / correctness
        # mismatch + hint) that root-cause analysis needs. Before this,
        # the subagent only saw round + 60-char description per FAIL and
        # had to guess which Ascend constraint was violated.
        fail_details_block = (
            f"Last 3 FAILs (use these as the primary evidence):\n"
            f"{fail_details}\n"
            if fail_details
            else "Last 3 FAILs: (none yet - use history.jsonl if needed)\n\n"
        )
        subagent_prompt = (
            f"Diagnose why the current optimization rounds are failing, then "
            f"Write a structured report to a fixed path.\n\n"
            f"Target: arch={arch}{metric_line}\n"
            f"plan_version={plan_version}\n\n"
            f"Recent rounds (last 5 from history.jsonl):\n"
            f"{recent_summary or '  (none settled yet)'}\n"
            f"{fail_details_block}"
            f"Read these task files for context:\n"
            f"  - {task_dir}/reference.py\n"
            f"{editable_paths}\n"
            f"  - {task_dir}/.ar_state/plan.md\n"
            f"  - {task_dir}/.ar_state/history.jsonl (focus on the last "
            f"~10 rounds; older entries are usually stale)\n\n"
            f"{skills_block}"
            f"Hard constraints:\n"
            f"  - Do NOT search git history (`git log` / `git show` / "
            f"`git grep`) - per-round commits carry no keyword signal and "
            f"burn tool calls.\n"
            f"{scope_constraint}"
            f"  - Stop after at most 12 tool uses.\n"
            f"  - Write tool may ONLY target the artifact path below. Do "
            f"NOT Write kernel.py, plan.md, or anywhere else.\n\n"
            f"REQUIRED OUTPUT - your final action MUST be a Write call to "
            f"this exact path:\n"
            f"  {artifact_path}\n\n"
            f"The file body must contain ALL of:\n"
            f"  - heading section 'Root cause' (one paragraph grounded in "
            f"the FAIL summary / history)\n"
            f"  - heading section 'Fix directions' (≤3 STRUCTURALLY "
            f"different approaches: algorithmic / fusion / memory layout "
            f"/ data movement; NOT parameter tuning.{cite_clause})\n"
            f"  - heading section 'What to avoid' (≤3 patterns to NOT "
            f"repeat)\n"
            f"  - the magic marker line on its own line at the end:\n"
            f"      {marker}\n"
            f"Total ≤ 300 words across the three sections. The host "
            f"validates path + marker + the three section names after "
            f"this Task call returns; missing any element will force a "
            f"retry."
        )
        retry_note = ""
        if attempts > 0:
            retry_note = (
                f"\nThis is DIAGNOSE attempt {attempts + 1}/"
                f"{DIAGNOSE_ATTEMPTS_CAP}. The previous artifact was "
                f"missing or malformed - re-issue Task and ensure the "
                f"subagent ends its work with a Write of the marker line."
            )
        return (f"[AR Phase: DIAGNOSE] consecutive_failures >= 3.\n"
                f"Required action: call the "
                f"Task tool with subagent_type='ar-diagnosis' and this "
                f"EXACT prompt. Do not paraphrase. Do not add or remove "
                f"constraints. Do not Edit, Write, or Bash before this "
                f"Task call.\n"
                f"---BEGIN SUBAGENT PROMPT---\n"
                f"{subagent_prompt}\n"
                f"---END SUBAGENT PROMPT---\n"
                f"Artifact contract: the host gates plan creation on a valid "
                f"{os.path.basename(artifact_path)} (path + marker + 3 "
                f"sections). Up to "
                f"{DIAGNOSE_ATTEMPTS_CAP} Task attempts are allowed; after "
                f"that the gate is relaxed and you must write "
                f"plan_items.xml directly (manual-planning fallback) "
                f"before running create_plan.py - the DIAGNOSE phase "
                f"still requires a new plan, just without subagent help.{retry_note}")

    if phase == REPLAN:
        remaining = "?"
        plan_ver = 0
        if progress:
            remaining = str(progress.get("max_rounds", 0) - progress.get("eval_rounds", 0))
            plan_ver = progress.get("plan_version", 0)
        retry_hint = ""
        if plan_ver >= 2:
            retry_hint = (
                f"\nNote: plan_version is already {plan_ver}. Before "
                "inventing entirely new ideas, scan history.jsonl for "
                "DISCARD items whose metric was close to best (within "
                "~20%) - those ideas may compose differently now that "
                "the kernel's structural baseline has shifted. To revisit "
                "one, just include it as a new item with a fresh pid "
                "(reference the prior pid in <desc> for audit context)."
            )
        return (f"[AR Phase: REPLAN] All items settled. Budget: {remaining} rounds left. "
                f"Read .ar_state/history.jsonl. Analyze what worked/failed.{_skills_hint()}\n"
                f"\n"
                f"{_create_plan_instruction(task_dir)}"
                f"{retry_hint}")

    if phase == FINISH:
        best = progress.get("best_metric") if progress else "?"
        baseline = progress.get("baseline_metric") if progress else "?"
        return (f"[AR Phase: FINISH] Done. Best {primary_metric}: {best} "
                f"(baseline: {baseline}). Report auto-generated at "
                f".ar_state/report.md. Summarize for user; do not write any "
                f"files.")

    return f"[AR Phase: {phase}] Unknown phase."
