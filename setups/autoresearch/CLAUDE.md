# Claude AutoResearch

An iterative optimization framework powered by Claude Code: plan → edit →
eval → keep/discard, looping against a measurable metric. Standalone, no
external deps beyond Python + PyYAML.

## Quick Start

```bash
# 1. Open this project in Claude Code
cd claude-autoresearch
claude

# 2. Drop sources into workspace/<op_name>_ref.py (and optional _kernel.py),
#    then start a task. --devices is required.
/autoresearch --ref workspace/<op_name>_ref.py --op-name <op_name> --devices 0

# 3. Resume later
/autoresearch --resume

# 4. Monitor in a separate terminal
python .autoresearch/scripts/dashboard.py <task_dir> --watch
```

`/autoresearch` is the only slash command — full operational details in
[.claude/commands/autoresearch.md](.claude/commands/autoresearch.md).

For unattended long runs, wrap in self-paced loop: `/loop /autoresearch --resume`.

## Skills Library

Single source: `skills/triton/latency-optimizer/references/*.md` —
flat perf-tuning markdown (no frontmatter); pick by filename topic
(autotune, scalar_to_vector, tiling_optimization, dimension-merge,
vector_core_partition, libdevice-usage, load-order,
loop-invariant-hoisting, pass-merge, discrete_memory_access,
constexpr_parameters, checklist, avoid_scalar_lowering).

During PLAN, Glob this directory and Read 1-3 most relevant; cite
filename in plan rationales.

## Invariants (hook-driven flow)

Hooks emit `[AR Phase: ...]` messages on stderr after every state-changing
event. Follow the latest one. Don't try to fetch guidance manually —
`phase_machine` is a Python package used by hooks, not a CLI;
`hook_guard_bash` rejects direct invocation.

The following invariants are non-negotiable:

1. **`.ar_state/plan.md` is the source of truth.** Only `create_plan.py` /
   `settle.py` / `pipeline.py` write it. Never hand-edit. TodoWrite is a
   UI mirror, not a substitute.
2. **Plan IDs are globally monotonic.** `p1, p2, ...` from
   `progress.json.next_pid`. Never reuse, never skip.
3. **Every `pN` either settles (KEEP / DISCARD / FAIL in `history.jsonl`)
   or is dropped at a REPLAN/DIAGNOSE boundary.** `create_plan.py` does
   not synthesize DISCARD rows for superseded items — they're silently
   dropped; the pid counter still advances.
4. **Phase transitions are hook-controlled.** Never write
   `.ar_state/.phase` manually. Wait for the hook's guidance.
5. **Editable files are scoped by `task.yaml.editable_files`.** Editing
   anything else is rejected by `hook_guard_edit.py`.
6. **After a session break, resume with `/autoresearch --resume`.** Do
   not patch state files to recover.
7. **`create_plan.py` rejects mean the plan has a real problem**
   (diversity, repeated failure keywords, short rationale). Read stderr
   and rewrite — don't retry the same XML payload.
8. **TodoWrite sync is mandatory.** When a hook emits `additionalContext`
   with a TodoWrite payload, call TodoWrite with it verbatim next turn.
9. **AR scripts run as direct top-level Bash invocations only.**
   To *invoke* an AR script the command must be a single foreground
   call: `python .autoresearch/scripts/<name>.py <task_dir> [args...]`
   (env-var prefixes, Python flags, and FD redirection like `> log
   2>&1` are fine). Wrappers (`nohup`, `bash -lc`, `sh -c`, subshells,
   `$(...)`), chains (`&&`, `||`, `;`, `|`), and backgrounding (`&`)
   are unsupported and rejected by `hook_guard_bash`. Run multiple
   AR scripts as separate Bash tool calls.

   *Reading* AR scripts (e.g. `cat .autoresearch/scripts/X.py`,
   `git diff -- .autoresearch/scripts/X.py`) is allowed because the
   classifier sees those heads as read-only and the args don't
   execute. The Read tool is still preferred — it's the idiomatic
   way to inspect file contents in Claude Code.
10. **DIAGNOSE phase ends with a new plan.** Two paths to that end:
   - **Preferred (subagent route).** Call `Task(subagent_type='ar-diagnosis')`;
     the subagent's prompt asks it to Write a structured artifact at
     `<task_dir>/.ar_state/diagnose_v<plan_version>.md` containing three
     sections (`Root cause` / `Fix directions` / `What to avoid`),
     useful citations of recent FAIL rounds by `R<n>`, and the marker
     line `[AR DIAGNOSE COMPLETE marker_v<plan_version>]`. The host
     gates on file presence, marker, and section names; then write
     `plan_items.xml` and run `create_plan.py`.
   - **Fallback (manual planning).** After 5 failed Task attempts on the
     same `plan_version`, the artifact gate is relaxed: write
     `plan_items.xml` yourself using `history.jsonl` + `plan.md`, then
     run `create_plan.py`. Further Task calls are blocked at this point.

   While the artifact is invalid AND attempts < cap, Bash is locked to
   read-only / lifecycle ops (no AR scripts beyond `create_plan.py`,
   which is itself gated on artifact validity). Stop is blocked the
   entire time DIAGNOSE is active — only `create_plan.py` advancing
   phase to EDIT releases the lock.

   Provenance note: hook payloads do NOT distinguish main agent from
   subagent, so the host validates the artifact's CONTENT only — not who
   wrote it. The subagent path is preferred because the prompt and
   read-only-by-default tool isolation produce a more reliable diagnosis,
   not because the host can prove the subagent wrote the file.

## Dependencies

- Python >= 3.10
- PyYAML (`pip install pyyaml`)
- Claude Code CLI
