"""engine/ — orchestration scripts the LLM and hooks invoke via subprocess.

Holds the blessed-script set: pipeline.py (main post-edit driver) plus
the single-purpose CLIs it spawns (quick_check, eval_wrapper,
eval_kernel, settle), the BASELINE-phase entry (baseline.py), the
PLAN-phase entry (create_plan.py), and the /autoresearch arg dispatcher
(parse_args.py).

Body-level logic (record_round, run_baseline_init) lives in workflow/
and is now called in-process; the earlier shell wrappers
(keep_or_discard.py, _baseline_init.py) have been deleted because every
caller went through workflow.* directly.

These are CLIs, not a library — they are exec'd via Bash, not imported.
The package marker exists so cross-package imports (e.g. phase_machine
referencing engine.quick_check.check_editable_files) resolve cleanly.
"""
