"""task_config package — facade over three single-concern submodules.

Layout:

    loader            — TaskConfig dataclass + load_task_config (YAML
                        parsing). No internal deps; everyone else
                        consumes TaskConfig from here.
    metric_policy     — EvalResult, is_improvement, check_constraints,
                        format_result_summary. Pure data + arithmetic;
                        no I/O. Imported by keep_or_discard, dashboard.
    eval_client       — Local subprocess dispatcher, result assembly.
                        Depends on loader + metric_policy. Drives the
                        static `eval_kernel.py` (one subprocess per
                        round) via `eval_runner.local_eval`.

This `__init__.py` re-exports every public name. New code may prefer
sub-module imports for readability:

    from task_config.metric_policy import EvalResult, is_improvement
    from task_config.eval_client    import run_eval
"""
# fmt: off
from .loader import (
    TaskConfig, load_task_config,
)
from .metric_policy import (
    EvalResult, check_constraints, is_improvement, format_result_summary,
    # Operator table — internal but referenced by some debug scripts that
    # introspect supported constraint operators.
    _CONSTRAINT_OPS,
)
from .eval_client import (
    run_eval, run_local_eval, _assemble_eval_result,
)
# fmt: on
