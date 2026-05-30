"""Eval dispatcher — one local-subprocess transport, one assembler.

Public entry point: `run_eval(task_dir, config, device_id=None) ->
EvalResult`. Probes the Ascend runtime and dispatches to
`run_local_eval`, which builds an EvalRequest, calls
`utils.eval_runner.local_eval`, and assembles the response into an
EvalResult via `eval_assemble`.

Request-time logic (case-count probe, timeout scaling, sticky lookup)
lives in `eval_request`; response interpretation lives in
`eval_assemble`. This file is a thin orchestrator.
"""
import os
import sys
from typing import Optional

from .eval_assemble import assemble_eval_result as _assemble_eval_result
from .eval_request import build_eval_request
from .loader import TaskConfig
from .metric_policy import EvalOutcome, EvalResult


def _log_request(prefix: str, request) -> None:
    if request.num_cases > 1:
        print(f"[{prefix}] eval_timeout scaled per shape: "
              f"{request.config.eval_timeout}s/shape x "
              f"{request.num_cases} cases = {request.timeout}s",
              file=sys.stderr)
    note = request.sticky_note()
    if note:
        print(f"[{prefix}] Skipping ref profile; {note}", file=sys.stderr)


def run_local_eval(task_dir: str, config: TaskConfig,
                   device_id: Optional[int] = None) -> EvalResult:
    """Drive a single `eval_kernel.py` subprocess (via two passes — ref
    first, kernel second) and assemble an EvalResult."""
    _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from utils.eval_runner import local_eval

    if device_id is not None:
        dev = int(device_id)
    elif config.devices:
        dev = int(config.devices[0])
    else:
        # No explicit device on the call AND task.yaml has no `devices`
        # field. Fall back to NPU 0 with a loud warning — legitimate
        # callers (notebooks, ad-hoc reruns) do hit this path, but a
        # SILENT fallback to 0 is what once let `--devices 6` get
        # rewritten to 0 and OOM on a busy NPU.
        dev = 0
        print(
            "[local_eval] WARNING: no device specified (no device_id arg, "
            "no `devices` field in task.yaml). Defaulting to NPU 0. If "
            "another card is intended, pass --device-id N or set "
            "`devices: [N]` in task.yaml.",
            file=sys.stderr,
        )

    request = build_eval_request(task_dir, config)
    _log_request("local_eval", request)
    kernel_basename = config.editable_files[0].replace(".py", "")
    ref_basename = config.ref_file.replace(".py", "")
    print(f"[local_eval] device={dev}; eval_kernel.py "
          f"(verify + profile_gen"
          f"{'' if request.sticky else ' + profile_base'})...",
          file=sys.stderr)

    verify_resp, profile_resp = local_eval(
        task_dir=task_dir,
        op_name=config.name,
        kernel_file=kernel_basename,
        ref_file=ref_basename,
        timeout=request.timeout,
        device_id=dev,
        override_base_time_us=request.override_base_us,
        override_base_per_shape_us=request.override_base_per_shape_us,
    )
    return _assemble_eval_result(verify_resp, profile_resp)


def run_eval(task_dir: str, config: TaskConfig,
             device_id: Optional[int] = None) -> EvalResult:
    """Probe the Ascend runtime; on success run `run_local_eval`,
    otherwise return an EvalResult carrying a clear "no execution
    backend" error so the user can fix the local install.
    """
    _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from utils.eval_runner import detect_local_backend
    ok, why = detect_local_backend()
    if ok:
        print(f"[eval] ascend runtime ok: {why}", file=sys.stderr)
        return run_local_eval(task_dir, config, device_id=device_id)

    return EvalResult(
        outcome=EvalOutcome.INFRA_FAIL,
        error=(
            f"ascend runtime unavailable: {why}. Install torch + "
            f"torch_npu + CANN locally."
        ),
    )
