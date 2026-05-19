"""task.yaml loader and the TaskConfig dataclass.

This module's job is parsing — turning the on-disk YAML into a typed
struct. It's the lowest layer in the task_config package: eval_client
depends on TaskConfig but TaskConfig depends on nothing of ours.

The fields here are the schema. Adding a new task.yaml key means adding
a field on `TaskConfig` and reading it in `load_task_config`. Don't
reach into `raw` dicts elsewhere; route every consumer through the
typed dataclass.
"""
from dataclasses import dataclass, field
from typing import Optional

import os
import yaml


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """Minimal task configuration parsed from task.yaml.

    The repo is locked to a single combination by construction
    (Triton-Ascend kernel, Ascend NPU, PyTorch ref). Downstream code
    refers to those constants directly rather than carrying them on
    TaskConfig. `arch` (e.g. `ascend910b3`) varies per machine and is
    derived from the picked --devices via npu-smi.
    """
    name: str
    description: str = ""

    arch: Optional[str] = None

    # Files
    editable_files: list = field(default_factory=list)
    ref_file: str = "reference.py"

    # Eval params
    # Per-SHAPE budget for verify/profile in seconds. eval_client scales it
    # by num_cases (probed from the ref module) before invoking the eval
    # subprocess, so the wall-clock cap is eval_timeout * num_cases.
    # Single-shape refs (num_cases=1) keep the original semantics.
    eval_timeout: int = 600

    # Metric
    primary_metric: str = "score"
    lower_is_better: bool = True
    improvement_threshold: float = 0.0

    # Constraints: {metric_name: (operator_str, threshold)}
    constraints: dict = field(default_factory=dict)

    # Smoke test (optional — quick_check.py runs it before eval when configured)
    smoke_test_script: Optional[str] = None
    smoke_test_timeout: int = 10

    # Triton regression check (validate_triton_impl) on editable files.
    # Default on; disable per-task via `code_checker.enabled: false` in
    # task.yaml or scaffold's --no-code-checker flag. The yaml key name
    # is kept as `code_checker.enabled` for back-compat with existing
    # task.yaml files. When off, quick_check and validate_kernel skip
    # the regression check but still reject the scaffold TODO placeholder.
    code_checker_enabled: bool = True

    # Agent budget
    max_rounds: int = 30

    # Local devices
    devices: list = field(default_factory=list)
    """Device IDs for local eval (written by scaffold from --devices). When
    non-empty, run_local_eval uses devices[0] as default device_id."""


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_task_config(task_dir: str) -> Optional[TaskConfig]:
    """Load TaskConfig from task_dir/task.yaml. Returns None if not found."""
    yaml_path = os.path.join(task_dir, "task.yaml")
    if not os.path.exists(yaml_path):
        return None

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: expected YAML dict, got {type(raw).__name__}")

    name = raw.get("name")
    if not name:
        raise ValueError(f"{yaml_path}: 'name' is required")

    eval_block = raw.get("eval", {})
    metric_block = raw.get("metric", {})
    smoke_block = raw.get("smoke_test", {})
    agent_block = raw.get("agent", {})
    code_checker_block = raw.get("code_checker", {})

    # Parse constraints
    constraints = {}
    for metric_name, spec in raw.get("constraints", {}).items():
        if isinstance(spec, dict):
            constraints[metric_name] = (spec["op"], spec["value"])
        elif isinstance(spec, (list, tuple)) and len(spec) == 2:
            constraints[metric_name] = tuple(spec)

    # Parse devices list. Accepts [5] / "5" / "0,1,2".
    devices_raw = raw.get("devices", [])
    if isinstance(devices_raw, int):
        devices = [devices_raw]
    elif isinstance(devices_raw, str):
        devices = [int(d.strip()) for d in devices_raw.split(",") if d.strip()]
    elif isinstance(devices_raw, list):
        devices = [int(d) for d in devices_raw]
    else:
        devices = []

    return TaskConfig(
        name=name,
        description=raw.get("description", ""),
        arch=raw.get("arch"),
        editable_files=raw.get("editable_files", []),
        ref_file=agent_block.get("ref_file") or "reference.py",
        eval_timeout=eval_block.get("timeout", 600),
        primary_metric=metric_block.get("primary", "score"),
        lower_is_better=metric_block.get("lower_is_better", True),
        improvement_threshold=metric_block.get("improvement_threshold", 0.0),
        constraints=constraints,
        smoke_test_script=smoke_block.get("script"),
        smoke_test_timeout=smoke_block.get("timeout", 10),
        code_checker_enabled=bool(code_checker_block.get("enabled", True)),
        max_rounds=agent_block.get("max_rounds", 30),
        devices=devices,
    )
