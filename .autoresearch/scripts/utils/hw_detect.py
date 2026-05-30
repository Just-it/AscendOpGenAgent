"""Hardware detection — arch derivation from a local NPU device id.

This module's only job is to map `--devices N` to an `arch` string
(e.g. `ascend910b3`) by parsing `npu-smi info`.

No autoresearch/task dependencies — only needs stdlib + subprocess.
Safe to import from scaffold, baseline, or anywhere.
"""
from __future__ import annotations

import re
import subprocess
from typing import Optional


def derive_arch(device_id: int) -> Optional[str]:
    """Return the Ascend NPU arch string (e.g. 'ascend910b3') for
    `device_id`, or None when `npu-smi info` is missing / unparseable.
    Caller decides whether None is fatal."""
    try:
        r = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    # The npu-smi main table row looks like:
    #     | 5     910B3               | Alarm         | ...
    # Match the leading <device_id> and capture the next token →
    # 'ascend910b3'. `npu-smi info -t board -i N` exposes Product/Model
    # but not the architecture string, so we go through the main table.
    pat = re.compile(rf"^\|\s*{int(device_id)}\s+(\S+)\s*\|", re.MULTILINE)
    m = pat.search(r.stdout)
    if not m:
        return None
    name = m.group(1).strip().lower()
    return f"ascend{name}"
