"""Local subprocess driver for autoresearch eval.

Runs the static `eval_kernel.py` once per round (one subprocess does
verify + profile_gen + optional profile_base in sequence — the triton
JIT cache populated during verify is then warm for profile_gen).
Replaced an earlier "materialize verify_<op>.py / profile_<op>_*.py
under <task_dir>/.ar_eval/ then run each separately" flow.

Public surface:
  - detect_local_backend() -> (ok, why)
  - local_eval(task_dir, op_name, kernel_file, ref_file,
               timeout, device_id, warmup, repeats,
               override_base_time_us) -> (verify_resp, profile_resp)

Precision: allclose-style |diff| <= atol + rtol*|ref| per `correctness.py`.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ascend runtime probe
# ---------------------------------------------------------------------------

# Subprocess probe — keeps torch import out of the parent process so a
# half-broken install can't poison hooks/scaffold/etc.
_PROBE_SCRIPT = r"""
import sys
try:
    import torch
except Exception as e:
    print(f"NO: torch missing or broken: {e}")
    sys.exit(1)
try:
    import torch_npu  # noqa: F401
except Exception as e:
    print(f"NO: torch_npu missing: {e}")
    sys.exit(1)
try:
    n = torch.npu.device_count()
except Exception as e:
    print(f"NO: torch.npu unavailable: {e}")
    sys.exit(1)
print(f"OK: npu devices={n}")
sys.exit(0)
"""

_DETECT_CACHE: list = []  # holds (ok, why) once probed


def detect_local_backend() -> tuple[bool, str]:
    """Probe whether this machine can run Ascend NPU eval locally.

    Cached so repeated calls in the same Python process don't pay the
    subprocess cost. Returns (ok, human-readable reason).
    """
    if _DETECT_CACHE:
        return _DETECT_CACHE[0]
    probe_env = os.environ.copy()
    # Windows libiomp5 double-load workaround — same as we set for the
    # eval subprocess. Without it, torch.import on Windows aborts with OMP
    # Error #15 and the probe falsely reports the runtime unavailable.
    probe_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        r = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            capture_output=True, text=True, timeout=30, env=probe_env,
        )
    except subprocess.TimeoutExpired:
        result = (False, "ascend probe timed out (>30s)")
    except Exception as e:
        result = (False, f"ascend probe failed to launch: {e}")
    else:
        line = (r.stdout or r.stderr or "").strip().splitlines()
        msg = line[-1] if line else "(no output)"
        result = (r.returncode == 0, msg)
    _DETECT_CACHE.append(result)
    return result


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------

def _build_env(device_id: int) -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["DEVICE_ID"] = str(device_id)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows libiomp5 dup-load (no-op on Linux)
    return env


def _run_subprocess(cmd: list[str], cwd: str, env: dict,
                    timeout: int) -> tuple[int, str, str]:
    """subprocess.run wrapper that returns (rc, stdout, stderr).

    Returns rc=124 on timeout (matching the GNU `timeout(1)` convention) and
    a stderr describing the timeout. Process-group cleanup uses os.setsid
    on POSIX so a hung kernel can't leave orphan children; on Windows we
    rely on subprocess's own kill().
    """
    popen_kwargs = {
        "cwd": cwd,
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if hasattr(os, "setsid"):
        popen_kwargs["preexec_fn"] = os.setsid
    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except Exception as e:
        return 1, "", f"failed to launch eval: {e}"

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return (proc.returncode or 0,
                (stdout or b"").decode(errors="replace"),
                (stderr or b"").decode(errors="replace"))
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg"):
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            stdout, stderr = b"", b""
        return (124,
                (stdout or b"").decode(errors="replace"),
                (stderr or b"").decode(errors="replace") +
                f"\n[eval_runner] eval timed out after {timeout}s")


def _avg_us(d: Optional[dict]) -> Optional[float]:
    if not isinstance(d, dict):
        return None
    v = d.get("avg_time_us")
    if isinstance(v, (int, float)) and 0 < v < float("inf"):
        return float(v)
    return None


# ---------------------------------------------------------------------------
# Combined verify + profile (single subprocess)
# ---------------------------------------------------------------------------

def local_eval(task_dir: str, op_name: str,
               kernel_file: str, ref_file: str,
               timeout: int, device_id: int = 0,
               warmup: int = 10, repeats: int = 100,
               override_base_time_us: Optional[float] = None,
               ) -> tuple[dict, dict]:
    """Run eval_kernel.py once with verify + profile_gen (+ profile_base
    when no sticky baseline is supplied). Returns
    (verify_resp, profile_resp) in the shape `_assemble_eval_result`
    expects, so eval_client doesn't have to know how phases are
    packaged.
    """
    skip_base = (override_base_time_us is not None
                 and override_base_time_us > 0
                 and override_base_time_us < float("inf"))
    phases = ["verify", "profile_gen"] + ([] if skip_base else ["profile_base"])

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(scripts_dir, "eval_kernel.py")
    # CANN's C runtime writes "[Warning]: tiling struct ..." to stdout
    # without trailing newlines, which would concatenate onto our JSON
    # if we tried to parse stdout. Use a sidecar file instead.
    out_path = os.path.join(os.path.abspath(task_dir), ".eval_result.json")
    cmd = [
        sys.executable, eval_script,
        "--task-dir", os.path.abspath(task_dir),
        "--op-name", op_name,
        "--kernel-file", kernel_file,
        "--ref-file", ref_file,
        "--device-id", str(device_id),
        "--warmup", str(warmup),
        "--repeats", str(repeats),
        "--phases", ",".join(phases),
        "--output", out_path,
    ]

    # Always wipe a stale result file before launch — if the subprocess
    # crashes before writing, we want a missing file rather than the
    # previous round's data masquerading as this round's.
    try:
        os.remove(out_path)
    except FileNotFoundError:
        pass

    env = _build_env(device_id)
    rc, stdout, stderr = _run_subprocess(cmd, cwd=task_dir, env=env,
                                          timeout=timeout)
    log_combined = (stdout + ("\n" + stderr if stderr else "")).strip()

    payload: dict = {}
    if os.path.isfile(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            logger.warning("eval_runner: cannot parse %s: %s", out_path, e)

    verify_block = payload.get("verify")
    base_block = payload.get("profile_base")
    gen_block = payload.get("profile_gen")

    verify_correct = (isinstance(verify_block, dict)
                      and bool(verify_block.get("correctness")))
    verify_resp = {
        "success": verify_correct,
        "log": log_combined,
        "artifacts": {},
        "returncode": rc,
    }

    artifacts: dict[str, str] = {}
    if isinstance(base_block, dict):
        artifacts["base_profile_result.json"] = json.dumps(base_block)
    if isinstance(gen_block, dict):
        artifacts["generation_profile_result.json"] = json.dumps(gen_block)

    base_time = (float(override_base_time_us) if skip_base
                 else _avg_us(base_block))
    gen_time = _avg_us(gen_block)
    profile_resp = {
        "success": gen_time is not None or base_time is not None,
        "log": log_combined,
        "artifacts": artifacts,
        "gen_time": gen_time,
        "base_time": base_time,
    }

    if skip_base:
        # Stay shape-compatible with old log: keep the explicit hint so the
        # round transcript still records why no base-profile artifact landed.
        profile_resp["log"] = (
            f"[eval_runner] sticky baseline override = "
            f"{override_base_time_us:.2f} us; profile_base skipped\n"
            + profile_resp["log"]
        )

    return verify_resp, profile_resp
