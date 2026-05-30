"""Local subprocess driver for autoresearch eval.

Runs the static `eval_kernel.py` per round. Two subprocesses are
launched in sequence so a kernel-induced SIGKILL / device hang in the
second cannot prevent ref measurement from landing on disk:

  1. ref subprocess  — phases=profile_base (loads ref only)
  2. kernel subprocess — phases=verify,profile_gen (loads ref + kernel;
     triton JIT cache populated by verify is warm for profile_gen)

When the caller supplies a sticky `override_base_time_us`, step 1 is
skipped (ref doesn't need re-measuring round-to-round). Earlier the
two passes lived in ONE subprocess (verify → profile_gen →
profile_base); a kernel UB overflow or device fault during verify /
profile_gen would kill the process before profile_base ran, leaving
`baseline_metric=None` permanently — that's the failure mode this
split fixes.

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
# Verify + profile (two subprocesses: ref first, then kernel)
# ---------------------------------------------------------------------------

def _read_sidecar(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        logger.warning("eval_runner: cannot parse %s: %s", path, e)
        return {}


def local_eval(task_dir: str, op_name: str,
               kernel_file: str, ref_file: str,
               timeout: int, device_id: int = 0,
               warmup: int = 10, repeats: int = 100,
               override_base_time_us: Optional[float] = None,
               override_base_per_shape_us: Optional[list] = None,
               ) -> tuple[dict, dict]:
    """Run eval_kernel.py in two passes (see module docstring):
      ref pass:    --phases profile_base
      kernel pass: --phases verify,profile_gen
    Then merge per-phase sidecars and assemble (verify_resp,
    profile_resp) in the shape `_assemble_eval_result` consumes.

    When `override_base_time_us` is supplied (sticky baseline), the
    ref pass is skipped — ref doesn't need re-measuring round-to-round.

    override_base_per_shape_us: when provided alongside the aggregate,
    profile_resp gets a synthesized base profile artifact that includes
    per_shape entries — keeps speedup_vs_ref aggregation (geomean of
    per-shape ratios) consistent across sticky-baseline rounds and the
    initial round that actually measured ref.
    """
    skip_base = (override_base_time_us is not None
                 and override_base_time_us > 0
                 and override_base_time_us < float("inf"))

    # __file__ is scripts/utils/eval_runner.py — climb one level then
    # dive into engine/ where eval_kernel.py lives post-restructure.
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_script = os.path.join(scripts_dir, "engine", "eval_kernel.py")
    abs_task = os.path.abspath(task_dir)
    env = _build_env(device_id)

    # Distinct sidecars per pass so the second pass can't overwrite the
    # first. Both files persist under task_dir for forensics; downstream
    # code reads from in-memory verify_resp / profile_resp, not the
    # files.
    ref_path = os.path.join(abs_task, ".eval_result_ref.json")
    kernel_path = os.path.join(abs_task, ".eval_result_kernel.json")
    for p in (ref_path, kernel_path):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    def _run_phases(phases: list[str], out_path: str) -> tuple[int, str]:
        cmd = [
            sys.executable, eval_script,
            "--task-dir", abs_task,
            "--op-name", op_name,
            "--kernel-file", kernel_file,
            "--ref-file", ref_file,
            "--device-id", str(device_id),
            "--warmup", str(warmup),
            "--repeats", str(repeats),
            "--phases", ",".join(phases),
            "--output", out_path,
        ]
        rc, stdout, stderr = _run_subprocess(
            cmd, cwd=task_dir, env=env, timeout=timeout)
        log = (stdout + ("\n" + stderr if stderr else "")).strip()
        return rc, log

    # --- Pass 1: ref-only subprocess (immune to kernel-induced death) ----
    if skip_base:
        ref_log = ""
        ref_payload: dict = {}
    else:
        _, ref_log = _run_phases(["profile_base"], ref_path)
        ref_payload = _read_sidecar(ref_path)

    # --- Pass 2: kernel subprocess (verify + profile_gen, JIT warm) -------
    # rc here is the kernel-side returncode — the one downstream readers
    # treat as authoritative (verify_resp.returncode). A ref crash
    # surfaces via verify_resp.error_source == "ref" / missing base_time.
    rc, kernel_log = _run_phases(["verify", "profile_gen"], kernel_path)
    kernel_payload = _read_sidecar(kernel_path)

    log_combined = "\n".join(s for s in (ref_log, kernel_log) if s).strip()
    verify_block = kernel_payload.get("verify")
    gen_block = kernel_payload.get("profile_gen")
    base_block = ref_payload.get("profile_base") if not skip_base else None

    verify_correct = (isinstance(verify_block, dict)
                      and bool(verify_block.get("correctness")))
    # error_source: "ref" | "kernel" | None. run_verify tags it on the
    # verify_block; eval_client reads it via verify_resp to decide
    # ref-broken vs kernel-broken (drives INFRA_FAIL vs KERNEL_FAIL).
    error_source = (verify_block.get("error_source")
                    if isinstance(verify_block, dict) else None)
    verify_resp = {
        "success": verify_correct,
        "log": log_combined,
        "artifacts": {},
        "returncode": rc,
        "error_source": error_source,
        # Pass the full verify_block through so eval_client can pull
        # failed_indices / per_case / diagnostics for DIAGNOSE context
        # without re-parsing the log JSON tail (eval_kernel writes its
        # structured result to .eval_result.json, not to stderr).
        "verify_block": verify_block if isinstance(verify_block, dict) else {},
    }

    artifacts: dict[str, str] = {}
    if skip_base and isinstance(override_base_per_shape_us, list) and override_base_per_shape_us:
        # Materialise a base profile artifact from the sticky per-shape
        # baseline so _assemble_eval_result sees per_base alongside
        # per_gen — speedup_vs_ref then computes as geomean of per-shape
        # ratios just like the round-0 baseline did.
        synth_per_shape = [{"avg_time_us": float(v)}
                           for v in override_base_per_shape_us]
        synth_avg = sum(s["avg_time_us"] for s in synth_per_shape) / len(synth_per_shape)
        artifacts["base_profile_result.json"] = json.dumps({
            "avg_time_us": synth_avg,
            "per_shape": synth_per_shape,
            "sticky": True,
        })
    elif isinstance(base_block, dict):
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
