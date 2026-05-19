"""Batch driver for /autoresearch.

Loads a manifest from <batch_dir>/manifest.{yaml,json}, resolves the op
list against the <op_name>_{ref,kernel}.py naming convention, then drives each
op end-to-end via headless `claude --print`. Streams stdout to console and
batch.log, updates batch_progress.json after every op.

Usage:
    python .autoresearch/scripts/batch/run.py <batch_dir> \\
        --mode {ref-kernel,ref} --devices N \\
        [--max-rounds 30] [--eval-timeout 120] [--timeout-min 180] \\
        [--only op1,op2] [--limit N] [--retry-errored] [--cooldown-sec 5]
"""
from __future__ import annotations

import argparse
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf

# Force line-buffered stdout so logs flush in real time when run via nohup.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass
os.environ.setdefault("PYTHONUNBUFFERED", "1")


PROMPT_TEMPLATE = """\
/autoresearch --ref {ref}{kernel_arg} --op-name {op} {hw} --max-rounds {rounds} --eval-timeout {timeout}

CRITICAL rules — read carefully, this session is non-interactive:

1. After scaffold prints "Task directory created: <path>", your VERY FIRST
   subsequent action MUST be exactly:
       export AR_TASK_DIR="<that path>"
   The double quotes are required so paths with spaces or backslashes
   (e.g. C:\\Users\\Foo Bar\\...) survive shell parsing. This single
   command writes .autoresearch/.active_task, which activates the hook
   chain — every PostToolUse gate keys off that file. THIS IS THE
   SINGLE MOST IMPORTANT STEP.

2. {mode_block}

3. In EDIT phase use the Edit tool (or Write for full rewrites).
   PostToolUse validates kernel.py and auto-advances on pass.

4. Treat hook output as authoritative. Each hook prints the legal next
   action on stderr (or as additionalContext). Hooks gate every script
   to the right phase (e.g. baseline.py runs only in BASELINE); when a
   hook blocks something, the rejection reason is the next step.

5. Keep working through whatever phase the hooks indicate, until the
   framework itself writes phase=FINISH (which only happens when
   eval_rounds reaches max-rounds — settling all current plan items
   triggers REPLAN, not FINISH). The session is fully unattended; the
   orchestrator detects completion by reading .ar_state/.phase. When
   the hooks have nothing more to ask of you, the session ends
   naturally on your last tool call.
"""

MODE_BLOCK = {
    "ref-kernel": (
        "The kernel.py we passed via --kernel is a verified seed. "
        "Scaffold's --run-baseline runs it; on PASS .ar_state/.phase is "
        "set to PLAN immediately, skipping GENERATE_KERNEL. Your job is "
        "PERFORMANCE OPTIMIZATION via PLAN -> EDIT -> VERIFY for the "
        "configured max-rounds: propose targeted incremental edits to "
        "ModelNew (block sizes, memory layout, vectorization, fewer DRAM "
        "round-trips) and let pipeline.py measure the speedup. If "
        "baseline unexpectedly fails, the hook demotes to "
        "GENERATE_KERNEL; recover with the smallest diff against the "
        "seed that fixes the regression."
    ),
    "ref": (
        "Only the reference is supplied. Scaffold runs GENERATE_KERNEL "
        "first; produce a working kernel.py that passes baseline, then "
        "optimize via PLAN -> EDIT -> VERIFY for the remaining rounds."
    ),
}

LOCK_FILENAME = ".batch.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            SYNCHRONIZE = 0x00100000
            h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if not h:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        except Exception:
            # Can't tell — err on the safe side and assume alive so the user
            # has to confirm by removing the lock manually.
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def acquire_lock(batch_dir: Path) -> Path:
    """Prevent two run.py instances racing on the same batch_progress.json.
    Stale locks (PID gone) are auto-cleared; live locks abort with a hint."""
    lock = batch_dir / LOCK_FILENAME
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = -1
        if pid > 0 and _pid_alive(pid):
            sys.exit(
                f"\nanother batch run is active on this batch dir "
                f"(pid={pid}, lock={lock}).\n"
                f"if you're sure no run.py is running, remove {lock} and retry.\n"
            )
        # stale lock — overwrite below
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return lock


def release_lock(lock: Path) -> None:
    try:
        lock.unlink()
    except OSError:
        pass


def recover_stale_running(progress: dict) -> int:
    """Demote any 'running' cases to 'error'. We hold the batch dir lock by
    the time this is called, so anything still 'running' is an orphan from a
    previous run.py that died (SIGKILL, OOM, machine reboot)."""
    cases = progress.get("cases", {})
    n = 0
    now = mf.now_iso()
    for c in cases.values():
        if c.get("status") == "running":
            c["status"] = "error"
            c["finished_at"] = now
            existing = (c.get("note") or "").strip()
            tag = "stale running, demoted on batch restart"
            c["note"] = f"{existing}; {tag}" if existing else tag
            n += 1
    return n


def build_prompt(case: dict, mode: str, hw_arg: str,
                 max_rounds: int, eval_timeout: int) -> str:
    """Quote every value-bearing flag with shlex.quote so paths with
    spaces (e.g. batch dir under `C:\\Users\\Foo Bar\\...`, or
    `--output-dir "my tasks"`) reach /autoresearch as one argv each."""
    kernel_arg = (f" --kernel {shlex.quote(case['kernel'])}"
                  if case.get("kernel") else "")
    return PROMPT_TEMPLATE.format(
        ref=shlex.quote(case["ref"]),
        kernel_arg=kernel_arg,
        op=shlex.quote(case["op_name"]),
        hw=hw_arg,
        rounds=max_rounds,
        timeout=eval_timeout,
        mode_block=MODE_BLOCK[mode],
    )


def build_claude_cmd(args: argparse.Namespace, prompt: str) -> list[str]:
    cmd = [
        args.claude_bin,
        "--print",
        "--permission-mode", "acceptEdits",
        "--output-format", "text",
    ]
    if args.model:
        cmd += ["--model", args.model]
    cmd += args.extra_claude_arg
    cmd += [prompt]
    return cmd


def env_with_no_proxy() -> dict[str, str]:
    env = os.environ.copy()
    extras = "127.0.0.1,localhost"
    existing = env.get("NO_PROXY", "")
    env["NO_PROXY"] = f"{existing},{extras}".strip(",") if existing else extras
    env["no_proxy"] = env["NO_PROXY"]
    return env


def run_one(batch_dir: Path, case: dict, args: argparse.Namespace,
            mode: str, hw_arg: str, log_fp) -> int:
    op = case["op_name"]
    repo_root = mf.repo_root()
    prompt = build_prompt(case, mode, hw_arg,
                          args.max_rounds, args.eval_timeout)
    cmd = build_claude_cmd(args, prompt)

    started = time.time()
    started_iso = mf.now_iso()
    mf.update_case(batch_dir, op,
                   status="running",
                   started_at=started_iso,
                   finished_at=None,
                   task_dir=None,
                   final_phase=None,
                   rc=None,
                   note="")

    header = (f"\n{'=' * 72}\n"
              f"[run {datetime.now().isoformat(timespec='seconds')}] op={op} "
              f"{hw_arg} rounds={args.max_rounds}\n"
              f"[run] launching: {args.claude_bin} --print "
              f"(cwd={repo_root}, timeout={args.timeout_min}min)\n"
              f"{'─' * 72}\n")
    sys.stdout.write(header)
    sys.stdout.flush()
    log_fp.write(header)
    log_fp.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env_with_no_proxy(),
    )

    # Background reader thread + bounded queue.get poll. The earlier
    # `for line in proc.stdout` form blocks on readline indefinitely when
    # claude is alive but silent (API retry, deep IO wait), so the
    # wall-clock check inside the loop never fires and `--timeout-min`
    # becomes a no-op. Selectors aren't an option because Windows can't
    # select() on pipe handles, so we use a thread + queue (cross-platform).
    line_q: "queue.Queue[str]" = queue.Queue()
    reader_done = threading.Event()

    def _reader():
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line_q.put(line)
        finally:
            reader_done.set()

    threading.Thread(target=_reader, daemon=True).start()

    timeout_s = args.timeout_min * 60
    interrupted = False
    try:
        while True:
            try:
                # Short poll so a silent claude still hits the wall-clock
                # check below within 5s of crossing the deadline.
                line = line_q.get(timeout=5)
            except queue.Empty:
                if time.time() - started > timeout_s:
                    msg = (f"[run] WALL-CLOCK TIMEOUT after "
                           f"{args.timeout_min}min, killing claude\n")
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    log_fp.write(msg)
                    log_fp.flush()
                    proc.kill()
                    break
                if reader_done.is_set() and line_q.empty():
                    # Reader saw EOF and queue is fully drained.
                    break
                continue
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fp.write(line)
            log_fp.flush()
        proc.wait(timeout=30)
    except KeyboardInterrupt:
        interrupted = True
        msg = "\n[run] Ctrl-C received, killing claude\n"
        sys.stdout.write(msg)
        log_fp.write(msg)
        try:
            proc.kill()
        except Exception:
            pass

    elapsed = time.time() - started
    footer = (f"{'─' * 72}\n"
              f"[run] claude exited rc={proc.returncode} after {elapsed:.0f}s\n")
    sys.stdout.write(footer)
    log_fp.write(footer)
    log_fp.flush()

    # Authoritative task discovery: scan ar_tasks/ for the freshest dir
    # matching this op. There is no in-band marker — the model is forbidden
    # from self-declaring "done"/"stuck" (any such instruction would let an
    # early-stop heuristic creep back in). Truth lives in .ar_state/.phase.
    td = mf.find_recent_task_dir(op, since_ts=started - 5)
    if td is None:
        mf.update_case(batch_dir, op,
                       status="error",
                       finished_at=mf.now_iso(),
                       rc=proc.returncode,
                       note=f"no task_dir found; rc={proc.returncode}"
                            + ("; interrupted" if interrupted else ""))
        return 130 if interrupted else 2
    task_dir = td
    phase = mf.read_phase(td)

    result = mf.read_task_state(task_dir)
    final_status = ("done" if phase == "FINISH" and not interrupted
                    else "error")
    note = ""
    if final_status == "error":
        note = f"phase={phase} rc={proc.returncode}"
        if interrupted:
            note += "; interrupted"

    mf.update_case(batch_dir, op,
                   status=final_status,
                   task_dir=str(task_dir.resolve()),
                   finished_at=mf.now_iso(),
                   final_phase=phase,
                   rc=proc.returncode,
                   result=result,
                   note=note)

    sys.stdout.write(
        f"[run] result: op={op} task_dir={task_dir} phase={phase} "
        f"status={final_status}\n"
    )
    if interrupted:
        return 130
    return 0 if final_status == "done" else 1


def filter_queue(progress: dict, args: argparse.Namespace) -> list[dict]:
    statuses = {"pending"}
    if args.retry_errored:
        statuses.add("error")
    only = {s.strip() for s in (args.only or "").split(",") if s.strip()}
    out: list[dict] = []
    for v in progress.get("cases", {}).values():
        if v.get("status") not in statuses:
            continue
        if only and v.get("op_name") not in only:
            continue
        out.append(v)
    return out


def print_summary(batch_dir: Path, total_elapsed: float,
                  hw_arg: str) -> None:
    """Compact end-of-batch report + concrete next-step commands.

    Status lines: just done / error counts (skip / pending only shown when
    nonzero). Speedup distribution collapses into a single line — regress
    cases are part of `done`, not called out separately.

    Next-step commands echo back enough of the original invocation that
    the user can paste directly: batch dir path + the hardware flag we
    were called with. mode is read from the manifest by run.py so we
    don't repeat it.
    """
    progress = mf.load_progress(batch_dir)
    cases = progress.get("cases", {})
    counts = {"done": 0, "error": 0, "skip": 0, "pending": 0, "running": 0}
    speedups: list[float] = []
    for v in cases.values():
        s = v.get("status", "pending")
        counts[s] = counts.get(s, 0) + 1
        if s != "done":
            continue
        r = v.get("result") or {}
        bm, best = r.get("baseline_metric"), r.get("best_metric")
        if isinstance(bm, (int, float)) and isinstance(best, (int, float)) and best > 0:
            speedups.append(bm / best)

    print()
    print("=" * 72)
    print(f"[batch done] elapsed={total_elapsed/60:.1f}min")

    if speedups:
        import statistics
        speed_note = (f"  (median {statistics.median(speedups):.2f}x, "
                      f"best {max(speedups):.2f}x, "
                      f"worst {min(speedups):.2f}x; "
                      f"{len(speedups)} with metric)")
    else:
        speed_note = ""
    print(f"  done : {counts['done']}{speed_note}")
    print(f"  error: {counts['error']}")
    if counts["skip"]:
        print(f"  skip : {counts['skip']}")
    if counts["pending"]:
        print(f"  pending: {counts['pending']}")

    # Resolve the batch dir path the way the user is most likely to type it
    # (relative to repo root if it's under there; absolute otherwise).
    repo_root = mf.repo_root()
    try:
        ws_str = str(batch_dir.relative_to(repo_root))
    except ValueError:
        ws_str = str(batch_dir)

    suggestions: list[tuple[str, str]] = []
    if counts["error"]:
        suggestions.append((
            f"retry {counts['error']} errored ops",
            f"python .autoresearch/scripts/batch/run.py {ws_str} "
            f"{hw_arg} --retry-errored",
        ))
    if counts["pending"]:
        suggestions.append((
            f"resume {counts['pending']} pending ops",
            f"python .autoresearch/scripts/batch/run.py {ws_str} {hw_arg}",
        ))

    if suggestions:
        print()
        print("next steps:")
        for label, cmd in suggestions:
            print(f"  {label}:")
            print(f"    {cmd}")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch driver for /autoresearch.")
    ap.add_argument("batch_dir", help="dir containing manifest.yaml/json")
    ap.add_argument("--mode", choices=mf.VALID_MODES,
                    help="ref-kernel or ref (overrides manifest.mode)")
    ap.add_argument("--devices", default="",
                    help="NPU device ids, e.g. 0 or 0,1 (required)")
    ap.add_argument("--max-rounds", type=int, default=30)
    ap.add_argument("--eval-timeout", type=int, default=120,
                    help="per-shape verify/profile budget in seconds. The "
                         "eval call is capped at eval_timeout * num_cases "
                         "(num_cases comes from get_input_groups() / get_inputs()). "
                         "Single-shape ops keep the original semantics.")
    ap.add_argument("--timeout-min", type=int, default=180,
                    help="hard wall-clock cap per op in minutes")
    ap.add_argument("--only", default="", help="comma-separated op names")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N ops (0 = no limit)")
    ap.add_argument("--retry-errored", action="store_true",
                    help="also queue ops with status=error")
    ap.add_argument("--cooldown-sec", type=int, default=5,
                    help="seconds to sleep between ops")
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--model", default="")
    ap.add_argument("--extra-claude-arg", action="append", default=[],
                    help="extra arg to pass to claude (repeatable)")
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        sys.exit(f"batch dir not found: {batch_dir}")

    try:
        manifest_path = mf.find_manifest(batch_dir)
    except mf.ManifestError as e:
        sys.exit(str(e))

    try:
        manifest_data = mf.load_manifest(manifest_path)
    except mf.ManifestError as e:
        sys.exit(f"failed to load {manifest_path}: {e}")

    mode = args.mode or manifest_data.get("mode")
    if not mode:
        sys.exit("--mode is required (also accepted as `mode:` in manifest)")
    if mode not in mf.VALID_MODES:
        sys.exit(f"--mode must be one of {mf.VALID_MODES}, got {mode!r}")

    if not args.devices:
        sys.exit("--devices is required (local-only eval)")
    hw_arg = f"--devices {args.devices}"

    try:
        cases = mf.resolve_cases(batch_dir, manifest_data, mode)
    except mf.ManifestError as e:
        sys.exit(f"manifest validation failed: {e}")

    lock_path = acquire_lock(batch_dir)
    try:
        progress = mf.load_progress(batch_dir)
        demoted = recover_stale_running(progress)
        progress, dropped = mf.merge_cases(progress, cases, mode)
        mf.save_progress(batch_dir, progress)
        if demoted:
            print(f"[batch] demoted {demoted} stale 'running' op(s) "
                  f"from a previous run -> error")
        if dropped:
            preview = ", ".join(dropped[:5]) + (
                f", ... (+{len(dropped) - 5} more)" if len(dropped) > 5 else "")
            print(f"[batch] dropped {len(dropped)} op(s) no longer in manifest: "
                  f"{preview}")

        queue = filter_queue(progress, args)
        if not queue:
            print("nothing to run.")
            return 0
        if args.limit:
            queue = queue[: args.limit]

        print(f"[batch {datetime.now().isoformat(timespec='seconds')}] "
              f"batch_dir={batch_dir}  mode={mode}  {hw_arg}\n"
              f"[batch] queue size: {len(queue)}  rounds={args.max_rounds}")

        log_path = batch_dir / mf.LOG_FILENAME
        log_fp = log_path.open("a", encoding="utf-8", buffering=1)

        succeeded = failed = skipped = 0
        total_started = time.time()
        rc_final = 0
        try:
            for i, case in enumerate(queue, 1):
                op = case["op_name"]
                current = filter_queue(mf.load_progress(batch_dir), args)
                if not any(c["op_name"] == op for c in current):
                    print(f"[{i}/{len(queue)}] {op}: status changed underfoot, skipping")
                    skipped += 1
                    continue

                print(f"\n[{i}/{len(queue)}] starting op={op}  "
                      f"elapsed_total={(time.time()-total_started)/60:.1f}min")

                try:
                    rc = run_one(batch_dir, case, args, mode, hw_arg, log_fp)
                except KeyboardInterrupt:
                    print("\n[batch] Ctrl-C — current op recorded, stopping.")
                    rc_final = 130
                    break

                if rc == 0:
                    succeeded += 1
                elif rc == 130:
                    failed += 1
                    print("\n[batch] op interrupted, stopping.")
                    rc_final = 130
                    break
                else:
                    failed += 1

                print(f"[{i}/{len(queue)}] {op} done rc={rc}  "
                      f"running totals: ok={succeeded} fail={failed} skipped={skipped}")

                if i < len(queue) and args.cooldown_sec > 0:
                    time.sleep(args.cooldown_sec)
        finally:
            log_fp.close()

        print_summary(batch_dir, time.time() - total_started, hw_arg)
        if rc_final:
            return rc_final
        return 0 if failed == 0 else 1
    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    sys.exit(main())
