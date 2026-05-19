#!/usr/bin/env python3
"""
Live dashboard for autoresearch progress.

Run in a separate terminal:
    python .autoresearch/scripts/dashboard.py <task_dir> [--watch N]

--watch N: refresh every N seconds (default: 5). Ctrl+C to stop.
Without --watch: print once and exit.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_machine as _pm

# ---------------------------------------------------------------------------
# Non-blocking keyboard input (cross-platform)
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import msvcrt

    def read_key_nonblocking():
        """Return key name or None. Handles arrows/page keys via escape prefix."""
        if not msvcrt.kbhit():
            return None
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):  # Arrow/function key prefix
            if not msvcrt.kbhit():
                return None
            code = msvcrt.getch()
            return {
                b"H": "UP", b"P": "DOWN",
                b"I": "PGUP", b"Q": "PGDN",
                b"G": "HOME", b"O": "END",
            }.get(code)
        if ch == b"\x1b":
            return "ESC"
        if ch == b"q":
            return "QUIT"
        return None

    _old_tty = None

    def setup_keyboard(): pass
    def restore_keyboard(): pass

else:
    import select
    import termios
    import tty

    _old_tty = None

    def setup_keyboard():
        global _old_tty
        _old_tty = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def restore_keyboard():
        if _old_tty:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _old_tty)

    def read_key_nonblocking():
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # Could be ESC or arrow key sequence
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r:
                return "ESC"
            seq = sys.stdin.read(2)
            return {
                "[A": "UP", "[B": "DOWN",
                "[5": "PGUP", "[6": "PGDN",
                "[H": "HOME", "[F": "END",
            }.get(seq)
        if ch == "q":
            return "QUIT"
        return None

# ANSI colors
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


def _read_raw(path):
    # Read everything until EOF — single os.read() returns at most one chunk
    # and short-reads on regular files (multi-shape history.jsonl trivially
    # passes 256 KB after ~25 rounds with 60 cases).
    fd = os.open(path, os.O_RDONLY)
    try:
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        os.close(fd)


def load_json(path):
    if not os.path.exists(path):
        return None
    return json.loads(_read_raw(path))


def load_jsonl(path):
    """Load all entries from a JSONL file; silently drops malformed lines."""
    if not os.path.exists(path):
        return []
    out = []
    for line in _read_raw(path).split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def load_plan(path):
    if not os.path.exists(path):
        return "(no plan yet)", None
    mtime = os.path.getmtime(path)
    return _read_raw(path), mtime


def bar(fraction, width=30):
    filled = int(fraction * width)
    return f"[{'#' * filled}{'.' * (width - filled)}]"


# Visible prefix widths for the two table rows (ANSI colour codes excluded).
# History row:  "  {rnd:>3}  │ {dec:8} │ {metric:>13} │ "
#               2 + 3 + 4 + 8 + 3 + 13 + 3  = 36
# Plan row:     "  {item_id:>4}  │ {status:9} │ "
#               2 + 4 + 4 + 9 + 2 = 21
_HIST_PREFIX_VIS = 36
_PLAN_PREFIX_VIS = 21


def _fit(text: str, avail: int) -> str:
    """Truncate with single-char ellipsis only when the text would overflow
    the available column width. Every description column across the dashboard
    routes through this helper so behaviour stays consistent — render as much
    as the terminal can fit, truncate just enough to land in the column.
    """
    if avail <= 0:
        return ""
    if len(text) <= avail:
        return text
    if avail == 1:
        return "…"
    return text[: avail - 1] + "…"


def render(task_dir, history_offset=0, history_window=None):
    """Render dashboard.

    history_offset: how many rounds to skip from the END (0 = latest).
    history_window: how many rounds to show (None = auto based on terminal height).
    """
    progress = load_json(_pm.progress_path(task_dir))
    history_all = load_jsonl(_pm.history_path(task_dir))
    plan_text, plan_mtime = load_plan(_pm.plan_path(task_dir))

    # Get terminal width for responsive layout. Tables render as wide as the
    # terminal allows; descriptions are truncated only when necessary.
    try:
        term_width = os.get_terminal_size().columns
    except Exception:
        term_width = 100
    hist_desc_avail = max(10, term_width - _HIST_PREFIX_VIS - 2)
    plan_desc_avail = max(10, term_width - _PLAN_PREFIX_VIS - 2)
    divider_width = max(40, term_width - 2)

    lines = []
    lines.append(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗{RESET}")
    lines.append(f"{BOLD}{CYAN}║          AUTORESEARCH DASHBOARD                             ║{RESET}")
    lines.append(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════╝{RESET}")

    if progress is None:
        lines.append(f"\n  {RED}No progress.json found at {_pm.progress_path(task_dir)}{RESET}")
        lines.append(f"  Run /autoresearch --ref ... --op-name ... first.")
        return "\n".join(lines)

    task = progress.get("task", "?")
    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", 20)
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")
    best_commit = progress.get("best_commit", "?")
    failures = progress.get("consecutive_failures", 0)
    plan_ver = progress.get("plan_version", 0)
    status = progress.get("status", "?")
    updated_raw = progress.get("last_updated", "?")
    # Convert UTC to local time
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(updated_raw)
        if dt.tzinfo is not None:
            dt = dt.astimezone()  # Convert to local timezone
        updated = dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        updated = updated_raw

    # Improvement (baseline is PyTorch reference latency; best is kernel latency)
    if best is not None and baseline is not None and baseline != 0 and best != 0:
        improv_pct = (baseline - best) / abs(baseline) * 100
        speedup = baseline / best
        color = GREEN if improv_pct > 0 else RED
        improv_str = f"{color}{speedup:.2f}x vs ref ({improv_pct:+.1f}%){RESET}"
    else:
        improv_str = f"{DIM}N/A{RESET}"

    frac = rounds / max_rounds if max_rounds > 0 else 0
    budget_bar = bar(frac)
    budget_color = GREEN if frac < 0.5 else (YELLOW if frac < 0.8 else RED)

    status_color = GREEN if status == "active" else (YELLOW if status == "no_plan" else CYAN)
    fail_color = RED if failures >= 3 else (YELLOW if failures > 0 else GREEN)

    lines.append("")
    lines.append(f"  {BOLD}Task:{RESET}     {task}")
    lines.append(f"  {BOLD}Status:{RESET}   {status_color}{status}{RESET}  (plan v{plan_ver})")
    lines.append(f"  {BOLD}Updated:{RESET}  {DIM}{updated}{RESET}")
    lines.append("")
    lines.append(f"  {BOLD}Budget:{RESET}   {budget_color}{budget_bar} {rounds}/{max_rounds}{RESET}")
    seed = progress.get("seed_metric")
    baseline_tags = {
        "ref": f"{DIM}(PyTorch reference){RESET}",
        "seed_fallback": f"{YELLOW}(fallback: seed — ref profile failed){RESET}",
    }
    baseline_tag = baseline_tags.get(progress.get("baseline_source"), f"{DIM}(source unknown){RESET}")
    lines.append(f"  {BOLD}Baseline:{RESET} {baseline}  {baseline_tag}")
    if seed is None:
        lines.append(f"  {BOLD}Seed:{RESET}     {RED}FAILED TO PROFILE{RESET}  "
                     f"{DIM}(seed kernel passed verify but produced no timing){RESET}")
    elif seed != baseline:
        lines.append(f"  {BOLD}Seed:{RESET}     {seed}  {DIM}(initial kernel){RESET}")
    lines.append(f"  {BOLD}Best:{RESET}     {GREEN}{best}{RESET}  ({improv_str})")
    lines.append(f"  {BOLD}Commit:{RESET}   {best_commit}")
    lines.append(f"  {BOLD}Failures:{RESET} {fail_color}{failures}{RESET} consecutive" +
                 (f"  {RED}⚠ DIAGNOSIS WILL TRIGGER{RESET}" if failures >= 3 else ""))

    # History table — windowed view
    n_total = len(history_all)
    if history_window is None:
        # Auto: reserve ~25 lines for header+plan, use rest for history
        try:
            term_h = os.get_terminal_size().lines
        except Exception:
            term_h = 40
        history_window = max(5, term_h - 28)

    # offset=0 means show latest; offset=5 means skip last 5 rounds
    history_offset = max(0, min(history_offset, max(0, n_total - history_window)))
    end = n_total - history_offset
    start = max(0, end - history_window)
    history = history_all[start:end]

    scroll_info = ""
    if n_total > history_window:
        scroll_info = f" [{start+1}-{end} of {n_total}, ↑↓ PgUp/PgDn Home/End q=quit]"

    lines.append("")
    lines.append(f"  {BOLD}History{RESET}{DIM}{scroll_info}{RESET}")
    lines.append(f"  {BOLD}{'─' * divider_width}{RESET}")
    lines.append(f"  {BOLD}  #  │ Decision │ Metric        │ Description{RESET}")
    lines.append(f"  {BOLD}{'─' * divider_width}{RESET}")

    for rec in history:
        rnd = rec.get("round")
        rnd = "?" if rnd is None else str(rnd)
        decision = rec.get("decision", "?")
        metrics = rec.get("metrics", {})
        raw_desc = rec.get("description", "")
        pid = rec.get("plan_item")
        # Prefix description with plan-item id when we have one, so every row
        # is unambiguously traceable back to plan.md. Older rounds (pre-fix)
        # may lack plan_item; render without the prefix.
        if pid:
            desc = f"{pid}: {raw_desc}"
        else:
            desc = raw_desc
        desc = _fit(desc, hist_desc_avail)

        # Find primary metric value. Limited to known performance keys -
        # earlier we fell back to `next(iter(metrics.values()))` when neither
        # was present, which on FAIL rows displayed `num_cases` (e.g. 72) as
        # if it were the kernel's latency. FAIL rows now show "—" cleanly.
        metric_val = "—"
        for k in ["latency_us", "score"]:
            if k in metrics and metrics[k] is not None:
                metric_val = f"{metrics[k]:.1f}" if isinstance(metrics[k], float) else str(metrics[k])
                break

        if decision == "KEEP":
            dec_str = f"{GREEN}  KEEP  {RESET}"
        elif decision == "DISCARD":
            dec_str = f"{YELLOW}DISCARD {RESET}"
        elif decision == "FAIL":
            dec_str = f"{RED}  FAIL  {RESET}"
        elif decision == "SEED":
            dec_str = f"{CYAN}  SEED  {RESET}"
        else:
            # Older history.jsonl files may carry deprecated decisions
            # (e.g. REACTIVATE, the now-removed pid-revival marker). Fall
            # through to the generic dim renderer rather than colour-code
            # them — the model never produces new ones.
            dec_str = f"{DIM}{decision:^8}{RESET}"

        lines.append(f"  {rnd:>3}  │ {dec_str} │ {metric_val:>13} │ {desc}")

    lines.append(f"  {BOLD}{'─' * divider_width}{RESET}")

    # Plan summary — structured table
    lines.append("")
    plan_age = ""
    if plan_mtime:
        age_sec = time.time() - plan_mtime
        if age_sec < 60:
            plan_age = f"{DIM}(updated {int(age_sec)}s ago){RESET}"
        else:
            plan_age = f"{DIM}(updated {int(age_sec/60)}m ago){RESET}"
    lines.append(f"  {BOLD}Current Plan:{RESET} {plan_age}")
    lines.append(f"  {BOLD}{'─' * divider_width}{RESET}")
    lines.append(f"  {BOLD}  #   │ Status    │ Description{RESET}")
    lines.append(f"  {BOLD}{'─' * divider_width}{RESET}")

    # Plan parsing goes through phase_machine.parse_plan_text — single
    # source of truth shared with hook validators. The dashboard's old
    # in-line regex drifted out of sync the moment plan.md format
    # changed; use the canonical parser instead.
    for item in _pm.parse_plan_text(plan_text):
        item_id = item["id"]
        is_active = item["active"]
        tag = item["tag"]
        # tag carries the leading bracket content like "KEEP, metric=..."
        # or "DISCARD" or "FAIL" — collapse to the keyword for display.
        outcome = ""
        for kw in ("KEEP", "DISCARD", "FAIL"):
            if tag.startswith(kw):
                outcome = kw
                break

        desc = _fit(item["description"], plan_desc_avail)

        if is_active:
            status_str = f"{CYAN}> ACTIVE {RESET}"
            desc_str = f"{CYAN}{desc}{RESET}"
        elif outcome == "KEEP":
            status_str = f"{GREEN}  KEEP   {RESET}"
            desc_str = f"{DIM}{desc}{RESET}"
        elif outcome == "DISCARD":
            status_str = f"{YELLOW} DISCARD {RESET}"
            desc_str = f"{DIM}{desc}{RESET}"
        elif outcome == "FAIL":
            status_str = f"{RED}  FAIL   {RESET}"
            desc_str = f"{DIM}{desc}{RESET}"
        else:
            status_str = " pending "  # 9 visible chars, matches other statuses
            desc_str = desc

        lines.append(f"  {item_id:>4}  │ {status_str}│ {desc_str}")

    lines.append(f"  {BOLD}{'─' * divider_width}{RESET}")

    lines.append("")
    lines.append(f"  {DIM}Press Ctrl+C to stop watching{RESET}")

    return "\n".join(lines)


def _auto_detect_task_dir() -> str:
    """Auto-detect task_dir: latest-modified ar_tasks/ dir wins.

    Uses file modification times (specifically .ar_state/progress.json or .phase)
    to pick the actively running task, not the one pointed to by stale .active_task.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Find all task dirs and pick the one with most recent activity
    tasks_dir = os.path.join(project_root, "ar_tasks")
    if os.path.isdir(tasks_dir):
        subdirs_with_mtime = []
        for d in os.listdir(tasks_dir):
            full = os.path.join(tasks_dir, d)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, "task.yaml")):
                # Prefer progress.json or .phase mtime (indicates active work),
                # fallback to dir mtime
                candidates = [
                    _pm.progress_path(full),
                    _pm.state_path(full, ".phase"),
                    full,
                ]
                latest_mtime = 0
                for c in candidates:
                    if os.path.exists(c):
                        latest_mtime = max(latest_mtime, os.path.getmtime(c))
                subdirs_with_mtime.append((full, latest_mtime))

        if subdirs_with_mtime:
            # Pick most recently modified
            subdirs_with_mtime.sort(key=lambda x: x[1], reverse=True)
            return subdirs_with_mtime[0][0]

    # Fallback: .active_task pointer written by hook_post_bash on activation
    active_file = os.path.join(project_root, ".autoresearch", ".active_task")
    if os.path.exists(active_file):
        with open(active_file, "r") as f:
            td = f.read().strip()
        if td and os.path.isdir(td):
            return td

    return ""


def main():
    parser = argparse.ArgumentParser(
        description="AutoResearch live dashboard. Auto-detects task if no path given.",
    )
    parser.add_argument("task_dir", nargs="?", default=None,
                        help="Path to task directory (auto-detected if omitted)")
    parser.add_argument("--watch", type=int, nargs="?", const=5, default=5,
                        help="Refresh interval in seconds (default: 5, use 0 for one-shot)")
    args = parser.parse_args()

    if args.task_dir:
        task_dir = os.path.abspath(args.task_dir)
    else:
        task_dir = _auto_detect_task_dir()
        if not task_dir:
            print("No task found. Pass a task_dir or start /autoresearch first.")
            sys.exit(1)
        print(f"Auto-detected: {task_dir}", file=sys.stderr)

    # Force UTF-8 output on Windows + enable ANSI escape codes
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        # Enable ANSI escapes on Windows 10+
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

    if args.watch and args.watch > 0:
        def clear_screen():
            # Use native clear (reliably clears both visible screen + scrollback)
            os.system("cls" if sys.platform == "win32" else "clear")
        history_offset = 0
        last_render = 0

        # Detect if stdin is a real TTY; if not, fallback to pure refresh (no keyboard)
        interactive = False
        try:
            interactive = sys.stdin.isatty()
        except Exception:
            interactive = False

        if interactive:
            try:
                setup_keyboard()
            except Exception:
                interactive = False

        try:
            while True:
                now = time.time()
                needs_render = False

                # Keyboard handling (only if interactive)
                if interactive:
                    try:
                        key = read_key_nonblocking()
                    except Exception:
                        key = None
                    if key == "QUIT" or key == "ESC":
                        break
                    elif key == "UP":
                        history_offset += 1
                        needs_render = True
                    elif key == "DOWN":
                        history_offset = max(0, history_offset - 1)
                        needs_render = True
                    elif key == "PGUP":
                        history_offset += 10
                        needs_render = True
                    elif key == "PGDN":
                        history_offset = max(0, history_offset - 10)
                        needs_render = True
                    elif key == "HOME":
                        history_offset = 999999
                        needs_render = True
                    elif key == "END":
                        history_offset = 0
                        needs_render = True

                # Auto-refresh
                if now - last_render >= args.watch:
                    needs_render = True

                if needs_render:
                    clear_screen()
                    print(render(task_dir, history_offset=history_offset), flush=True)
                    last_render = now

                time.sleep(0.1 if interactive else max(0.5, args.watch / 2))
        except KeyboardInterrupt:
            pass
        finally:
            if interactive:
                try:
                    restore_keyboard()
                except Exception:
                    pass
            print(f"\n{DIM}Dashboard stopped.{RESET}")
    else:
        print(render(task_dir))


if __name__ == "__main__":
    main()
