#!/usr/bin/env python3
"""session_anchor.py — Session identity anchor file manager.

Prevents the ops-evo agent from losing track of its output
directory when context compression truncates the TIMESTAMP variable.

Usage:
  # At evolution start (step 3 init)
  python3 .claude/skills/evolution-world-model/scripts/session_anchor.py \
    write \
    --op-name ai_infra_sparse_flash_attention_gqa \
    --evo-dir $(pwd)/output/ai_infra_sparse_flash_attention_gqa_ops-evo_20260430_202933 \
    --requested-rounds 5

  # At any later step (step 5 report, etc.)
  python3 .claude/skills/evolution-world-model/scripts/session_anchor.py \
    read \
    --op-name ai_infra_sparse_flash_attention_gqa

  # Clear anchor after evolution completes
  python3 .claude/skills/evolution-world-model/scripts/session_anchor.py \
    clear \
    --op-name ai_infra_sparse_flash_attention_gqa

  # Verify a candidate directory matches the anchor
  python3 .claude/skills/evolution-world-model/scripts/session_anchor.py \
    verify \
    --op-name ai_infra_sparse_flash_attention_gqa \
    --evo-dir $(pwd)/output/ai_infra_sparse_flash_attention_gqa_ops-evo_20260430_202933

The anchor file is written to:
  output/.ops-evo_current_session_{op_name}.json

This is a sibling of the per-operator evo directories so that it survives
cleanups of individual evo directories but is scoped to the operator name.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta


def _anchor_path(op_name: str, output_dir: str = "output") -> str:
    return os.path.join(output_dir, f".ops-evo_current_session_{op_name}.json")


def cmd_write(args: argparse.Namespace) -> None:
    """Write a new session anchor."""
    evo_dir = os.path.abspath(args.evo_dir)
    if not os.path.isdir(evo_dir):
        print(f"session-anchor write: FATAL — evo_dir does not exist: {evo_dir}",
              file=sys.stderr)
        sys.exit(1)

    now = datetime.now(timezone(timedelta(hours=8)))
    anchor = {
        "session_id": f"{args.op_name}_ops-evo_{now.strftime('%Y%m%d_%H%M%S')}",
        "start_time": now.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "evo_dir": evo_dir,
        "op_name": args.op_name,
        "requested_rounds": args.requested_rounds,
        "actual_rounds_completed": 0,
        "pid": os.getpid(),
    }

    path = _anchor_path(args.op_name, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(anchor, f, ensure_ascii=False, indent=2)

    print(f"session-anchor: written to {path}")
    print(f"  session_id={anchor['session_id']}")
    print(f"  evo_dir={evo_dir}")
    print(f"  requested_rounds={args.requested_rounds}")


def cmd_read(args: argparse.Namespace) -> None:
    """Read the current session anchor and print key fields."""
    path = _anchor_path(args.op_name, args.output_dir)
    if not os.path.isfile(path):
        print(f"session-anchor read: FATAL — no anchor file found at {path}",
              file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        anchor = json.load(f)

    # Check if the PID that wrote the anchor is still alive
    writer_pid = anchor.get("pid")
    pid_alive = False
    if writer_pid:
        try:
            os.kill(writer_pid, 0)
            pid_alive = True
        except (OSError, ProcessLookupError):
            pid_alive = False

    print(json.dumps({
        "session_id": anchor.get("session_id"),
        "evo_dir": anchor.get("evo_dir"),
        "requested_rounds": anchor.get("requested_rounds"),
        "actual_rounds_completed": anchor.get("actual_rounds_completed"),
        "start_time": anchor.get("start_time"),
        "writer_pid": writer_pid,
        "writer_pid_alive": pid_alive,
    }, ensure_ascii=False, indent=2))


def cmd_clear(args: argparse.Namespace) -> None:
    """Remove the session anchor file."""
    path = _anchor_path(args.op_name, args.output_dir)
    if os.path.isfile(path):
        os.remove(path)
        print(f"session-anchor: cleared {path}")
    else:
        print(f"session-anchor: nothing to clear (no file at {path})")


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify that a candidate directory matches the session anchor."""
    path = _anchor_path(args.op_name, args.output_dir)
    if not os.path.isfile(path):
        print(f"session-anchor verify: FATAL — no anchor file found at {path}",
              file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        anchor = json.load(f)

    expected_dir = os.path.abspath(anchor.get("evo_dir", ""))
    actual_dir = os.path.abspath(args.evo_dir)

    if expected_dir != actual_dir:
        print(
            f"session-anchor verify: FATAL — directory mismatch\n"
            f"  expected (from anchor): {expected_dir}\n"
            f"  actual (provided):      {actual_dir}",
            file=sys.stderr,
        )
        sys.exit(2)

    actual_rounds = anchor.get("actual_rounds_completed", 0)
    requested_rounds = anchor.get("requested_rounds", 0)
    if actual_rounds < requested_rounds:
        print(
            f"[WARNING] session-anchor verify: WARNING — only {actual_rounds}/{requested_rounds} "
            f"rounds completed. Report must clearly state this.",
            file=sys.stderr,
        )

    print(f"session-anchor verify: OK — {actual_dir} matches anchor")
    print(f"  rounds: {actual_rounds}/{requested_rounds} completed")


def cmd_update(args: argparse.Namespace) -> None:
    """Update actual_rounds_completed in the anchor file."""
    path = _anchor_path(args.op_name, args.output_dir)
    if not os.path.isfile(path):
        print(f"session-anchor update: FATAL — no anchor file found at {path}",
              file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        anchor = json.load(f)

    anchor["actual_rounds_completed"] = max(
        anchor.get("actual_rounds_completed", 0),
        args.actual_rounds
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(anchor, f, ensure_ascii=False, indent=2)

    print(f"session-anchor update: actual_rounds_completed={anchor['actual_rounds_completed']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session identity anchor file manager for CAKE3 evolution."
    )
    parser.add_argument("--output-dir", default="output",
                        help="Base output directory (default: output)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_write = subparsers.add_parser("write", help="Write a new session anchor")
    p_write.add_argument("--op-name", required=True)
    p_write.add_argument("--evo-dir", required=True)
    p_write.add_argument("--requested-rounds", type=int, default=5)
    p_write.set_defaults(func=cmd_write)

    p_read = subparsers.add_parser("read", help="Read the current session anchor")
    p_read.add_argument("--op-name", required=True)
    p_read.set_defaults(func=cmd_read)

    p_clear = subparsers.add_parser("clear", help="Clear the session anchor")
    p_clear.add_argument("--op-name", required=True)
    p_clear.set_defaults(func=cmd_clear)

    p_verify = subparsers.add_parser("verify", help="Verify evo_dir matches anchor")
    p_verify.add_argument("--op-name", required=True)
    p_verify.add_argument("--evo-dir", required=True)
    p_verify.set_defaults(func=cmd_verify)

    p_update = subparsers.add_parser("update", help="Update actual_rounds_completed")
    p_update.add_argument("--op-name", required=True)
    p_update.add_argument("--actual-rounds", type=int, required=True)
    p_update.set_defaults(func=cmd_update)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
