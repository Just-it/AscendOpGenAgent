#!/usr/bin/env python3
"""Deterministic argument parser for /autoresearch.

This script is the single source of truth for the slash command's args.
It eats the user's raw `$ARGUMENTS`, decides which mode applies (resume /
scaffold / ask), validates required fields, and emits a JSON dispatch
record. The slash command tells the LLM to run only the `command` field
verbatim and to read flag values only from the `values` field — no
inventing values, no pulling defaults from docstrings, no paraphrasing.

The previous architecture handed `$ARGUMENTS` straight into the LLM as
prose context and asked it to construct the bash itself, which let the
LLM rewrite or substitute flag values on retries (e.g. quietly turning
`--devices 6` into `--devices 0` after a hook block). Putting an argparse
between the user and the LLM closes that drift.

Modes:
  resume     — `--resume [task_dir]` or a bare existing task path
  scaffold   — init flags (--ref/--desc + --op-name + --devices)
  ask        — empty args, or scaffold flags incomplete

Output (single JSON line on stdout):
  {"mode": "scaffold|resume|ask",
   "command": "python ... (verbatim, ready to exec)" | null,
   "values":  {parsed flag dict — ground truth for the LLM},
   "missing": [human-readable required fields, ask-mode only]}

Exit code is always 0; errors surface inside the JSON. Failing on the
shell side would force the LLM to guess what happened, which is exactly
what this script exists to prevent.
"""
import json
import os
import shlex
import sys


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False))
    sys.exit(0)


def _build_scaffold_command(args) -> str:
    """Reconstruct an exec-ready scaffold invocation from parsed args.

    Every value comes from the argparse Namespace, never from the raw
    input — that's the whole point: once argparse has accepted the args,
    the canonical form is what scaffold sees, regardless of any quoting
    or whitespace quirks in the user's typed string.
    """
    parts = ["python", ".autoresearch/scripts/scaffold.py"]
    if args.ref:
        parts += ["--ref", shlex.quote(args.ref)]
    if args.desc:
        parts += ["--desc", shlex.quote(args.desc)]
    if args.kernel:
        parts += ["--kernel", shlex.quote(args.kernel)]
    if args.op_name:
        parts += ["--op-name", shlex.quote(args.op_name)]
    if args.devices:
        parts += ["--devices", str(args.devices)]
    parts += ["--max-rounds", str(args.max_rounds)]
    parts += ["--eval-timeout", str(args.eval_timeout)]
    parts += ["--output-dir", args.output_dir or "ar_tasks"]
    parts.append("--run-baseline")
    if args.no_code_checker:
        parts.append("--no-code-checker")
    return " ".join(parts)


def main():
    tokens = sys.argv[1:]

    # --- empty: ask mode ---
    if not tokens:
        _emit({
            "mode": "ask",
            "command": None,
            "values": {},
            "missing": [
                "--ref <file> or --desc \"...\"",
                "--op-name <name>",
                "--devices <N>",
                "--max-rounds (optional, default 20)",
            ],
            "note": ("no arguments — ask the user for the missing fields, "
                     "then re-invoke /autoresearch with the full flag set."),
        })

    # --- resume forms ---
    if tokens[0] == "--resume":
        task_dir = tokens[1] if len(tokens) > 1 else ""
        cmd_parts = ["python", ".autoresearch/scripts/resume.py"]
        if task_dir:
            cmd_parts.append(shlex.quote(task_dir))
        _emit({
            "mode": "resume",
            "command": " ".join(cmd_parts),
            "values": {"task_dir": task_dir or None},
            "missing": [],
        })

    # bare path → resume that task
    if not tokens[0].startswith("--"):
        path = tokens[0]
        if not os.path.isdir(path):
            _emit({
                "mode": "ask",
                "command": None,
                "values": {"first_token": path},
                "missing": [f"first token {path!r} is neither a flag nor an "
                            f"existing directory — clarify with the user "
                            f"before re-invoking /autoresearch."],
            })
        _emit({
            "mode": "resume",
            "command": f"python .autoresearch/scripts/resume.py {shlex.quote(path)}",
            "values": {"task_dir": path},
            "missing": [],
        })

    # --- scaffold form ---
    # Reuse scaffold's parser so flag spec stays in lockstep.
    from scaffold import _make_arg_parser
    parser = _make_arg_parser()

    # Capture argparse's own error path (it normally prints to stderr and
    # sys.exit(2)) and convert into a structured ask-mode payload so the LLM
    # gets a JSON to react to instead of a stderr message.
    class _CapturedExit(Exception):
        def __init__(self, msg):
            self.msg = msg

    def _err(msg):
        raise _CapturedExit(msg)

    parser.error = _err  # type: ignore[assignment]

    try:
        args = parser.parse_args(tokens)
    except _CapturedExit as e:
        _emit({
            "mode": "ask",
            "command": None,
            "values": {"raw_tokens": tokens},
            "missing": [f"argparse rejected the args: {e.msg}"],
        })

    # Workflow-level required fields. argparse already enforces --ref XOR
    # --desc; the rest we check here so the LLM gets a single error list.
    missing = []
    if not args.op_name and not args.desc:
        # op_name is auto-derivable from --desc inside scaffold but not
        # from --ref alone — explicitly require it whenever --ref is used.
        missing.append("--op-name <name>")
    if not args.devices:
        missing.append("--devices <N>")

    values = {
        "ref": args.ref,
        "desc": args.desc,
        "kernel": args.kernel,
        "op_name": args.op_name,
        "devices": args.devices,
        "max_rounds": args.max_rounds,
        "eval_timeout": args.eval_timeout,
        "output_dir": args.output_dir or "ar_tasks",
        "run_baseline": True,
        "no_code_checker": args.no_code_checker,
    }

    if missing:
        _emit({
            "mode": "ask",
            "command": None,
            "values": values,
            "missing": missing,
        })

    _emit({
        "mode": "scaffold",
        "command": _build_scaffold_command(args),
        "values": values,
        "missing": [],
    })


if __name__ == "__main__":
    main()
