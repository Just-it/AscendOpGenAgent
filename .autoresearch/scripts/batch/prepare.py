"""One-shot preparation step: discover ops + verify Tier 1.

Combines two mechanical pre-flight steps that always run together when
seeding a batch dir:

  1. Scan refs/ + kernels/ for the <op>_ref.py / <op>_kernel.py naming
     convention; write/update manifest.yaml's ops list.
  2. For every discovered op, compile the file, import the module, and
     check the required exports (Model / get_inputs / get_init_inputs in
     ref; ModelNew in kernel) are present. Per-op subprocess isolation —
     a missing dependency in one op doesn't poison the others.

This is the only step where merging makes sense. Everything else (run,
monitor, summarize) involves user decisions and stays as separate
commands.

Usage:
    python .autoresearch/scripts/batch/prepare.py <batch_dir> --mode ref-kernel
    python .autoresearch/scripts/batch/prepare.py <batch_dir>
        # re-run after adding/removing files; inherits mode from manifest

Flags mirror discover.py (filter / exclude / dirs) and verify.py (only).
Exits 0 only if both steps pass; on discover failure verify is skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import discover
import manifest as mf
import verify


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prepare a batch dir: discover ops + verify Tier 1.",
    )
    ap.add_argument("batch_dir")
    ap.add_argument("--mode", choices=mf.VALID_MODES,
                    help="ref-kernel or ref (overrides existing manifest)")
    ap.add_argument("--ref-dir", default="",
                    help="ref subdirectory (default: from manifest, else 'refs')")
    ap.add_argument("--kernel-dir", default="",
                    help="kernel subdirectory (default: from manifest, else 'kernels')")
    ap.add_argument("--filter", default="",
                    help="glob to KEEP only matching op names (e.g. '*norm')")
    ap.add_argument("--exclude", action="append", default=[],
                    help="glob(s) to drop matching op names; repeatable")
    ap.add_argument("--only", default="",
                    help="restrict the verify step to comma-separated op names "
                         "(does not affect what gets written to the manifest)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="run discover only; don't invoke Tier 1 verify")
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        sys.exit(f"batch dir not found: {batch_dir}")

    # ---- Step 1: discover -------------------------------------------------
    print(f"[prepare 1/2] discover  batch_dir={batch_dir}")

    existing: dict = {}
    try:
        manifest_path = mf.find_manifest(batch_dir)
        existing = mf.load_manifest(manifest_path)
    except mf.ManifestError:
        pass

    mode = args.mode or existing.get("mode")
    if not mode:
        sys.exit("--mode required (no existing manifest to inherit from)")
    if mode not in mf.VALID_MODES:
        sys.exit(f"--mode must be one of {mf.VALID_MODES}, got {mode!r}")

    ref_dir = args.ref_dir or existing.get("ref_dir") or "refs"
    kernel_dir = args.kernel_dir or existing.get("kernel_dir") or "kernels"
    kernel_dir_for_scan = kernel_dir if mode == "ref-kernel" else None

    ops = discover.discover(
        batch_dir, mode, ref_dir, kernel_dir_for_scan,
        include_glob=args.filter or None,
        exclude_globs=list(args.exclude),
    )
    if not ops:
        sys.exit("no ops discovered. Expected files matching "
                 "<op_name>_ref.py / <op_name>_kernel.py in the configured "
                 "ref_dir / kernel_dir.")

    target = discover.write_manifest(
        batch_dir, mode, ref_dir,
        kernel_dir if mode == "ref-kernel" else None,
        ops,
    )
    print(f"  wrote {len(ops)} ops to {target.name}")
    for op in ops:
        print(f"  - {op}")

    if args.skip_verify:
        print("\n[prepare 2/2] verify Tier 1: skipped (--skip-verify)")
        return 0

    # ---- Step 2: verify Tier 1 -------------------------------------------
    print(f"\n[prepare 2/2] verify Tier 1")
    rc = verify.run_verification(
        batch_dir, mode=mode, full=False, only=args.only,
    )
    if rc == 0:
        print("\n[prepare] all checks passed; batch dir is ready to run.")
    else:
        print("\n[prepare] verify Tier 1 reported failures; "
              "fix the offending files and re-run prepare.py before "
              "starting the batch.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
