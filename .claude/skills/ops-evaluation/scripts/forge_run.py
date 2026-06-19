#!/usr/bin/env python3
"""Unified entry point for forge evaluation — wraps resolver + evaluator."""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_soc() -> str:
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"npu-smi not available: {exc}") from exc
    for line in result.stdout.splitlines():
        match = re.search(r"\b(910B\w*)\b", line, re.IGNORECASE)
        if match:
            return "ascend" + match.group(1).lower()
    raise RuntimeError(f"Could not detect SoC from npu-smi output:\n{result.stdout}")


def detect_repo_type(repo_root: str) -> str:
    build_sh = Path(repo_root) / "build.sh"
    if not build_sh.is_file():
        raise FileNotFoundError(f"build.sh not found in {repo_root}")
    content = build_sh.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'REPOSITORY_NAME\s*=\s*["\']?(\S+?)["\']?\s', content)
    if match:
        return "omni-ops"
    return "omni-ops"


def resolve_op_path(repo_root: str, op_name: str) -> str:
    bare_name = op_name
    if bare_name.endswith("_custom"):
        bare_name = bare_name[:-len("_custom")]
    root = Path(repo_root)
    # 递归搜索多层目录 (支持 omni-ops 的 src/ops-transformer/attention/ 嵌套结构)
    for candidate in root.rglob(bare_name):
        if candidate.is_dir() and (candidate / "op_host").is_dir():
            rel = candidate.relative_to(root)
            return str(rel)
    # fallback: 只在直接子目录搜索
    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        candidate = category_dir / bare_name
        if candidate.is_dir():
            return f"{category_dir.name}/{bare_name}"
    searched = [str(d.relative_to(root)) for d in root.rglob("*") if d.is_dir() and bare_name in d.name]
    raise FileNotFoundError(
        f"Operator directory for '{bare_name}' not found under {repo_root}. "
        f"Searched {len(searched)} candidates: {searched[:20]}"
    )


def derive_shared_dir(install_path: str) -> str:
    p = Path(install_path)
    for ancestor in [p.parent, p.parent.parent, p.parent.parent.parent]:
        candidate = ancestor / "shared"
        if candidate.is_dir():
            return str(candidate)
    return str(p.parent / "shared")


def derive_vendor_name(install_path: str, repo_root: str = "") -> str | None:
    """推导 vendors 子目录名。优先扫描 install_path 实际目录，否则从 repo_root 路径推断。"""
    vendors_dir = Path(install_path) / "vendors"
    if vendors_dir.is_dir():
        subdirs = [d for d in vendors_dir.iterdir() if d.is_dir()]
        for d in subdirs:
            if d.name.startswith("omni_"):
                return d.name
        for d in subdirs:
            if d.name.startswith("custom"):
                return d.name
        return subdirs[0].name if subdirs else None

    # fallback: 从 repo_root 路径推断 vendor 名
    if repo_root:
        root_lower = repo_root.lower().replace("\\", "/")
        if "/training/" in root_lower or root_lower.endswith("/training"):
            return "omni_training_custom_ops"
        if "/inference/" in root_lower or root_lower.endswith("/inference"):
            return "omni_custom_transformer"
        # 再尝试从 build.sh 中的 REPOSITORY_NAME 推断
        build_sh = Path(repo_root) / "build.sh"
        if build_sh.is_file():
            content = build_sh.read_text(encoding="utf-8", errors="replace")
            if 'REPOSITORY_NAME="transformer"' in content:
                return "omni_custom_transformer"
            if "training" in content.lower():
                return "omni_training_custom_ops"
    return None


def derive_set_env_path(install_path: str, repo_root: str = "") -> str | None:
    vendor = derive_vendor_name(install_path, repo_root)
    if not vendor:
        return None
    p = Path(install_path) / "vendors" / vendor / "bin" / "set_env.bash"
    # 如果 vendors 目录已存在，则校验文件是否存在；否则直接返回推导路径（build 前使用）
    if (Path(install_path) / "vendors").is_dir():
        return str(p) if p.is_file() else None
    return str(p)


def derive_build_artifact(repo_root: str) -> str | None:
    output_dir = Path(repo_root) / "output"
    if not output_dir.is_dir():
        return None
    runs = sorted(output_dir.glob("*.run"), key=lambda p: p.stat().st_mtime, reverse=True)
    if runs:
        return str(runs[0].relative_to(repo_root))
    return None


def derive_forge_config_dir(forge_bin: str = "forge") -> str | None:
    """自动推导 forge 配置目录，无需硬编码路径。

    策略优先级：
    1. 通过 importlib 找到 forge 包路径，取父目录下的 configs/
    2. 通过 which 找到 forge 可执行文件，向上查找包含 configs/ 的目录
    3. fallback 到几个常见默认路径
    """
    # Strategy 1: locate via installed Python package
    try:
        import importlib.util
        spec = importlib.util.find_spec("forge")
        if spec and spec.origin:
            pkg_root = Path(spec.origin).resolve().parent.parent
            candidate = pkg_root / "configs"
            if candidate.is_dir():
                return str(candidate)
    except Exception:
        pass

    # Strategy 2: resolve forge binary and walk upward
    try:
        result = subprocess.run(
            ["which", forge_bin],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            bin_path = Path(result.stdout.strip()).resolve()
            for ancestor in [bin_path.parent, bin_path.parent.parent, bin_path.parent.parent.parent]:
                candidate = ancestor / "configs"
                if candidate.is_dir():
                    return str(candidate)
    except Exception:
        pass

    # Strategy 3: fallback defaults (env override via FORGE_CONFIG_FALLBACKS=path1:path2)
    default_fallbacks = ["/mnt/workspace/forge/configs"]
    extra = os.environ.get("FORGE_CONFIG_FALLBACKS", "")
    for p in extra.split(":"):
        p = p.strip()
        if p:
            default_fallbacks.append(p)
    for fallback in default_fallbacks:
        if Path(fallback).is_dir():
            return fallback

    return None


def build_evaluator_command(
    *,
    forge_config: str,
    forge_config_dir: str,
    forge_bin: str,
    op_name: str,
    repo_root: str,
    install_path: str,
    test_path: str,
    test_script: str,
    zsearch_side: str,
    zsearch_build_success: str,
    zsearch_repo_type: str,
    zsearch_soc: str,
    zsearch_task_type: str,
    zsearch_precision_passed: str,
    zsearch_correctness_message: str,
    baseline_time_us: float,
    output: str,
    mode: str,
    build_artifact: str | None = None,
    set_env_path: str | None = None,
) -> list[str]:
    evaluator_script = str(Path(__file__).parent / "forge_evaluator.py")
    cmd = [
        sys.executable, evaluator_script,
        "--forge-config", forge_config,
        "--forge-config-dir", forge_config_dir,
        "--forge-bin", forge_bin,
        "--op-name", op_name,
        "--repo-root", repo_root,
        "--install-path", install_path,
        "--test-path", test_path,
        "--test-script", test_script,
        "--zsearch-side", zsearch_side,
        "--zsearch-build-success", zsearch_build_success,
        "--zsearch-repo-type", zsearch_repo_type,
        "--zsearch-soc", zsearch_soc,
        "--zsearch-task-type", zsearch_task_type,
        "--zsearch-precision-passed", zsearch_precision_passed,
        "--zsearch-correctness-message", zsearch_correctness_message,
        "--baseline-time-us", str(baseline_time_us),
        "--output", output,
        "--mode", mode,
    ]
    if build_artifact:
        cmd.extend(["--build-artifact", build_artifact])
    if set_env_path:
        cmd.extend(["--set-env-path", set_env_path])
    return cmd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Unified forge evaluation entry point")
    parser.add_argument("--op-name", required=True, help="Operator name (e.g. ada_layer_norm_custom)")
    parser.add_argument("--repo-root", required=True, help="Ops repo root directory")
    parser.add_argument("--install-path", required=True, help="Install artifact path")
    parser.add_argument("--mode", required=True, choices=["build", "accuracy", "perf", "both"])
    parser.add_argument("--output", required=True, help="Output path for evaluation_results.json")
    parser.add_argument("--baseline-time-us", type=float, default=0.0)
    parser.add_argument("--zsearch-side", default="evolved", choices=["baseline", "evolved"])
    parser.add_argument("--forge-config", default="omni_ops_performance_pytest")
    parser.add_argument("--forge-config-dir", default=None, help="Forge configs directory (auto-detected if omitted)")
    parser.add_argument("--forge-bin", default="forge")
    parser.add_argument("--soc", default=None)
    parser.add_argument("--shared-dir", default=None)
    parser.add_argument("--generated-test-script", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    soc = args.soc or detect_soc()
    forge_config_dir = args.forge_config_dir or derive_forge_config_dir(args.forge_bin)
    if not forge_config_dir:
        logger.error("Could not auto-detect forge config directory. Please pass --forge-config-dir explicitly.")
        return 1
    logger.info("Using forge config dir: %s", forge_config_dir)

    repo_type = detect_repo_type(args.repo_root)
    op_relative_path = resolve_op_path(args.repo_root, args.op_name)
    shared_dir = args.shared_dir or derive_shared_dir(args.install_path)
    generated_test_script = args.generated_test_script or f"{args.op_name}_custom.py"

    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))
    from forge_test_resolver import resolve_forge_test_script

    resolver_result = resolve_forge_test_script(
        repo_root=args.repo_root,
        op_relative_path=op_relative_path,
        custom_op_name=args.op_name,
        generated_test_path=shared_dir,
        generated_test_script=generated_test_script,
    )
    logger.info("Resolver result: %s", json.dumps(resolver_result))

    build_artifact = derive_build_artifact(args.repo_root)
    set_env_path = derive_set_env_path(args.install_path, args.repo_root)
    if build_artifact:
        logger.info("Detected build artifact: %s", build_artifact)
    if set_env_path:
        logger.info("Detected set_env.bash: %s", set_env_path)

    cmd = build_evaluator_command(
        forge_config=args.forge_config,
        forge_config_dir=forge_config_dir,
        forge_bin=args.forge_bin,
        op_name=args.op_name,
        repo_root=args.repo_root,
        install_path=args.install_path,
        test_path=resolver_result["test_path"],
        test_script=resolver_result["test_script"],
        zsearch_side=args.zsearch_side,
        zsearch_build_success="true",
        zsearch_repo_type=repo_type,
        zsearch_soc=soc,
        zsearch_task_type="performance",
        zsearch_precision_passed="true",
        zsearch_correctness_message="PASS",
        baseline_time_us=args.baseline_time_us,
        output=args.output,
        mode=args.mode,
        build_artifact=build_artifact,
        set_env_path=set_env_path,
    )

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
