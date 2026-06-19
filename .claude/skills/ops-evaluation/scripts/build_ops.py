#!/usr/bin/env python3
"""
ops仓算子构建+安装封装脚本。

封装 ops-nn/cv/math/transformer 仓及 omni-ops 仓的 build.sh 构建流程，
执行编译并将生成的 .run 文件安装到指定路径。
自动检测仓类型并使用对应的构建命令。

用法:
    python build_ops.py --repo-root /path/to/ops-nn --op-name ada_layer_norm_custom --soc ascend910b --install-path /abs/path/to/install

参数:
    --repo-root: ops仓根目录（如 /home/user/ops-nn）
    --op-name: 算子名（如 ada_layer_norm_custom，带 _custom 后缀）
    --soc: 目标芯片（如 ascend910b, ascend950）
    --install-path: 安装目标路径（必须是绝对路径）

依赖:
    - ops仓根目录下需要有 build.sh
    - ASCEND_HOME_PATH 环境变量需要设置
"""

import argparse
import logging
import os
import platform
import re
import shutil
import subprocess
import sys


def detect_cpu_arch() -> str:
    """检测CPU架构。"""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    elif machine in ("aarch64", "arm64"):
        return "aarch64"
    else:
        logging.warning(f"未知CPU架构: {machine}, 默认使用aarch64")
        return "aarch64"


def detect_repo_type(repo_root: str) -> str:
    """
    通过 build.sh 中的 REPOSITORY_NAME 或目录名检测仓类型。

    Returns:
        "nn" / "cv" / "math" / "transformer" / "omni"
    """
    build_sh = os.path.join(repo_root, "build.sh")
    if os.path.exists(build_sh):
        try:
            with open(build_sh, "r") as f:
                content = f.read()

            # omni-ops 特征检测（优先于标准仓检测）:
            # omni-ops 的 build.sh 使用 -n/--op-name 参数风格，
            # 且存在 src/ 目录（如 src/ops-transformer/）
            if (re.search(r'(-n\b|--op-name)', content)
                    and os.path.isdir(os.path.join(repo_root, "src"))):
                return "omni"

            # 尝试匹配 REPOSITORY_NAME 变量
            match = re.search(r'REPOSITORY_NAME\s*=\s*["\']?(\w+)', content)
            if match:
                name = match.group(1).lower()
                if "nn" in name:
                    return "nn"
                elif "cv" in name:
                    return "cv"
                elif "math" in name:
                    return "math"
                elif "transformer" in name:
                    return "transformer"
        except Exception as e:
            logging.warning(f"读取 build.sh 失败: {e}")

    # 回退: 通过目录名判断
    basename = os.path.basename(os.path.normpath(repo_root)).lower()
    if "omni" in basename:
        return "omni"
    elif "nn" in basename:
        return "nn"
    elif "cv" in basename:
        return "cv"
    elif "math" in basename:
        return "math"
    elif "transformer" in basename:
        return "transformer"

    logging.warning(f"无法检测仓类型, 默认 nn")
    return "nn"


def detect_is_omni(repo_root: str) -> bool:
    """检测是否为 omni-ops 仓。

    omni-ops 仓的特征：build.sh 支持 -n/--op-name 参数，且有 src/ 目录。
    """
    build_sh = os.path.join(repo_root, "build.sh")
    has_n_flag = False
    has_src = os.path.isdir(os.path.join(repo_root, "src"))
    if os.path.exists(build_sh):
        with open(build_sh, "r") as f:
            content = f.read()
        has_n_flag = bool(re.search(r'(-n\b|--op-name)', content))
    return has_n_flag and has_src


def detect_vendor_subdir(install_path: str) -> str:
    """
    检测安装后的 vendors 子目录名。

    根据 .run 安装后的实际目录结构确定:
    - ops-nn: vendors/custom_nn
    - ops-cv: vendors/custom_cv
    - 等等

    Returns:
        vendors子目录名 (如 "custom_nn")
    """
    vendors_dir = os.path.join(install_path, "vendors")
    if os.path.isdir(vendors_dir):
        subdirs = [d for d in os.listdir(vendors_dir)
                    if os.path.isdir(os.path.join(vendors_dir, d))
                    and (d.startswith("custom") or d.startswith("omni_custom"))]
        if subdirs:
            return subdirs[0]
    # 回退: 尝试 customize（与标准 ascendc-evaluation 一致）
    if os.path.isdir(os.path.join(vendors_dir, "customize")):
        return "customize"
    return "custom_nn"


def find_run_file(repo_root: str, is_omni: bool) -> str:
    """
    查找构建产物中的 .run 安装文件。

    omni-ops 仓的 .run 在 build/_CPack_Packages/ 下，
    标准仓的 .run 在 build_out/ 下。

    Args:
        repo_root: 仓根目录
        is_omni: 是否为 omni-ops 仓

    Returns:
        .run 文件的完整路径
    Raises:
        FileNotFoundError: 未找到 .run 文件
    """
    build_out_dir = os.path.join(repo_root, "build_out")
    cpack_dir = os.path.join(repo_root, "build", "_CPack_Packages")
    output_dir = os.path.join(repo_root, "output")

    if is_omni:
        search_dirs = [cpack_dir, output_dir, build_out_dir]
    else:
        search_dirs = [build_out_dir, output_dir]

    for base_dir in search_dirs:
        if not os.path.isdir(base_dir):
            continue
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f.endswith(".run"):
                    return os.path.join(root, f)

    searched = ", ".join(search_dirs)
    raise FileNotFoundError(
        f"未在以下目录中找到 .run 文件: {searched}。"
        f" 请检查构建是否成功完成。"
    )


def build_and_install(repo_root: str, op_name: str, soc: str,
                      install_path: str) -> dict:
    """
    构建并安装 ops 仓算子。

    Args:
        repo_root: ops仓根目录
        op_name: 算子名（如 ada_layer_norm_custom）
        soc: 目标芯片（如 ascend910b）
        install_path: 安装路径（绝对路径）

    Returns:
        dict: {
            "install_path": str,
            "vendor_subdir": str,
            "repo_type": str,
            "run_file": str,
        }

    Raises:
        RuntimeError: 构建或安装失败
    """
    # 验证输入
    if not os.path.isabs(install_path):
        raise ValueError(
            f"install_path 必须是绝对路径, 当前值: {install_path}"
        )

    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"ops仓根目录不存在: {repo_root}")

    build_sh = os.path.join(repo_root, "build.sh")
    if not os.path.isfile(build_sh):
        raise FileNotFoundError(f"build.sh 不存在: {build_sh}")

    repo_type = detect_repo_type(repo_root)
    is_omni = detect_is_omni(repo_root)
    cpu_arch = detect_cpu_arch()
    nproc = os.cpu_count() or 8

    logging.info(f"构建配置:")
    logging.info(f"  仓类型: {repo_type}")
    logging.info(f"  omni-ops: {is_omni}")
    logging.info(f"  CPU架构: {cpu_arch}")
    logging.info(f"  算子名: {op_name}")
    logging.info(f"  目标芯片: {soc}")
    logging.info(f"  安装路径: {install_path}")

    # Step 1: 清理旧的构建产物
    build_dir = os.path.join(repo_root, "build")
    build_out_dir = os.path.join(repo_root, "build_out")
    for d in [build_dir, build_out_dir]:
        if os.path.exists(d):
            logging.info(f"清理目录: {d}")
            subprocess.run(["rm", "-rf", d], check=True)

    # Step 2: 执行构建
    # omni-ops 仓使用 -n 参数指定算子名，标准仓使用 --pkg --ops= 参数
    if is_omni:
        # omni-ops build.sh 内部自动计算 JOB_NUM，不接受 -j 参数
        build_cmd = f"bash build.sh -n \"{op_name}\" -c {soc}"
    else:
        build_cmd = (
            f"bash build.sh --pkg --vendor_name=custom"
            f" --soc={soc} --ops={op_name} -j{nproc}"
        )
    logging.info(f"执行构建: {build_cmd}")

    # omni-ops 保护: 备份 repo_root/output 目录，防止 build.sh 误删
    repo_output = os.path.join(repo_root, "output")
    repo_output_bak = repo_output + f"_bak_{os.getpid()}"
    output_backed_up = False
    if is_omni and os.path.exists(repo_output):
        logging.info(f"omni-ops 保护: 备份 {repo_output} -> {repo_output_bak}")
        shutil.move(repo_output, repo_output_bak)
        output_backed_up = True

    try:
        result = subprocess.run(
            build_cmd,
            shell=True,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes (large ops repos like ops-transformer need longer)
        )
    finally:
        # 恢复备份的 output 目录
        if output_backed_up and os.path.exists(repo_output_bak):
            if os.path.exists(repo_output):
                # 构建过程可能生成了新的 output，保留原备份
                logging.info(f"omni-ops 保护: 构建产生了新的 output 目录，保留原备份")
                shutil.rmtree(repo_output)
            logging.info(f"omni-ops 保护: 恢复 {repo_output_bak} -> {repo_output}")
            shutil.move(repo_output_bak, repo_output)

    if result.returncode != 0:
        error_msg = (
            f"构建失败!\n"
            f"Exit Code: {result.returncode}\n"
            f"Stdout (last 2000 chars):\n{result.stdout[-2000:]}\n"
            f"Stderr (last 2000 chars):\n{result.stderr[-2000:]}"
        )
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    logging.info("构建成功")

    # Step 3: 查找 .run 文件
    run_file = find_run_file(repo_root, is_omni)
    logging.info(f"找到 .run 文件: {run_file}")

    # Step 4: 创建安装目录
    os.makedirs(install_path, exist_ok=True)

    # Step 5: 执行安装
    install_cmd = f"bash {run_file} --install-path={install_path}"
    logging.info(f"执行安装: {install_cmd}")

    result = subprocess.run(
        install_cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        error_msg = (
            f"安装失败!\n"
            f"Exit Code: {result.returncode}\n"
            f"Stdout:\n{result.stdout}\n"
            f"Stderr:\n{result.stderr}"
        )
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    logging.info("安装成功")

    # Step 6: 检测 vendor 子目录
    vendor_subdir = detect_vendor_subdir(install_path)
    vendor_path = os.path.join(install_path, "vendors", vendor_subdir)
    if not os.path.isdir(vendor_path):
        logging.warning(
            f"vendors子目录未找到: {vendor_path}, "
            f"请手动检查 {os.path.join(install_path, 'vendors')}/"
        )

    return {
        "install_path": install_path,
        "vendor_subdir": vendor_subdir,
        "repo_type": repo_type,
        "run_file": run_file,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build and install ops repository operator"
    )
    parser.add_argument(
        "--repo-root", required=True,
        help="ops仓根目录路径 (如 /home/user/ops-nn)"
    )
    parser.add_argument(
        "--op-name", required=True,
        help="算子名 (如 ada_layer_norm_custom)"
    )
    parser.add_argument(
        "--soc", required=True,
        help="目标芯片 (如 ascend910b, ascend950)"
    )
    parser.add_argument(
        "--install-path", required=True,
        help="安装目标路径 (必须是绝对路径)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        result = build_and_install(
            repo_root=args.repo_root,
            op_name=args.op_name,
            soc=args.soc,
            install_path=args.install_path,
        )
        print(f"\n构建安装完成:")
        print(f"  安装路径: {result['install_path']}")
        print(f"  仓类型: {result['repo_type']}")
        print(f"  vendor子目录: {result['vendor_subdir']}")
        print(f"  .run文件: {result['run_file']}")

        # 输出JSON供脚本解析
        import json
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        print(f"\n--- BUILD_RESULT_JSON ---")
        print(result_json)
        print(f"--- END_BUILD_RESULT_JSON ---")

    except Exception as e:
        logging.error(f"构建安装失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
