#!/usr/bin/env python3
"""
生成 AscendC 算子的 PyBind 绑定并编译。

用法:
    python generate_pybind.py <op_name> [--output_dir <目录>]

示例:
    # 为 Add 算子生成 PyBind 绑定（使用默认 output 目录）
    python generate_pybind.py Add

    # 指定输出目录
    python generate_pybind.py Add --output_dir /path/to/output/Add

    # 为自定义算子生成绑定
    python generate_pybind.py MyCustomOperator --output_dir /custom/path/MyCustomOperator

说明:
    该脚本会从指定目录下读取 <op_name>.cpp 文件，
    生成 PyBind11 绑定代码并编译成 Python 可导入的 wheel 包。
    编译成功后会自动安装生成的 wheel 包。

参数:
    op_name: 算子名称
    --output_dir: 输出目录路径（默认为 output/<op_name>）

依赖:
    - <output_dir>/<op_name>.cpp 必须存在
    - scripts/template/ 目录必须存在
"""

import sys
import shutil
from pathlib import Path
import subprocess
import logging
import argparse


def generate_pybind_bindings(work_dir: Path, op_cpp: Path) -> None:
    """
    生成 PyBind 绑定并编译 ascendc 算子。

    Args:
        work_dir (Path): 工作目录路径
        op_cpp (Path): 算子 op.cpp 文件路径

    Raises:
        FileNotFoundError: 模板目录或必要的文件不存在
        subprocess.CalledProcessError: 编译失败时抛出异常，包含错误信息
        Exception: 其他异常情况
    """
    work_dir = Path(work_dir).resolve()
    template_dir = Path(__file__).parent.joinpath("template")
    target_dir = work_dir.joinpath("ascend_op_pybind")

    # 检查模板目录是否存在
    if not template_dir.exists():
        raise FileNotFoundError(f"模板目录不存在: {template_dir}")

    # 如果目标目录不存在，则拷贝模板目录
    if not target_dir.exists():
        logging.info(f"拷贝模板目录到: {target_dir}")
        shutil.copytree(template_dir, target_dir)
    else:
        logging.info(f"目标目录已存在: {target_dir}")

    # 处理 op.cpp 文件
    op_cpp_path = Path(op_cpp).resolve()
    cpp_path = target_dir.joinpath("CppExtension/csrc/op.cpp")
    
    if not op_cpp_path.exists():
        raise FileNotFoundError(f"op.cpp 源文件不存在: {op_cpp_path}")
    
    # 如果目标文件存在，先删除
    if cpp_path.exists():
        logging.info(f"删除已存在的 op.cpp: {cpp_path}")
        cpp_path.unlink()
    
    # 拷贝源文件到目标位置
    logging.info(f"拷贝 op.cpp 文件: {op_cpp_path} -> {cpp_path}")
    cpp_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(op_cpp, cpp_path)

    # 执行编译
    try:
        logging.info(f"开始编译 PyBind 绑定")
        extension_dir = target_dir.joinpath("CppExtension")

        result = subprocess.run(
            [sys.executable, 'setup.py', 'build', 'bdist_wheel'],
            cwd=str(extension_dir),
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            logging.info("编译 wheel 包成功")

            # 安装 wheel 包
            dist_dir = extension_dir.joinpath("dist")
            if dist_dir.exists():
                for wheel_file in dist_dir.glob("*.whl"):
                    result_install = subprocess.run(
                        [sys.executable, '-m', 'pip', 'install', str(wheel_file), '--force-reinstall'],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if result_install.returncode == 0:
                        logging.info(f"安装 {wheel_file.name} 成功")
                    else:
                        logging.error(f"安装 {wheel_file.name} 失败: {result_install.stderr}")

            logging.info(f"PyBind 绑定编译成功")
        else:
            error_msg = (
                f"编译失败！\n"
                f"Exit Code: {result.returncode}\n"
                f"Stdout:\n{result.stdout}\n"
                f"Stderr:\n{result.stderr}"
            )
            logging.error(error_msg)
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)

    except subprocess.TimeoutExpired:
        raise Exception(f"编译超时（超过 300 秒）")
    except Exception as e:
        raise Exception(f"编译过程中发生错误: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Generate PyBind bindings for AscendC operators")
    parser.add_argument("op_name", type=str, help="Operator name")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory path (default: output/<op_name>)")
    
    args = parser.parse_args()
    
    try:
        if args.output_dir:
            work_dir = Path(args.output_dir).resolve()
        else:
            work_dir = Path("output").joinpath(args.op_name).resolve()
        op_cpp = work_dir.joinpath(f"{args.op_name}.cpp")
        generate_pybind_bindings(work_dir, op_cpp)
        logging.info("PyBind bindings generated successfully")
    except Exception as e:
        logging.error(e)
