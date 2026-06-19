#!/usr/bin/env python3
"""
NPU operator test task objects for LINGXI.

简化版本，移除了agenticKernelGenerator特定的目录结构和依赖。
LINGXI 使用 output/{op_name}/ 目录结构。
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestResult:
    """Test result base class."""
    success: bool
    log_file: str
    n_success: int = 0
    n_failed: int = 0
    results_dict: list = None

    def __post_init__(self):
        if self.results_dict is None:
            self.results_dict = []


class LINGXITaskObject:
    """
    LINGXI 项目的简化任务对象。

    LINGXI 目录结构:
        output/<op_name>/
            <op_name>_custom.py
            <op_name>_reference.py
            vendors/customize/op_api/lib/libcust_opapi.so
            profiling/
    """

    def __init__(self, op_name: str, output_base_path: str = "output"):
        self.op_name = op_name
        self.output_base_path = output_base_path

        # 设置路径
        self.base_path = Path(output_base_path) / op_name
        self.custom_lib_path = self.base_path / "vendors" / "customize" / "op_api" / "lib"
        self.profiling_path = self.base_path / "profiling"
        self.custom_code_path = self.base_path / f"{op_name}_custom.py"
        self.reference_code_path = self.base_path / f"{op_name}_reference.py"
        self.test_cases_path = self.base_path / "test_cases.csv"

    def check_custom_lib_exists(self) -> bool:
        """检查自定义算子库是否存在"""
        lib_path = self.custom_lib_path / "libcust_opapi.so"
        return lib_path.exists()

    def set_custom_opp_path(self):
        """设置环境变量 ASCEND_CUSTOM_OPP_PATH"""
        custom_opp_path = self.base_path / "vendors" / "customize"
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp_path)

        # 设置 LD_LIBRARY_PATH
        lib_path = str(self.custom_lib_path.resolve())
        existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_path not in existing_ld_path:
            new_ld_path = f"{lib_path}:{existing_ld_path}".rstrip(":")
            os.environ["LD_LIBRARY_PATH"] = new_ld_path


def find_lingxi2_task_path(op_name: str, base_path: str = "output") -> str:
    """
    查找 LINGXI 算子任务路径。

    Args:
        op_name: 算子名称
        base_path: 基础路径，默认为 "output"

    Returns:
        算子任务路径，如果不存在则返回 None
    """
    task_path = Path(base_path) / op_name
    if task_path.exists() and task_path.is_dir():
        return str(task_path)
    return None
