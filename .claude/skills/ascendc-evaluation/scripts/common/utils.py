#!/usr/bin/env python3
"""
Common utilities for AscendC evaluation scripts.

保留的通用工具函数，移除了agenticKernelGenerator特定的功能。
"""

import os
import subprocess
import logging


def validate_environment() -> bool:
    """Validate required environment variables."""
    required_vars = ["ASCEND_HOME_PATH"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    if missing_vars:
        print(f"[Error] Missing environment variable: {', '.join(missing_vars)}")
        print("Please run: source set_ascend.sh")
        return False

    return True


def check_torch_npu() -> bool:
    """Check torch_npu availability."""
    try:
        import torch_npu  # noqa: F401
        return True
    except ImportError:
        print("[Error] torch_npu not installed")
        print("Please install the torch_npu package to support NPU")
        return False


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Setup a logger with optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger


def run_subprocess_command(cmd, logger=None, cwd=None):
    """Run a subprocess command and capture output."""
    if logger:
        logger.info(f"[CMD]: {cmd}")
        if cwd:
            logger.info(f"[CWD]: {cwd}")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
        executable="/bin/bash",
        cwd=cwd,
    )

    if proc.stdout:
        if proc.returncode != 0:
            if logger:
                logger.error(proc.stdout)
        else:
            if logger:
                logger.info(proc.stdout)
    if proc.stderr:
        if proc.returncode != 0:
            if logger:
                logger.error(proc.stderr)
        else:
            if logger:
                logger.warning(proc.stderr)

    if proc.returncode != 0:
        error_prefix = "Command failed"
        raise RuntimeError(f"{error_prefix} with exit code {proc.returncode}")

    return proc
