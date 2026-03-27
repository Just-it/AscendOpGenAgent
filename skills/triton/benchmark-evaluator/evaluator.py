#!/usr/bin/env python3
"""
Benchmark Evaluator 工具库 — 供 LLM 通过 bash 子命令调用。

提供三个子命令：
  scan         扫描 KernelBench 任务列表
  save-result  保存单个任务的结构化结果
  summary      生成执行摘要

用法:
    python evaluator.py scan --benchmark_path /path --level_problems '{"1": null}'
    python evaluator.py save-result --output_path /path --level 1 --problem_id 3 --op_name softmax --summary_json /path/summary.json
    python evaluator.py summary --output_path /path --agent_name triton-ascend
"""

import os
import sys
import json
import re
import glob
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# TaskScanner — 任务扫描
# ============================================================

class TaskScanner:
    """任务扫描器"""

    @staticmethod
    def parse_problem_ids(problem_ids: Optional[Any]) -> Optional[List[int]]:
        """解析 problem_ids 参数

        支持格式:
        - None: 表示全选
        - List[int]: [1, 2, 3] 或 [20, 30]
        - str:
            - JSON 列表: "[1, 2, 3]" 或 "[20, 30]"
            - 范围格式: "1-10"
            - 逗号分隔: "1,2,3"
        """
        if problem_ids is None:
            return None

        if isinstance(problem_ids, list):
            return [int(x) for x in problem_ids]

        if isinstance(problem_ids, str):
            # 尝试解析为 JSON 列表
            try:
                parsed = json.loads(problem_ids)
                if isinstance(parsed, list):
                    return [int(x) for x in parsed]
            except json.JSONDecodeError:
                pass

            # 支持 "1-10" 或 "1,2,3" 格式
            result = []
            parts = problem_ids.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    start, end = part.split('-', 1)
                    result.extend(range(int(start), int(end) + 1))
                else:
                    result.append(int(part))
            return result

        return None

    @staticmethod
    def scan_tasks(
        benchmark_path: str,
        level_problems: Dict[int, Optional[Any]],
        completed_tasks: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """扫描任务并过滤已完成的任务

        Args:
            benchmark_path: Benchmark 根目录
            level_problems: {level: [problem_ids] or None}
            completed_tasks: 已完成任务列表 [{"level": 1, "problem_id": 1}, ...]

        Returns:
            {"total_scanned": N, "skipped": M, "pending": [...]}
        """
        # 构建已完成任务集合
        completed_set = set()
        if completed_tasks:
            for t in completed_tasks:
                completed_set.add((int(t["level"]), int(t["problem_id"])))

        all_tasks = []

        for level, problem_ids in level_problems.items():
            level = int(level)
            level_dir = os.path.join(benchmark_path, f"level{level}")

            if not os.path.exists(level_dir):
                logger.warning(f"Level 目录不存在: {level_dir}")
                continue

            parsed_ids = TaskScanner.parse_problem_ids(problem_ids)

            task_files = glob.glob(os.path.join(level_dir, "*.py"))

            for task_file in task_files:
                match = re.match(r'(\d+)_(.+)\.py', os.path.basename(task_file))
                if not match:
                    continue

                pid = int(match.group(1))
                op_name = match.group(2)

                if parsed_ids is not None and pid not in parsed_ids:
                    continue

                all_tasks.append({
                    'level': level,
                    'problem_id': pid,
                    'task_file': os.path.abspath(task_file),
                    'op_name': op_name
                })

        all_tasks.sort(key=lambda x: (x['level'], x['problem_id']))

        # 过滤已完成任务
        pending = []
        skipped = 0
        for task in all_tasks:
            if (task['level'], task['problem_id']) in completed_set:
                skipped += 1
            else:
                pending.append(task)

        return {
            "total_scanned": len(all_tasks),
            "skipped": skipped,
            "pending": pending
        }

    @staticmethod
    def classify_op_type(op_name: str) -> str:
        """分类算子类型"""
        op_name_lower = op_name.lower()

        if any(kw in op_name_lower for kw in ['matmul', 'bmm', 'linear', 'gemm']):
            return 'matmul'
        elif any(kw in op_name_lower for kw in ['conv']):
            return 'conv'
        elif any(kw in op_name_lower for kw in ['softmax', 'layernorm', 'batchnorm', 'sum', 'mean', 'max', 'min']):
            return 'reduce'
        elif any(kw in op_name_lower for kw in ['attention', 'mha', 'sdpa']):
            return 'attention'
        elif any(kw in op_name_lower for kw in ['add', 'mul', 'sub', 'div', 'relu', 'sigmoid', 'tanh', 'gelu', 'silu']):
            return 'elementwise'
        else:
            return 'other'


# ============================================================
# StateManager — 断点续跑状态管理
# ============================================================

class StateManager:
    """状态管理器（与 benchmark-scheduler Agent 的 .benchmark_state.json 格式对齐）"""

    def __init__(self, output_dir: str):
        self.state_file = os.path.join(output_dir, ".benchmark_state.json")
        self.state = {
            "completed_tasks": [],
            "failed_tasks": [],
            "arch": "",
            "npu_id": 0,
            "last_update": ""
        }
        self._load()

    def _load(self):
        """加载状态文件"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.state.update(data)
                logger.info(f"已加载状态: {len(self.state['completed_tasks'])} 个已完成, "
                            f"{len(self.state['failed_tasks'])} 个失败")
            except Exception as e:
                logger.warning(f"加载状态文件失败: {e}")

    def _save(self):
        """保存状态文件"""
        self.state["last_update"] = datetime.now().isoformat()
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存状态文件失败: {e}")

    def get_completed_tasks(self) -> List[Dict]:
        """获取已完成任务列表"""
        return self.state["completed_tasks"]

    def mark_completed(self, level: int, problem_id: int, retry_count: int = 0):
        """标记任务完成"""
        # 从 failed_tasks 中移除（如果存在）
        self.state["failed_tasks"] = [
            t for t in self.state["failed_tasks"]
            if not (t["level"] == level and t["problem_id"] == problem_id)
        ]
        # 添加到 completed_tasks（去重）
        if not any(t["level"] == level and t["problem_id"] == problem_id
                   for t in self.state["completed_tasks"]):
            self.state["completed_tasks"].append({
                "level": level,
                "problem_id": problem_id,
                "retry_count": retry_count
            })
        self._save()

    def mark_failed(self, level: int, problem_id: int, error_type: str, retry_count: int = 0):
        """标记任务失败"""
        # 更新或添加到 failed_tasks
        for t in self.state["failed_tasks"]:
            if t["level"] == level and t["problem_id"] == problem_id:
                t["error_type"] = error_type
                t["retry_count"] = retry_count
                self._save()
                return
        self.state["failed_tasks"].append({
            "level": level,
            "problem_id": problem_id,
            "error_type": error_type,
            "retry_count": retry_count
        })
        self._save()

    def set_metadata(self, arch: str, npu_id: int):
        """设置硬件元数据"""
        self.state["arch"] = arch
        self.state["npu_id"] = npu_id
        self._save()


# ============================================================
# ResultSaver — 保存单个任务结果
# ============================================================

def save_task_result(
    output_path: str,
    level: int,
    problem_id: int,
    op_name: str,
    summary_json_path: str
) -> Dict[str, Any]:
    """从 kernelgen-workflow 的 summary.json 提取结果并保存结构化数据

    Args:
        output_path: 根输出目录
        level: Level 编号
        problem_id: Problem ID
        op_name: 算子名称
        summary_json_path: kernelgen-workflow 输出的 summary.json 路径

    Returns:
        结构化的任务结果
    """
    task_dir = os.path.join(output_path, f"level_{level}", f"{problem_id}_{op_name}")
    op_type = TaskScanner.classify_op_type(op_name)

    result = {
        "level": level,
        "problem_id": problem_id,
        "op_name": op_name,
        "op_type": op_type,
        "status": "failed",
        "iterations": 0,
        "verify_passed": False,
        "perf_data": None,
        "failure_reason": None,
        "output_path": task_dir,
        "timestamp": datetime.now().isoformat()
    }

    # 读取 summary.json
    if os.path.exists(summary_json_path):
        try:
            with open(summary_json_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)

            result["iterations"] = summary.get("iterations", 0)

            if summary.get("success", False):
                result["status"] = "success"
                result["verify_passed"] = True
                result["perf_data"] = summary.get("perf_data")
            else:
                result["status"] = "failed"
                result["failure_reason"] = summary.get("failure_reason", summary.get("last_error", "未知错误"))

        except Exception as e:
            result["failure_reason"] = f"读取 summary.json 失败: {e}"
    else:
        result["failure_reason"] = f"summary.json 不存在: {summary_json_path}"

    # 保存结构化结果到任务目录
    result_file = os.path.join(task_dir, "eval_result.json")
    os.makedirs(task_dir, exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 更新状态文件
    state = StateManager(output_path)
    if result["status"] == "success":
        state.mark_completed(level, problem_id)
    else:
        error_type = "generation" if result["failure_reason"] else "unknown"
        state.mark_failed(level, problem_id, error_type)

    return result


# ============================================================
# SummaryGenerator — 生成执行摘要
# ============================================================

def generate_summary(output_path: str, agent_name: str) -> Dict[str, Any]:
    """扫描所有 eval_result.json，生成执行摘要

    Args:
        output_path: 根输出目录
        agent_name: Agent 名称

    Returns:
        执行摘要字典
    """
    results = []

    # 扫描所有 eval_result.json
    for root, dirs, files in os.walk(output_path):
        if "eval_result.json" in files:
            result_file = os.path.join(root, "eval_result.json")
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    results.append(json.load(f))
            except Exception as e:
                logger.warning(f"读取结果文件失败: {result_file}: {e}")

    results.sort(key=lambda x: (x.get("level", 0), x.get("problem_id", 0)))

    total = len(results)
    success = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")
    timeout = sum(1 for r in results if r.get("status") == "timeout")

    speedups = []
    for r in results:
        if r.get("perf_data") and r["perf_data"].get("speedup_vs_torch"):
            speedups.append(r["perf_data"]["speedup_vs_torch"])

    summary = {
        "agent_name": agent_name,
        "timestamp": datetime.now().isoformat(),
        "total_tasks": total,
        "completed_tasks": success,
        "failed_tasks": failed,
        "timeout_tasks": timeout,
        "avg_speedup": round(sum(speedups) / len(speedups), 2) if speedups else 0,
        "results": results
    }

    # 保存摘要
    summary_file = os.path.join(output_path, "eval_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"执行摘要已保存: {summary_file}")

    return summary


# ============================================================
# CLI 入口
# ============================================================

def cmd_scan(args):
    """scan 子命令"""
    level_problems = json.loads(args.level_problems)
    # 将字符串 key 转为 int
    level_problems = {int(k): v for k, v in level_problems.items()}

    completed_tasks = None
    if args.completed_tasks:
        completed_tasks = json.loads(args.completed_tasks)

    result = TaskScanner.scan_tasks(
        benchmark_path=args.benchmark_path,
        level_problems=level_problems,
        completed_tasks=completed_tasks
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_save_result(args):
    """save-result 子命令"""
    result = save_task_result(
        output_path=args.output_path,
        level=args.level,
        problem_id=args.problem_id,
        op_name=args.op_name,
        summary_json_path=args.summary_json
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_summary(args):
    """summary 子命令"""
    summary = generate_summary(
        output_path=args.output_path,
        agent_name=args.agent_name
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Benchmark Evaluator 工具库',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子命令:
  scan          扫描 KernelBench 任务列表
  save-result   保存单个任务的结构化结果
  summary       生成执行摘要
"""
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- scan ---
    p_scan = subparsers.add_parser('scan', help='扫描任务列表')
    p_scan.add_argument('--benchmark_path', required=True, help='Benchmark 根目录绝对路径')
    p_scan.add_argument('--level_problems', required=True, help='评测范围 JSON 字符串，如 \'{"1": null, "2": [1,5]}\'')
    p_scan.add_argument('--completed_tasks', default=None, help='已完成任务 JSON 字符串（可选，用于断点续跑）')
    p_scan.set_defaults(func=cmd_scan)

    # --- save-result ---
    p_save = subparsers.add_parser('save-result', help='保存单个任务结果')
    p_save.add_argument('--output_path', required=True, help='根输出目录绝对路径')
    p_save.add_argument('--level', type=int, required=True, help='Level 编号')
    p_save.add_argument('--problem_id', type=int, required=True, help='Problem ID')
    p_save.add_argument('--op_name', required=True, help='算子名称')
    p_save.add_argument('--summary_json', required=True, help='kernelgen-workflow 输出的 summary.json 路径')
    p_save.set_defaults(func=cmd_save_result)

    # --- summary ---
    p_summary = subparsers.add_parser('summary', help='生成执行摘要')
    p_summary.add_argument('--output_path', required=True, help='根输出目录绝对路径')
    p_summary.add_argument('--agent_name', required=True, help='Agent 名称')
    p_summary.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
