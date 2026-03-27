#!/usr/bin/env python3
"""性能测试脚本 — 使用 torch_npu.profiler 进行详细性能分析。

用法:
    python performance-test.py --op_name <算子名> [--verify_dir <目录>]
                              [--num_iterations <次数>] [--output <路径>]

指标:
    - device_self_duration_us: 算子在 Device 侧的耗时（除去内部调用），单位 us
    - device_total_duration_us: 算子在 Device 侧的耗时，单位 us
    - operator_details: 各算子的平均耗时统计
    - speedup: 生成实现相对于框架实现的加速比
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import tempfile


@dataclass(frozen=True)
class OperatorDuration:
    """单个算子的耗时数据"""
    name: str
    avg_device_self_duration_us: float
    avg_device_total_duration_us: float


@dataclass
class ProfilingResult:
    """单个实现的 profiling 结果"""
    device_self_duration_us: list[float] = field(default_factory=list)
    device_total_duration_us: list[float] = field(default_factory=list)
    operator_details: list[OperatorDuration] = field(default_factory=list)
    
    # 采集时间元数据
    collection_time_seconds: float = 0.0
    profiling_overhead_seconds: float = 0.0
    pure_execution_time_seconds: float = 0.0
    raw_iterations: int = 0
    actual_profiling_iterations: int = 0

    @property
    def avg_device_self_duration_us(self) -> float:
        return self._safe_average(self.device_self_duration_us)

    @property
    def avg_device_total_duration_us(self) -> float:
        return self._safe_average(self.device_total_duration_us)

    @staticmethod
    def _safe_average(durations: Sequence[float]) -> float:
        return sum(durations) / len(durations) if durations else 0.0

    @property
    def overhead_percentage(self) -> float:
        """计算采集开销百分比"""
        if self.collection_time_seconds > 0:
            return self.profiling_overhead_seconds / self.collection_time_seconds * 100
        return 0.0


@dataclass
class ModelContext:
    """模型运行上下文"""
    model: Any
    inputs: Any
    name: str


@dataclass(frozen=True)
class NPUProfilerConfig:
    """NPU Profiler 配置"""
    num_iterations: int = 50
    warm_up: int = 5
    initialization_warmup: int = 5
    aic_metrics: Any = None
    
    @property
    def skip_first(self) -> int:
        """第一跳过的迭代数（预热+1）"""
        return 1 + self.warm_up


class ProfilingAnalyzer:
    """基于 operator_details.csv 的性能分析器"""
    
    REQUIRED_COLUMNS = ('Name', 'Device Self Duration(us)', 'Device Total Duration(us)')
    
    def __init__(self, operator_details_dict: dict | None = None, 
                 device_self_durations: list | None = None,
                 device_total_durations: list | None = None) -> None:
        self.operator_details_dict = operator_details_dict or defaultdict(
            lambda: {'self_durations': [], 'total_durations': []}
        )
        self.device_self_durations = device_self_durations or []
        self.device_total_durations = device_total_durations or []

    def ingest_dataframe(self, df: Any) -> 'ProfilingAnalyzer':
        """从 DataFrame 摄取分析数据"""
        if not all(col in df.columns for col in self.REQUIRED_COLUMNS):
            return self

        for _, row in df.iterrows():
            self._process_row(row)
            
        return self

    def _process_row(self, row: Any) -> None:
        """处理单行数据"""
        name = str(row['Name']).strip()
        self_duration = row['Device Self Duration(us)']
        total_duration = row['Device Total Duration(us)']

        self_duration, total_duration = self._parse_durations(self_duration, total_duration)
        if self_duration is None:
            return

        self._add_durations(self_duration, total_duration, name)

    def _add_durations(self, self_duration: float, total_duration: float, name: str) -> None:
        """添加有效持续时间"""
        from pandas import isna as pd_isna
        
        if self_duration > 0 and not pd_isna(self_duration):
            self.device_self_durations.append(self_duration)
        
        if total_duration > 0 and not pd_isna(total_duration):
            self.device_total_durations.append(total_duration)
        
        if name and name != 'nan':
            self.operator_details_dict[name]['self_durations'].append(self_duration)
            self.operator_details_dict[name]['total_durations'].append(total_duration)

    @staticmethod
    def _parse_durations(self_duration: Any, total_duration: Any) -> tuple[float, float] | tuple[None, None]:
        """解析持续时间值"""
        try:
            return float(self_duration), float(total_duration)
        except (ValueError, TypeError):
            return None, None

    def build_result(self) -> ProfilingResult:
        """构建最终的分析结果"""
        operator_details = self._build_operator_list()
        
        # 按 avg_device_self_duration_us 降序排序
        operator_details.sort(key=lambda x: x.avg_device_self_duration_us, reverse=True)
        
        return ProfilingResult(
            device_self_duration_us=self.device_self_durations,
            device_total_duration_us=self.device_total_durations,
            operator_details=operator_details
        )

    def _build_operator_list(self) -> list[OperatorDuration]:
        """构建算子列表"""
        operator_details: list[OperatorDuration] = []
        
        for name, data in self.operator_details_dict.items():
            self_durations = data['self_durations']
            total_durations = data['total_durations']
            
            avg_self = sum(self_durations) / len(self_durations) if self_durations else 0.0
            avg_total = sum(total_durations) / len(total_durations) if total_durations else 0.0
            
            operator_details.append(OperatorDuration(
                name=name,
                avg_device_self_duration_us=round(avg_self, 4),
                avg_device_total_duration_us=round(avg_total, 4)
            ))
            
        return operator_details


@contextmanager
def profiler_session(op_name: str, config: NPUProfilerConfig) -> Path:
    """NPU Profiler 会话上下文管理器"""
    import pandas as pd
    import torch
    import torch_npu
    
    # 创建临时目录
    timestamp = int(time.time() * 1000)
    prof_dir = Path(tempfile.gettempdir()) / f"profile_{op_name}_{timestamp}"
    prof_dir.mkdir(parents=True, exist_ok=True)

    try:
        yield prof_dir
    finally:
        # 清理临时目录
        _cleanup_directory(prof_dir)


def _cleanup_directory(prof_dir: Path) -> None:
    """安全清理临时目录"""
    if prof_dir.exists():
        with suppress(Exception):
            shutil.rmtree(prof_dir)


class NPUProfiler:
    """NPU 性能分析器"""
    
    OPERATOR_DETAILS_FILE = 'operator_details.csv'
    TOP_OPERATOR_COUNT = 10
    
    def __init__(self, config: NPUProfilerConfig | None = None) -> None:
        self.config = config or NPUProfilerConfig()
        self._init_torch_npu()

    def _init_torch_npu(self) -> None:
        """初始化 torch_npu 导入（延迟导入避免全局依赖）"""
        from torch_npu.profiler import (
            profile as npu_profile,
            ProfilerActivity,
            schedule,
            tensorboard_trace_handler,
            _ExperimentalConfig,
            AiCMetrics,
            ProfilerLevel,
        )
        from pandas import isna as pd_isna
        
        self.npu_profile = npu_profile
        self.ProfilerActivity = ProfilerActivity
        self.schedule = schedule
        self.tensorboard_trace_handler = tensorboard_trace_handler
        self._ExperimentalConfig = _ExperimentalConfig
        self.AiCMetrics = AiCMetrics
        self.ProfilerLevel = ProfilerLevel
        self.pd_isna = pd_isna

    def profile(self, op_name: str, model: Any, inputs: Any) -> ProfilingResult:
        """对模型进行性能分析"""
        import torch
        import pandas as pd

        self._initialize_model(model, inputs)
        
        with profiler_session(op_name, self.config) as prof_dir:
            self._run_profiling(model, inputs, prof_dir)
            return self._analyze_results(prof_dir)

    def _initialize_model(self, model: Any, inputs: Any) -> None:
        """初始化模型（预热）"""
        import torch
        
        model.eval()
        with torch.no_grad():
            for _ in range(self.config.initialization_warmup):
                _ = model(*inputs)
        torch.npu.synchronize()

    def _run_profiling(self, model: Any, inputs: Any, prof_dir: Path) -> None:
        """执行 profiling 运行（带时间统计）"""
        import torch
        import torch_npu

        total = self._calculate_total_iterations()
        experimental_config = self._create_experimental_config()
        
        # 记录总采集开始时间
        collection_start = time.perf_counter()
        
        with self.npu_profile(
            activities=[self.ProfilerActivity.NPU, self.ProfilerActivity.CPU],
            schedule=self.schedule(
                wait=0,
                warmup=0,
                active=self.config.num_iterations,
                repeat=1,
                skip_first=self.config.skip_first
            ),
            on_trace_ready=self.tensorboard_trace_handler(str(prof_dir)),
            record_shapes=False,
            with_flops=False,
            with_modules=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=experimental_config
        ) as prof:
            # 记录纯执行开始时间
            execution_start = time.perf_counter()
            
            for i in range(total):
                with torch.no_grad():
                    _ = model(*inputs)
                prof.step()
                # 最后一个迭代后不需要手动同步，profiler 会处理
                if i < total - 1:
                    torch.npu.synchronize()
            
            # 确保所有 NPU 操作完成
            torch.npu.synchronize()
            
            # 记录纯执行结束时间
            execution_end = time.perf_counter()
        
        # 记录总采集结束时间
        collection_end = time.perf_counter()
        
        # 计算时间
        pure_execution_time = execution_end - execution_start
        total_collection_time = collection_end - collection_start
        profiling_overhead = total_collection_time - pure_execution_time
        
        # 保存到实例变量，供后续使用
        self._timing_info = {
            'collection_time': total_collection_time,
            'execution_time': pure_execution_time,
            'overhead_time': profiling_overhead,
            'raw_iterations': total,
            'actual_profiling_iterations': self.config.num_iterations
        }

    def _calculate_total_iterations(self) -> int:
        """计算总迭代数"""
        return self.config.skip_first + self.config.num_iterations

    def _create_experimental_config(self) -> Any:
        """创建实验配置"""
        aic_metrics = self.config.aic_metrics or self.AiCMetrics.PipeUtilization
        return self._ExperimentalConfig(
            aic_metrics=aic_metrics,
            profiler_level=self.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=False
        )

    def _analyze_results(self, prof_dir: Path) -> ProfilingResult:
        """分析 profiling 结果（包含计时元数据）"""
        import pandas as pd
        
        analyzer = ProfilingAnalyzer()
        
        for csv_path in self._find_csv_files(prof_dir):
            try:
                df = pd.read_csv(csv_path)
                analyzer.ingest_dataframe(df)
            except Exception as e:
                print(f"Warning: Failed to read {csv_path}: {e}")
        
        result = analyzer.build_result()
        
        # 合并计时信息到结果
        if hasattr(self, '_timing_info'):
            result.collection_time_seconds = self._timing_info['collection_time']
            result.profiling_overhead_seconds = self._timing_info['overhead_time']
            result.pure_execution_time_seconds = self._timing_info['execution_time']
            result.raw_iterations = self._timing_info['raw_iterations']
            result.actual_profiling_iterations = self._timing_info['actual_profiling_iterations']
        
        return result

    def _find_csv_files(self, prof_dir: Path) -> list[Path]:
        """查找所有 operator_details.csv 文件"""
        return [
            csv_path 
            for csv_path in prof_dir.rglob(self.OPERATOR_DETAILS_FILE)
        ]


class SpeedupCalculator:
    """加速比计算器"""
    
    @staticmethod
    def calculate(framework_result: ProfilingResult, 
                  impl_result: ProfilingResult) -> tuple[float, float]:
        """计算加速比"""
        speedup_self = SpeedupCalculator._calc_ratio(
            framework_result.avg_device_self_duration_us,
            impl_result.avg_device_self_duration_us
        )
        speedup_total = SpeedupCalculator._calc_ratio(
            framework_result.avg_device_total_duration_us,
            impl_result.avg_device_total_duration_us
        )
        return speedup_self, speedup_total

    @staticmethod
    def _calc_ratio(framework_avg: float, impl_avg: float) -> float:
        """计算单向加速比"""
        return framework_avg / impl_avg if impl_avg > 0 else 1.0


class ProfilingRunner:
    """Profiling 运行器"""
    
    SEED = 0
    DEVICE = 'npu'
    MAX_OPERATORS_DISPLAY = 10
    
    def __init__(self, profiler: NPUProfiler | None = None) -> None:
        self.profiler = profiler or NPUProfiler()

    def run(self, op_name: str, verify_dir: str, config: NPUProfilerConfig | None = None) -> dict:
        """运行完整的 profiling 流程"""
        if config:
            self.profiler.config = config
            
        sys.path.insert(0, verify_dir)
        
        try:
            framework, implementation = self._load_and_prepare_models(op_name, verify_dir)
            
            print(f"Profiling 框架实现 ({self.profiler.config.num_iterations} iterations)...")
            framework_result = self.profiler.profile(op_name, framework.model, framework.inputs)
            
            print(f"Profiling 生成实现 ({self.profiler.config.num_iterations} iterations)...")
            impl_result = self.profiler.profile(op_name, implementation.model, implementation.inputs)
            
            return self._build_final_result(op_name, config, framework_result, impl_result)
            
        finally:
            with suppress(ValueError):
                sys.path.remove(verify_dir)

    def _load_and_prepare_models(self, op_name: str, verify_dir: str) -> tuple[ModelContext, ModelContext]:
        """加载并准备模型"""
        import torch
        
        device = torch.device(self.DEVICE)
        
        # 动态导入模块
        torch_module = __import__(f"{op_name}_torch")
        impl_module = __import__(f"{op_name}_triton_ascend_impl")
        
        # 获取初始化参数
        init_params = torch_module.get_init_inputs()
        
        # 创建框架模型
        torch.manual_seed(self.SEED)
        torch.npu.manual_seed(self.SEED)
        framework_model = self._create_model(torch_module.Model, init_params, device)
        
        # 创建实现模型
        torch.manual_seed(self.SEED)
        torch.npu.manual_seed(self.SEED)
        impl_model = self._create_model(impl_module.ModelNew, init_params, device)
        
        # 准备输入
        framework_inputs = self._prepare_inputs(torch_module.get_inputs, device)
        impl_inputs = self._prepare_inputs(torch_module.get_inputs, device)
        
        return (
            ModelContext(framework_model, framework_inputs, f"{op_name}_torch"),
            ModelContext(impl_model, impl_inputs, f"{op_name}_triton_ascend_impl")
        )

    @staticmethod
    def _create_model(model_class: type, init_params: Any, device: Any) -> Any:
        """创建模型实例"""
        return model_class(*init_params).to(device)

    @staticmethod
    def _prepare_inputs(get_inputs_func: type, device: Any) -> list:
        """准备模型输入"""
        import torch
        
        torch.manual_seed(ProfilingRunner.SEED)
        torch.npu.manual_seed(ProfilingRunner.SEED)
        
        return [
            x.to(device) if isinstance(x, torch.Tensor) else x 
            for x in get_inputs_func()
        ]

    def _build_final_result(self, op_name: str, config: NPUProfilerConfig | None, 
                           framework_result: ProfilingResult, 
                           impl_result: ProfilingResult) -> dict:
        """构建最终结果"""
        config = config or NPUProfilerConfig()
        speedup_self, speedup_total = SpeedupCalculator.calculate(framework_result, impl_result)
        
        # 格式化 operator_details 为可序列化的格式
        def format_details(result: ProfilingResult) -> list[dict]:
            return [
                {
                    'name': detail.name,
                    'avg_device_self_duration_us': detail.avg_device_self_duration_us,
                    'avg_device_total_duration_us': detail.avg_device_total_duration_us
                }
                for detail in result.operator_details[:ProfilingRunner.MAX_OPERATORS_DISPLAY]
            ]
            
        return {
            'op_name': op_name,
            'num_iterations': config.num_iterations,
            'warm_up': config.warm_up,
            'timestamp': int(time.time()),
            'profiling': {
                'framework': {
                    'op_name': f"{op_name}_torch",
                    'avg_device_self_duration_us': round(framework_result.avg_device_self_duration_us, 4),
                    'avg_device_total_duration_us': round(framework_result.avg_device_total_duration_us, 4),
                    'device_self_duration_us': framework_result.device_self_duration_us,
                    'device_total_duration_us': framework_result.device_total_duration_us,
                    'operator_details': format_details(framework_result),
                    'collection_metadata': {
                        'collection_time_seconds': round(framework_result.collection_time_seconds, 3),
                        'profiling_overhead_seconds': round(framework_result.profiling_overhead_seconds, 3),
                        'pure_execution_time_seconds': round(framework_result.pure_execution_time_seconds, 3),
                        'overhead_percentage': round(framework_result.overhead_percentage, 1),
                        'raw_iterations': framework_result.raw_iterations,
                        'actual_profiling_iterations': framework_result.actual_profiling_iterations
                    }
                },
                'implementation': {
                    'op_name': f"{op_name}_triton_ascend_impl",
                    'avg_device_self_duration_us': round(impl_result.avg_device_self_duration_us, 4),
                    'avg_device_total_duration_us': round(impl_result.avg_device_total_duration_us, 4),
                    'device_self_duration_us': impl_result.device_self_duration_us,
                    'device_total_duration_us': impl_result.device_total_duration_us,
                    'operator_details': format_details(impl_result),
                    'collection_metadata': {
                        'collection_time_seconds': round(impl_result.collection_time_seconds, 3),
                        'profiling_overhead_seconds': round(impl_result.profiling_overhead_seconds, 3),
                        'pure_execution_time_seconds': round(impl_result.pure_execution_time_seconds, 3),
                        'overhead_percentage': round(impl_result.overhead_percentage, 1),
                        'raw_iterations': impl_result.raw_iterations,
                        'actual_profiling_iterations': impl_result.actual_profiling_iterations
                    }
                }
            },
            'speedup': {
                'device_self_duration_speedup': round(speedup_self, 2),
                'device_total_duration_speedup': round(speedup_total, 2)
            }
        }


class ResultDisplay:
    """结果展示器"""
    
    @staticmethod
    def print_summary(result: dict) -> None:
        """打印结果摘要"""
        fw = result['profiling']['framework']
        impl = result['profiling']['implementation']
        speedup = result['speedup']
        
        print(f"\nProfiling 结果摘要:")
        print(f"  框架实现 - 平均 Device Self Duration: {fw['avg_device_self_duration_us']:.4f} us")
        print(f"  框架实现 - 平均 Device Total Duration: {fw['avg_device_total_duration_us']:.4f} us")
        print(f"  生成实现 - 平均 Device Self Duration: {impl['avg_device_self_duration_us']:.4f} us")
        print(f"  生成实现 - 平均 Device Total Duration: {impl['avg_device_total_duration_us']:.4f} us")
        print(f"  加速比 (Self): {speedup['device_self_duration_speedup']:.2f}x")
        print(f"  加速比 (Total): {speedup['device_total_duration_speedup']:.2f}x")

    @staticmethod
    def save_to_file(result: dict, output_path: str | Path | None) -> None:
        """保存结果到文件或打印"""
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {output_path}")
        else:
            print("\n完整结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="NPU Profiler 性能测试脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument("--verify_dir", default=".", help="验证目录路径")
    parser.add_argument("--num_iterations", type=int, default=50, help="迭代次数（默认 50）")
    parser.add_argument("--warm_up", type=int, default=5, help="warmup 次数（默认 5）")
    parser.add_argument("--output", help="输出文件路径（JSON 格式）")
    args = parser.parse_args()
    
    verify_dir = Path(args.verify_dir).resolve()
    if not verify_dir.is_dir():
        print(f"错误: 验证目录不存在: {verify_dir}", file=sys.stderr)
        sys.exit(1)
    try:
        config = NPUProfilerConfig(
            num_iterations=args.num_iterations,
            warm_up=args.warm_up
        )
        
        runner = ProfilingRunner()
        result = runner.run(args.op_name, str(verify_dir), config)
        
        ResultDisplay().print_summary(result)
        ResultDisplay().save_to_file(result, args.output)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"Profiling 失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
