#!/usr/bin/env python3
"""
AscendC 算子多形状测试工具

根据 benchmark JSON 配置文件测试算子在所有输入形状下的正确性和性能。

用法:
    python evaluate.py \
        --op_name GELU \
        --benchmark_file /path/to/1_GELU.py \
        --benchmark_json /path/to/1_GELU.json \
        --output_dir /path/to/output/GELU \
        --warmup 5 \
        --repeats 50

参数:
    --op_name: 算子名称（必需）
    --benchmark_file: Benchmark .py 文件路径（必需）
    --benchmark_json: Benchmark .json 文件路径（必需）
    --output_dir: 输出目录路径（默认为 output/<op_name>）
    --warmup: Warmup 迭代次数（默认 5）
    --repeats: 性能测试重复次数（默认 50）
    --skip_setup: 跳过环境设置
"""

import os
import sys
import json
import logging
import argparse
import statistics
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
import torch_npu


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)


@dataclass
class TestCase:
    """单个测试用例"""
    case_id: int
    inputs_config: List[Dict[str, Any]]
    
    def get_tensor_inputs(self) -> List[torch.Tensor]:
        """根据配置生成张量输入"""
        tensors = []
        for inp in self.inputs_config:
            if inp.get("type") == "tensor":
                dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "int32": torch.int32,
                    "int64": torch.int64,
                }
                dtype = dtype_map.get(inp.get("dtype", "float32"), torch.float32)
                shape = inp["shape"]
                tensors.append(torch.randn(shape, dtype=dtype))
        return tensors
    
    def get_attr_inputs(self) -> List[Any]:
        """获取属性参数"""
        attrs = []
        for inp in self.inputs_config:
            if inp.get("type") == "attr":
                value = inp.get("value")
                dtype = inp.get("dtype", "str")
                if dtype == "int":
                    value = int(value) if value is not None else 0
                elif dtype == "float":
                    value = float(value) if value is not None else 0.0
                elif dtype == "bool":
                    value = bool(value) if value is not None else False
                attrs.append(value)
        return attrs
    
    def get_all_inputs(self) -> List[Any]:
        """按原始顺序获取所有输入（tensor 和 attr 混合）"""
        inputs = []
        tensor_idx = 0
        attr_idx = 0
        tensors = self.get_tensor_inputs()
        attrs = self.get_attr_inputs()
        
        for inp in self.inputs_config:
            if inp.get("type") == "tensor":
                inputs.append(tensors[tensor_idx])
                tensor_idx += 1
            elif inp.get("type") == "attr":
                inputs.append(attrs[attr_idx])
                attr_idx += 1
        return inputs
    
    def get_shape_info(self) -> str:
        """获取形状信息字符串"""
        shapes = []
        dtypes = []
        attrs = {}
        for inp in self.inputs_config:
            if inp.get("type") == "tensor":
                shapes.append(inp["shape"])
                dtypes.append(inp.get("dtype", "float32"))
            elif inp.get("type") == "attr":
                attrs[inp["name"]] = inp.get("value")
        return f"shapes={shapes}, dtypes={dtypes}, attrs={attrs}"


@dataclass
class TestResult:
    """单个测试结果"""
    case_id: int
    shape_info: str
    correctness: bool
    error_message: Optional[str] = None
    ref_time_ms: Optional[float] = None
    custom_time_ms: Optional[float] = None
    speedup: Optional[float] = None
    ref_avg_ms: Optional[float] = None
    ref_p50_ms: Optional[float] = None
    ref_p99_ms: Optional[float] = None
    custom_avg_ms: Optional[float] = None
    custom_p50_ms: Optional[float] = None
    custom_p99_ms: Optional[float] = None
    peak_memory_mb: Optional[float] = None


@dataclass
class TestReport:
    """测试报告"""
    op_name: str
    total_cases: int
    passed: int
    failed: int
    pass_rate: float
    avg_speedup: Optional[float]
    details: List[TestResult]


def install_run_file(work_dir: Path) -> bool:
    """安装 .run 文件"""
    work_dir = work_dir.resolve()
    run_file_pattern = work_dir.glob("*Custom/build_out/custom_opp_ubuntu_aarch64.run")
    run_files = list(run_file_pattern)
    
    if not run_files:
        logging.warning(f"No .run file found in {work_dir}")
        return False
    
    run_file = run_files[0]
    logging.info(f"Found run file: {run_file}")
    
    try:
        subprocess.run(["chmod", "+x", str(run_file)], check=True)
        result = subprocess.run(
            [str(run_file), "--install-path", str(work_dir)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            logging.info(f"Run file installed successfully")
            return True
        else:
            logging.error(f"Run file installation failed: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"Run file installation error: {e}")
        return False


def generate_pybind_bindings(work_dir: Path, op_name: str) -> bool:
    """生成 PyBind 绑定"""
    template_dir = Path(__file__).parent.joinpath("template")
    target_dir = work_dir.joinpath("ascend_op_pybind")
    
    if not template_dir.exists():
        logging.error(f"Template directory not found: {template_dir}")
        return False
    
    op_cpp = work_dir.joinpath(f"{op_name}.cpp")
    if not op_cpp.exists():
        logging.error(f"op.cpp not found: {op_cpp}")
        return False
    
    try:
        if not target_dir.exists():
            logging.info(f"Copying template directory to: {target_dir}")
            shutil.copytree(template_dir, target_dir)
        
        cpp_path = target_dir.joinpath("CppExtension/csrc/op.cpp")
        cpp_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(op_cpp, cpp_path)
        logging.info(f"Copied op.cpp to: {cpp_path}")
        
        extension_dir = target_dir.joinpath("CppExtension")
        result = subprocess.run(
            [sys.executable, 'setup.py', 'build', 'bdist_wheel'],
            cwd=str(extension_dir),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            logging.error(f"Build failed: {result.stderr}")
            return False
        
        logging.info("Build wheel package successfully")
        
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
                    logging.info(f"Installed {wheel_file.name} successfully")
                else:
                    logging.error(f"Install {wheel_file.name} failed: {result_install.stderr}")
        
        return True
    except Exception as e:
        logging.error(f"PyBind generation error: {e}")
        return False


def setup_environment(work_dir: Path, op_name: str) -> bool:
    """设置运行时环境"""
    work_dir = work_dir.resolve()
    
    if not install_run_file(work_dir):
        logging.warning("Run file installation skipped or failed")
    
    if not generate_pybind_bindings(work_dir, op_name):
        logging.warning("PyBind generation skipped or failed")
    
    custom_opp_path = work_dir.joinpath("vendors/customize")
    if custom_opp_path.exists():
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp_path)
        logging.info(f"Set ASCEND_CUSTOM_OPP_PATH={custom_opp_path}")
        
        custom_lib_path = custom_opp_path.joinpath("op_api/lib").resolve()
        if custom_lib_path.exists():
            custom_lib_path_str = str(custom_lib_path)
            existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if custom_lib_path_str not in existing_ld_path:
                new_ld_path = f"{custom_lib_path_str}:{existing_ld_path}".rstrip(":")
                os.environ["LD_LIBRARY_PATH"] = new_ld_path
                logging.info(f"Updated LD_LIBRARY_PATH to include: {custom_lib_path_str}")
    
    return True


class MultiShapeTester:
    """多形状测试器"""
    
    def __init__(
        self,
        op_name: str,
        benchmark_file: Path,
        custom_code_path: Path,
        project_root: Path,
        device: str = "npu:0",
        seed: int = 1024,
        warmup: int = 5,
        repeats: int = 50
    ):
        self.op_name = op_name
        self.benchmark_file = benchmark_file
        self.custom_code_path = custom_code_path
        self.project_root = project_root
        self.device = torch.device(device)
        self.seed = seed
        self.warmup = warmup
        self.repeats = repeats
        self.context = {}
        
        self._load_code()
    
    def _load_code(self):
        """加载 benchmark 和自定义算子代码"""
        if not self.benchmark_file.exists():
            raise FileNotFoundError(f"Benchmark file not found: {self.benchmark_file}")
        if not self.custom_code_path.exists():
            raise FileNotFoundError(f"Custom code file not found: {self.custom_code_path}")
        
        benchmark_code = self.benchmark_file.read_text(encoding='utf-8')
        custom_code = self.custom_code_path.read_text(encoding='utf-8')
        
        try:
            exec(benchmark_code, self.context)
            exec(custom_code, self.context)
        except Exception as e:
            raise RuntimeError(f"Failed to load code: {str(e)}")
        
        required = ['Model', 'ModelNew']
        for name in required:
            if name not in self.context:
                raise RuntimeError(f"Missing required component: {name}")
    
    def _synchronize(self):
        torch_npu.npu.synchronize()
    
    def _move_to_device(self, data):
        if isinstance(data, list):
            return [self._move_to_device(x) for x in data]
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return data
    
    def _create_models(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        ref_model = self.context['Model']().to(self.device)
        custom_model = self.context['ModelNew']().to(self.device)
        self._synchronize()
        return ref_model, custom_model
    
    def _check_correctness(
        self,
        ref_model: torch.nn.Module,
        custom_model: torch.nn.Module,
        inputs: List[Any]
    ) -> Tuple[bool, str]:
        """检查正确性"""
        inputs_device = self._move_to_device(inputs)
        
        with torch.no_grad():
            ref_output = ref_model(*inputs_device)
            custom_output = custom_model(*inputs_device)
        self._synchronize()
        
        if isinstance(ref_output, (list, tuple)):
            ref_output = ref_output[0] if len(ref_output) > 0 else ref_output
        if isinstance(custom_output, (list, tuple)):
            custom_output = custom_output[0] if len(custom_output) > 0 else custom_output
        
        ref_output = ref_output.to("cpu")
        custom_output = custom_output.to("cpu")
        
        if ref_output.shape != custom_output.shape:
            return False, f"Shape mismatch: expected {ref_output.shape}, got {custom_output.shape}"
        
        if ref_output.dtype in (torch.float16, torch.bfloat16):
            ref_output = ref_output.float()
            custom_output = custom_output.float()
        
        atol, rtol = 1e-2, 1e-2
        if torch.allclose(ref_output, custom_output, atol=atol, rtol=rtol):
            max_diff = (ref_output - custom_output).abs().max().item()
            return True, f"PASS (max_diff={max_diff:.5e})"
        else:
            close_mask = torch.isclose(ref_output, custom_output, atol=atol, rtol=rtol)
            match_rate = close_mask.sum().item() / close_mask.numel()
            return False, f"Mismatch (match_rate={match_rate*100:.2f}%)"
    
    def _measure_performance(
        self,
        model: torch.nn.Module,
        inputs: List[Any]
    ) -> Dict[str, float]:
        """测量性能，返回详细指标"""
        inputs_device = self._move_to_device(inputs)
        event_class = torch_npu.npu.Event
        elapsed_times = []
        
        with torch.no_grad():
            for _ in range(self.warmup):
                model(*inputs_device)
                self._synchronize()
            
            for _ in range(self.repeats):
                start_event = event_class(enable_timing=True)
                end_event = event_class(enable_timing=True)
                start_event.record()
                model(*inputs_device)
                end_event.record()
                self._synchronize()
                elapsed_times.append(start_event.elapsed_time(end_event))
        
        sorted_times = sorted(elapsed_times)
        n = len(sorted_times)
        
        return {
            "median": statistics.median(elapsed_times),
            "avg": statistics.mean(elapsed_times),
            "p50": sorted_times[n // 2] if n % 2 == 1 else (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2,
            "p99": sorted_times[int(n * 0.99)] if n > 1 else sorted_times[0],
        }
    
    def run_test(self, test_case: TestCase) -> TestResult:
        """运行单个测试用例"""
        set_seed(self.seed)
        
        inputs = test_case.get_all_inputs()
        ref_model, custom_model = self._create_models()
        
        correctness, message = self._check_correctness(ref_model, custom_model, inputs)
        
        result = TestResult(
            case_id=test_case.case_id,
            shape_info=test_case.get_shape_info(),
            correctness=correctness,
            error_message=None if correctness else message
        )
        
        if correctness:
            torch_npu.npu.reset_peak_memory_stats()
            
            ref_stats = self._measure_performance(ref_model, inputs)
            custom_stats = self._measure_performance(custom_model, inputs)
            
            peak_memory = torch_npu.npu.max_memory_allocated() / (1024 * 1024)
            
            result.ref_time_ms = ref_stats["median"]
            result.custom_time_ms = custom_stats["median"]
            result.speedup = ref_stats["median"] / custom_stats["median"] if custom_stats["median"] > 0 else float('inf')
            result.ref_avg_ms = ref_stats["avg"]
            result.ref_p50_ms = ref_stats["p50"]
            result.ref_p99_ms = ref_stats["p99"]
            result.custom_avg_ms = custom_stats["avg"]
            result.custom_p50_ms = custom_stats["p50"]
            result.custom_p99_ms = custom_stats["p99"]
            result.peak_memory_mb = peak_memory
        
        del ref_model, custom_model
        torch_npu.npu.empty_cache()
        self._synchronize()
        
        return result


def load_test_cases(json_path: Path) -> List[TestCase]:
    """从 JSON 文件加载测试用例"""
    test_cases = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                config = json.loads(line)
                test_cases.append(TestCase(
                    case_id=idx,
                    inputs_config=config.get("inputs", [])
                ))
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse line {idx}: {e}")
    return test_cases


def run_all_tests(
    op_name: str,
    benchmark_file: Path,
    benchmark_json: Path,
    custom_code_path: Path,
    project_root: Path,
    warmup: int = 5,
    repeats: int = 50,
    skip_setup: bool = False
) -> TestReport:
    """运行所有测试"""
    if not skip_setup:
        logging.info("Setting up environment...")
        setup_environment(project_root, op_name)
    
    logging.info(f"Loading test cases from: {benchmark_json}")
    test_cases = load_test_cases(benchmark_json)
    logging.info(f"Total test cases: {len(test_cases)}")
    
    tester = MultiShapeTester(
        op_name=op_name,
        benchmark_file=benchmark_file,
        custom_code_path=custom_code_path,
        project_root=project_root,
        warmup=warmup,
        repeats=repeats
    )
    
    results = []
    passed = 0
    failed = 0
    speedups = []
    
    print("=" * 80)
    print(f"Operator: {op_name}")
    print(f"Total test cases: {len(test_cases)}")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test_case.get_shape_info()}")
        
        try:
            result = tester.run_test(test_case)
            results.append(result)
            
            if result.correctness:
                passed += 1
                print(f"  Correctness: PASS")
                print(f"  Performance: ref={result.ref_time_ms:.3f}ms, "
                      f"custom={result.custom_time_ms:.3f}ms, "
                      f"speedup={result.speedup:.2f}x")
                speedups.append(result.speedup)
            else:
                failed += 1
                print(f"  Correctness: FAIL - {result.error_message}")
        except Exception as e:
            failed += 1
            result = TestResult(
                case_id=test_case.case_id,
                shape_info=test_case.get_shape_info(),
                correctness=False,
                error_message=str(e)
            )
            results.append(result)
            print(f"  Error: {str(e)}")
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total: {len(test_cases)} | Passed: {passed} | Failed: {failed}")
    print(f"Pass Rate: {passed/len(test_cases)*100:.2f}%")
    if speedups:
        print(f"Avg Speedup: {statistics.mean(speedups):.2f}x")
    print("=" * 80)
    
    return TestReport(
        op_name=op_name,
        total_cases=len(test_cases),
        passed=passed,
        failed=failed,
        pass_rate=passed/len(test_cases) if test_cases else 0,
        avg_speedup=statistics.mean(speedups) if speedups else None,
        details=results
    )


def save_report(report: TestReport, output_path: Path):
    """保存测试报告"""
    report_dict = {
        "op_name": report.op_name,
        "total_cases": report.total_cases,
        "passed": report.passed,
        "failed": report.failed,
        "pass_rate": report.pass_rate,
        "avg_speedup": report.avg_speedup,
        "details": [asdict(r) for r in report.details]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Report saved to: {output_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Test AscendC operator against all shapes in benchmark JSON"
    )
    parser.add_argument("--op_name", type=str, required=True, help="Operator name")
    parser.add_argument("--benchmark_file", type=str, required=True, help="Path to benchmark .py file")
    parser.add_argument("--benchmark_json", type=str, required=True, help="Path to benchmark .json file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory path (default: output/<op_name>)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=50, help="Performance test repeats")
    parser.add_argument("--skip_setup", action="store_true", help="Skip environment setup")
    
    args = parser.parse_args()
    
    benchmark_file = Path(args.benchmark_file)
    benchmark_json = Path(args.benchmark_json)
    
    if args.output_dir:
        project_root = Path(args.output_dir).resolve()
    else:
        project_root = Path("output").joinpath(args.op_name).resolve()
    
    custom_code_path = project_root.joinpath(f"{args.op_name}_custom.py")
    
    report = run_all_tests(
        op_name=args.op_name,
        benchmark_file=benchmark_file,
        benchmark_json=benchmark_json,
        custom_code_path=custom_code_path,
        project_root=project_root,
        warmup=args.warmup,
        repeats=args.repeats,
        skip_setup=args.skip_setup
    )
    
    output_path = project_root / "test_report.json"
    save_report(report, output_path)


if __name__ == "__main__":
    main()
