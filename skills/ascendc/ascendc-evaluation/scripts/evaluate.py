#!/usr/bin/env python3
"""
AscendC 自定义算子评估工具

用于评估自定义算子的正确性和性能表现。

命令行用法:
    python evaluate.py <op_name>

示例:
    python evaluate.py Add
    python evaluate.py MyCustomOperator

API 用法:
    from evaluate import AscendBackend

    # 初始化后端
    backend = AscendBackend(eval_code, ref_code)

    # 1. 精度评估
    success, message = backend.evaluate_correctness()
    # 返回: (bool, str) - 是否通过及详细信息

    # 2. 性能测试 - 单个模型
    median_time = backend.measure_performance('ModelNew', num_warmup=10, num_perf_trials=100)
    # 返回: float - 中位数耗时(毫秒)

    # 3. 性能对比 - 两个模型
    ref_median, custom_median = backend.compare_performance(num_warmup=10, num_perf_trials=100)
    # 返回: (float, float) - 参考模型和自定义模型的中位数耗时(毫秒)

输入代码要求:
    评估代码必须包含以下组件:
    - Model: 参考实现类 (torch.nn.Module)
    - ModelNew: 自定义算子实现类 (torch.nn.Module)
    - get_inputs(): 返回测试输入数据的函数
    - get_init_inputs(): 返回模型初始化参数的函数

目录结构:
    output/<op_name>/
        <op_name>_reference.py  # 参考代码
        <op_name>_custom.py     # 自定义算子代码

环境要求:
    - vendors/customize/op_api/lib/ 库文件必须存在
    - 需要设置 ASCEND_CUSTOM_OPP_PATH 环境变量
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch_npu


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)


class AscendBackend:
    def __init__(self, eval_src: str, ref_src: str, seed_num: int = 1024, num_correct_trials: int = 5):
        self.context = {}
        self.device = torch.device('npu:0')
        self.seed_num = seed_num
        self.num_correct_trials = num_correct_trials
        self._set_context(eval_src, ref_src)
    
    def _set_context(self, eval_src: str, ref_src: str):
        try:
            exec(eval_src, self.context)
            exec(ref_src, self.context)
        except Exception as e:
            raise RuntimeError(f"Failed to compile reference model: {str(e)}")
    
    def _synchronize(self):
        """Synchronize NPU operations."""
        torch_npu.npu.synchronize()
    
    def _move_to_device(self, data):
        """Move tensor data to NPU device."""
        if isinstance(data, list):
            return [self._move_to_device(x) for x in data]
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return data
    
    def _prepare_inputs(self):
        """Get inputs from context and move to device."""
        get_inputs = self.context["get_inputs"]
        inputs = get_inputs()
        return self._move_to_device(inputs)
    
    def _prepare_init_inputs(self):
        """Get init_inputs from context and move to device."""
        get_init_inputs = self.context["get_init_inputs"]
        init_inputs = get_init_inputs()
        return self._move_to_device(init_inputs)
    
    def _create_model(self, model_name='ModelNew', init_inputs=None):
        """Create model instance and move to device.

        Args:
            model_name: Name of the model class in context
            init_inputs: Optional pre-prepared init inputs. If None, will call _prepare_init_inputs()
        """
        ModelClass = self.context[model_name]
        if init_inputs is None:
            init_inputs = self._prepare_init_inputs()

        model = ModelClass(*init_inputs).to(self.device)
        self._synchronize()
        return model

    def _normalize_output(self, output, index):
        """Extract single output tensor from list/tuple or return as-is."""
        if isinstance(output, (list, tuple)):
            return output[index]
        return output

    def _check_shape(self, ref_output, new_output, output_idx):
        """Check if output shapes match. Returns error message or None."""
        if ref_output.shape != new_output.shape:
            return f"[FAIL] Output shape mismatch at output {output_idx}: Expected {ref_output.shape}, got {new_output.shape}"
        return None

    def _check_values(self, ref_output, new_output, output_idx, atol=1e-02, rtol=1e-02):
        """Check if output values are close. Returns (error_msg, pass_info) tuple."""
        if ref_output.dtype in (torch.float16, torch.bfloat16):
            ref_output = ref_output.float()
            new_output = new_output.float()
        close_mask = torch.isclose(ref_output, new_output, atol=atol, rtol=rtol)
        total = close_mask.numel()
        matched = close_mask.sum().item()
        match_rate = matched / total

        if torch.allclose(ref_output, new_output, atol=atol, rtol=rtol):
            max_diff = (ref_output - new_output).abs().max().item()
            mean_diff = (ref_output - new_output).abs().float().mean().item()
            pass_info = (
                f"Output {output_idx}: shape={list(ref_output.shape)}, "
                f"match_rate=100.00% ({matched}/{total}), "
                f"max_diff={max_diff:.5e}, mean_diff={mean_diff:.5e}"
            )
            return None, pass_info

        mismatch_idx = (~close_mask).nonzero(as_tuple=False)[0]
        mismatch_idx_tuple = tuple(mismatch_idx.tolist())
        ref_val = ref_output[mismatch_idx_tuple].item()
        new_val = new_output[mismatch_idx_tuple].item()

        error_msg = (
            f"[FAIL] Output {output_idx} mismatch\n"
            f"Match rate: {match_rate * 100:.2f}% ({matched}/{total})\n"
            f"Example mismatch at index {tuple(mismatch_idx.tolist())}: "
            f"ref={ref_val}, new={new_val}"
        )
        return error_msg, None

    def evaluate_correctness(self):
        """Execute correctness check between reference and custom models."""
        try:
            set_seed(self.seed_num)
            inputs = self._prepare_inputs()
            init_inputs = self._prepare_init_inputs()
            ref_model = self._create_model('Model', init_inputs)
            new_model = self._create_model('ModelNew', init_inputs)
            with torch.no_grad():
                ref_output = ref_model(*inputs)
                new_output = new_model(*inputs)
            self._synchronize()

            has_error, message = self._compare_outputs(ref_output, new_output)

            if has_error:
                return False, message
            return True, message

        except Exception as e:
            logging.error("[FAIL] runtime error when evaluating correctness")
            return False, f"[FAIL] {str(e)}"

    def _compare_outputs(self, ref_output, new_output):
        """Compare model outputs and return (has_error, message)."""
        error_parts = []
        pass_parts = []
        num_outputs = len(ref_output) if isinstance(ref_output, (list, tuple)) else 1

        for i in range(num_outputs):
            ref_out = self._normalize_output(ref_output, i).to("cpu")
            new_out = self._normalize_output(new_output, i).to("cpu")

            error = self._check_shape(ref_out, new_out, i)
            if error:
                error_parts.append(error)
                continue

            error, pass_info = self._check_values(ref_out, new_out, i)
            if error:
                error_parts.append(error)
            else:
                pass_parts.append(pass_info)

        if error_parts:
            return True, "\n".join(error_parts)
        return False, "[PASS]\n" + "\n".join(pass_parts)

    def measure_performance(self, model_name='ModelNew', num_warmup=10, num_perf_trials=100):
        """Measure performance for specified model.

        Args:
            model_name: Model name to measure ('Model' or 'ModelNew')
            num_warmup: Number of warmup iterations
            num_perf_trials: Number of performance measurement iterations

        Returns:
            Median elapsed time in milliseconds
        """
        import statistics

        event_class = torch_npu.npu.Event
        elapsed_times = []

        model = self._create_model(model_name)
        inputs = self._prepare_inputs()

        with torch.no_grad():
            def _run_performance_test(kernel_fn, times_list):
                for _ in range(num_warmup):
                    kernel_fn(*inputs)
                    self._synchronize()
                for _ in range(num_perf_trials):
                    start_event = event_class(enable_timing=True)
                    end_event = event_class(enable_timing=True)
                    start_event.record()
                    kernel_fn(*inputs)
                    end_event.record()
                    self._synchronize()
                    elapsed_time_ms = start_event.elapsed_time(end_event)
                    times_list.append(elapsed_time_ms)

            _run_performance_test(model, elapsed_times)

        return statistics.median(elapsed_times)

    def compare_performance(self, num_warmup=10, num_perf_trials=100):
        """Compare performance between reference and custom models.

        Args:
            num_warmup: Number of warmup iterations
            num_perf_trials: Number of performance measurement iterations

        Returns:
            Tuple of (ref_median_time, custom_median_time) in milliseconds
        """
        ref_median = self.measure_performance('Model', num_warmup, num_perf_trials)
        custom_median = self.measure_performance('ModelNew', num_warmup, num_perf_trials)

        return ref_median, custom_median

    def cleanup(self):
        del self.context
        torch_npu.npu.empty_cache()
        self._synchronize()

def setup_ascend_runtime_environment(project_root: Path) -> None:
    """
    设置 AscendC 自定义算子所需的运行时环境变量。

    Args:
        project_root (Path): 项目根目录路径
    Sets:
        - ASCEND_CUSTOM_OPP_PATH
        - LD_LIBRARY_PATH (追加自定义库路径)
    """
    project_root = Path(project_root).resolve()

    # 1. 设置 ASCEND_CUSTOM_OPP_PATH
    custom_opp_path = project_root.joinpath("vendors/customize")
    
    if not custom_opp_path.exists():
        raise FileNotFoundError(f"ASCEND custom OPP directory not found: {custom_opp_path}")
    
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp_path)
    logging.info(f"Set ASCEND_CUSTOM_OPP_PATH={custom_opp_path}")

    # 2. 更新 LD_LIBRARY_PATH
    custom_lib_path = Path(custom_opp_path).joinpath("op_api/lib").resolve()
    
    if not custom_lib_path.exists():
        raise FileNotFoundError(f"ASCEND custom OP API library directory not found: {custom_lib_path}")
    
    custom_lib_path_str = str(custom_lib_path)

    existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    # 避免重复添加（防止路径膨胀）
    if custom_lib_path_str not in existing_ld_path:
        new_ld_path = f"{custom_lib_path_str}:{existing_ld_path}".rstrip(":")
        os.environ["LD_LIBRARY_PATH"] = new_ld_path
        logging.info(f"Updated LD_LIBRARY_PATH to include: {custom_lib_path_str}")
    else:
        logging.info(f"LD_LIBRARY_PATH already contains: {custom_lib_path_str}")

def evaluate_operator(
    eval_src_path: Path,
    ref_src_path: Path,
    project_root_path: Path
) -> Tuple[bool, str]:
    """
    评估算子代码正确性。

    Args:
        eval_src (Path): 要评估的代码文件路径（包含自定义算子实现的 ModelNew 类）
        ref_src (Path): 参考代码文件路径（包含参考实现的 Model 类）
        project_root_path (Path): 项目根目录路径，用于设置运行时环境

    Returns:
        Tuple of (success, output_message):
            - success (bool): 评估是否成功
            - output_message (str): 评估结果详情或错误信息

    评估代码文件必须包含以下组件：
        - Model: 参考实现类（继承 torch.nn.Module）
        - ModelNew: 自定义算子实现类（继承 torch.nn.Module）
        - get_inputs(): 生成测试输入数据的函数
        - get_init_inputs(): 生成初始化参数的函数
    """
    # 读取文件内容
    setup_ascend_runtime_environment(project_root_path)

    if not eval_src_path.exists():
        raise FileNotFoundError(f'Evaluation code file not found: {eval_src_path}')
    if not ref_src_path.exists():
        raise FileNotFoundError(f'Reference code file not found: {ref_src_path}')

    eval_code = eval_src_path.read_text(encoding='utf-8')
    ref_code = ref_src_path.read_text(encoding='utf-8')

    ascend_backend = AscendBackend(eval_code,ref_code)
    flag_bool, run_info = ascend_backend.evaluate_correctness()
    logging.info(f"Evaluation correctness: {run_info}")

    if flag_bool:
        ref_median, custom_median = ascend_backend.compare_performance()
        speedup = ref_median / custom_median if custom_median > 0 else float('inf')
        logging.info(f"Evaluation performance: ref={ref_median:.3f}ms, custom={custom_median:.3f}ms, speedup={speedup:.2f}x")
    return flag_bool, run_info

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Evaluate AscendC custom operator correctness")
    parser.add_argument("op_name", type=str, help="Operator name")
    
    args = parser.parse_args()
    
    try:
        work_dir = Path("output").joinpath(args.op_name).resolve()
        eval_src = work_dir.joinpath(f"{args.op_name}_custom.py")
        ref_src = work_dir.joinpath(f"{args.op_name}_reference.py")
        evaluate_operator(eval_src, ref_src, work_dir)
    except Exception as e:
        logging.error(f"Evaluation error: {e}")
