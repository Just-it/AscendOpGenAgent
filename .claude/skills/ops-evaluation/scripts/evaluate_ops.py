#!/usr/bin/env python3
"""
ops仓算子 baseline vs evolved 对比评估工具。

通过子进程隔离分别评估 baseline 和 evolved 两个版本的精度和性能，
然后合并结果生成对比报告。

使用子进程隔离的原因：CANN 运行时加载 OPP 库后无法在同一进程中切换到另一个版本。

用法:
    python evaluate_ops.py {op_name} \
        --baseline-path {baseline_install_path} \
        --evolved-path {evolved_install_path} \
        --reference-py {reference.py} \
        --custom-py {custom.py} \
        --device-id 0 \
        --task-type vector

输出:
    evaluation_results.json - baseline vs evolved 对比报告
"""

import argparse
import fcntl
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
# evaluate.py lives in the ascendc-evaluation scripts dir
EVALUATE_SCRIPT = SCRIPT_DIR.parent.parent / "ascendc-evaluation" / "scripts" / "evaluate.py"


def _acquire_eval_lock(lock_path: str, timeout: float = 300) -> int:
    """阻塞获取评估排队锁。返回 fd。"""
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except (BlockingIOError, OSError):
            if time.monotonic() >= deadline:
                os.close(fd)
                raise TimeoutError(
                    f"Failed to acquire eval lock {lock_path} within {timeout}s"
                )
            time.sleep(1)


def _release_eval_lock(fd: int):
    """释放评估排队锁。"""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        pass


def detect_vendor_subdir(install_path: str) -> str:
    """检测安装后的 vendors 子目录名。"""
    vendors_dir = os.path.join(install_path, "vendors")
    if os.path.isdir(vendors_dir):
        subdirs = [d for d in os.listdir(vendors_dir)
                    if os.path.isdir(os.path.join(vendors_dir, d))
                    and d.startswith("custom")]
        if subdirs:
            return subdirs[0]
    if os.path.isdir(os.path.join(vendors_dir, "customize")):
        return "customize"
    return "custom_nn"


def evaluate_single_version(
    op_name: str,
    install_path: str,
    reference_py: str,
    custom_py: str,
    device_id: int,
    task_type: str,
    profile_dir: str,
    num_trials: int,
    tag: str,
) -> dict:
    """
    在独立子进程中评估单个版本。

    通过生成一个临时 Python 脚本，在子进程中设置 OPP 环境变量后
    调用 AscendBackend 进行精度和性能评估。

    Args:
        op_name: 算子名
        install_path: 安装路径（含 vendors/ 目录）
        reference_py: 参考实现 Python 文件路径
        custom_py: 自定义算子 Python 文件路径
        device_id: NPU 设备 ID
        task_type: 算子类型 (vector/cube/cv-mix/unknown)
        profile_dir: profiling 输出目录
        num_trials: profiling 试验次数
        tag: 版本标签 ("baseline" / "evolved")

    Returns:
        dict: 评估结果
    """
    vendor_subdir = detect_vendor_subdir(install_path)
    opp_path = os.path.join(install_path, "vendors", vendor_subdir)
    lib_path = os.path.join(opp_path, "op_api", "lib")

    if not os.path.isdir(opp_path):
        return {
            "tag": tag,
            "error": f"OPP directory not found: {opp_path}",
            "precision_passed": False,
            "time_us": -1,
        }

    # 创建 profiling 目录
    os.makedirs(profile_dir, exist_ok=True)

    # 生成评估子脚本
    eval_script = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import os
        import sys
        import json
        import logging

        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

        # 设置 OPP 环境变量
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = {repr(opp_path)}

        # 添加 op_api/lib 到 LD_LIBRARY_PATH
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = {repr(lib_path)} + ":" + existing_ld

        # 设置设备
        os.environ["ASCEND_DEVICE_ID"] = str({device_id})

        # 添加 evaluate.py 所在目录到 sys.path
        eval_script_dir = {repr(str(EVALUATE_SCRIPT.parent))}
        if eval_script_dir not in sys.path:
            sys.path.insert(0, eval_script_dir)

        from pathlib import Path
        import torch
        import torch_npu

        # 设置 NPU 设备
        torch.npu.set_device({device_id})

        # 导入 AscendBackend
        from evaluate import AscendBackend

        # 读取代码
        ref_code = Path({repr(reference_py)}).read_text()
        custom_code = Path({repr(custom_py)}).read_text()

        # 注入 pybind_lib（如果存在）
        pybind_lib = os.path.join({repr(install_path)}, "pybind_lib")
        if os.path.isdir(pybind_lib) and pybind_lib not in sys.path:
            sys.path.insert(0, pybind_lib)

        # 创建后端
        backend = AscendBackend(custom_code, ref_code)

        result = {{
            "tag": {repr(tag)},
            "install_path": {repr(install_path)},
            "precision_passed": False,
            "correctness_message": "",
            "time_us": -1,
            "pipeline": {{}},
            "bottleneck": "unknown",
            "profiling_dir": {repr(profile_dir)},
        }}

        # 1. 精度评估
        try:
            success, message = backend.evaluate_correctness()
            result["precision_passed"] = success
            result["correctness_message"] = message
            tag_str = result["tag"]
            logging.info(f"[{{tag_str}}] 精度: {{'PASS' if success else 'FAIL'}} - {{message[:200]}}")
        except Exception as e:
            result["correctness_message"] = f"精度评估异常: {{e}}"
            tag_str = result["tag"]
            logging.error(f"[{{tag_str}}] 精度评估异常: {{e}}")

        # 2. 性能评估（仅在精度通过时执行）
        if result["precision_passed"]:
            try:
                profile_root = Path({repr(profile_dir)})
                ref_time, ref_data, ref_dir, ref_cv, custom_time, custom_data, custom_dir, custom_cv = \\
                    backend.compare_performance_advanced(
                        profile_root=profile_root,
                        num_trials={num_trials},
                        task_type={repr(task_type)},
                    )
                result["time_us"] = custom_time
                result["ref_time_us"] = ref_time
                result["cv_pct"] = custom_cv if custom_cv is not None else 0.0

                # 提取 pipeline 信息
                if custom_data:
                    pipeline = {{}}
                    for row in custom_data:
                        if isinstance(row, dict):
                            for key in ("mte2_ratio", "vec_ratio", "scalar_ratio", "mte3_ratio",
                                        "mte2_pct", "vec_pct", "scalar_pct", "mte3_pct"):
                                if key in row:
                                    pipeline[key] = row[key]
                    result["pipeline"] = pipeline

                # 推断瓶颈类型
                if pipeline:
                    mte2 = pipeline.get("mte2_pct", pipeline.get("mte2_ratio", 0))
                    vec = pipeline.get("vec_pct", pipeline.get("vec_ratio", 0))
                    scalar = pipeline.get("scalar_pct", pipeline.get("scalar_ratio", 0))
                    if mte2 > 50:
                        result["bottleneck"] = "memory_bound"
                    elif vec > 60:
                        result["bottleneck"] = "compute_bound"
                    elif scalar > 30:
                        result["bottleneck"] = "scalar_bound"
                    else:
                        result["bottleneck"] = "balanced"

                tag_str = result["tag"]
                logging.info(f"[{{tag_str}}] 性能: custom={{custom_time:.2f}}us, ref={{ref_time:.2f}}us")
            except Exception as e:
                result["correctness_message"] += f"; 性能评估异常: {{e}}"
                tag_str = result["tag"]
                logging.error(f"[{{tag_str}}] 性能评估异常: {{e}}")

        # 输出 JSON 结果
        print("--- EVAL_RESULT_JSON ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("--- END_EVAL_RESULT_JSON ---")
    """)

    # 写入临时脚本
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"eval_{tag}_",
        delete=False, dir=profile_dir,
    ) as f:
        f.write(eval_script)
        script_path = f.name

    try:
        logging.info(f"[{tag}] 启动评估子进程: {script_path}")

        env = os.environ.copy()
        env["ASCEND_CUSTOM_OPP_PATH"] = opp_path
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = lib_path + ":" + existing_ld
        env["ASCEND_DEVICE_ID"] = str(device_id)

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes
            env=env,
        )

        # 解析输出
        stdout = result.stdout
        if "--- EVAL_RESULT_JSON ---" in stdout:
            json_start = stdout.index("--- EVAL_RESULT_JSON ---") + len("--- EVAL_RESULT_JSON ---")
            json_end = stdout.index("--- END_EVAL_RESULT_JSON ---")
            json_str = stdout[json_start:json_end].strip()
            return json.loads(json_str)
        else:
            logging.error(
                f"[{tag}] 子进程未输出结果JSON\n"
                f"stdout (last 2000):\n{stdout[-2000:]}\n"
                f"stderr (last 2000):\n{result.stderr[-2000:]}"
            )
            return {
                "tag": tag,
                "error": f"子进程未输出结果: returncode={result.returncode}",
                "precision_passed": False,
                "time_us": -1,
                "stderr_tail": result.stderr[-500:],
            }

    except subprocess.TimeoutExpired:
        logging.error(f"[{tag}] 评估子进程超时（600秒）")
        return {
            "tag": tag,
            "error": "评估超时",
            "precision_passed": False,
            "time_us": -1,
        }
    except Exception as e:
        logging.error(f"[{tag}] 评估异常: {e}")
        return {
            "tag": tag,
            "error": str(e),
            "precision_passed": False,
            "time_us": -1,
        }
    finally:
        # 清理临时脚本（保留用于调试）
        pass


def compare_versions(baseline_result: dict, evolved_result: dict) -> dict:
    """
    计算 baseline 和 evolved 的对比指标。

    Args:
        baseline_result: baseline 评估结果
        evolved_result: evolved 评估结果

    Returns:
        dict: 对比指标
    """
    baseline_time = baseline_result.get("time_us", -1)
    evolved_time = evolved_result.get("time_us", -1)

    comparison = {
        "compilation_success": True,  # 如果到达这里，说明编译成功
        "precision_passed": (
            baseline_result.get("precision_passed", False) and
            evolved_result.get("precision_passed", False)
        ),
    }

    if baseline_time > 0 and evolved_time > 0:
        comparison["speedup"] = baseline_time / evolved_time
        comparison["time_delta_us"] = evolved_time - baseline_time
    else:
        comparison["speedup"] = 0.0
        comparison["time_delta_us"] = 0.0

    # 瓶颈变化
    baseline_bn = baseline_result.get("bottleneck", "unknown")
    evolved_bn = evolved_result.get("bottleneck", "unknown")
    comparison["bottleneck_change"] = f"{baseline_bn} -> {evolved_bn}"

    # 评测质量
    evolved_cv = evolved_result.get("cv_pct", 0.0)
    comparison["cv_pct"] = evolved_cv
    if evolved_cv < 5.0:
        comparison["measurement_quality"] = "good"
    elif evolved_cv < 15.0:
        comparison["measurement_quality"] = "acceptable"
    else:
        comparison["measurement_quality"] = "noisy"

    return comparison


def evaluate_ops(
    op_name: str,
    baseline_path: str,
    evolved_path: str,
    reference_py: str,
    custom_py: str,
    device_id: int = 0,
    task_type: str = "vector",
    output_path: str = None,
    num_trials: int = 50,
    soc: str = "",
    repo_type: str = "",
    eval_lock: str = None,
    eval_lock_timeout: float = 300,
    baseline_cache: str = None,
) -> dict:
    """
    对比评估 baseline 和 evolved 两个版本。

    Args:
        op_name: 算子名
        baseline_path: baseline 安装路径
        evolved_path: evolved 安装路径
        reference_py: 参考实现文件路径
        custom_py: 自定义算子文件路径
        device_id: NPU 设备 ID
        task_type: 算子类型
        output_path: 结果输出路径
        num_trials: profiling 试验次数
        soc: 目标芯片
        repo_type: 仓类型
        eval_lock: 评估排队锁文件路径（多变体串行排队使用同一张卡）
        eval_lock_timeout: 评估锁等待超时秒数
        baseline_cache: baseline 评估结果缓存文件路径。若指定且文件存在，
                        跳过 baseline 评估直接复用缓存结果，减少持锁时间。

    Returns:
        dict: 完整对比结果
    """
    # 确定 profiling 目录
    baseline_profile_dir = os.path.join(
        os.path.dirname(evolved_path), "baseline_profiling"
    )
    evolved_profile_dir = os.path.join(
        os.path.dirname(evolved_path), "evolved_profiling"
    )

    # 尝试加载 baseline 缓存（在获取锁之前，减少持锁时间）
    baseline_result = None
    if baseline_cache and os.path.isfile(baseline_cache):
        try:
            with open(baseline_cache, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # 从缓存中提取 baseline 结果
            cached_baseline = cached.get("baseline")
            if cached_baseline and cached_baseline.get("time_us", -1) > 0:
                baseline_result = cached_baseline
                logging.info(
                    f"使用 baseline 缓存: {baseline_cache} "
                    f"(time_us={baseline_result['time_us']:.2f})"
                )
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Baseline 缓存读取失败，将重新评估: {e}")

    # 评估排队锁：多个子 agent 共享同一张卡时串行排队
    lock_fd = None
    if eval_lock:
        logging.info(f"等待评估锁: {eval_lock} (超时 {eval_lock_timeout}s)")
        lock_fd = _acquire_eval_lock(eval_lock, eval_lock_timeout)
        logging.info("评估锁已获取，开始评估")

    try:
        # 仅在无缓存时评估 baseline
        if baseline_result is None:
            logging.info(f"评估 baseline: {baseline_path}")
            baseline_result = evaluate_single_version(
                op_name=op_name,
                install_path=baseline_path,
                reference_py=reference_py,
                custom_py=custom_py,
                device_id=device_id,
                task_type=task_type,
                profile_dir=baseline_profile_dir,
                num_trials=num_trials,
                tag="baseline",
            )

        logging.info(f"评估 evolved: {evolved_path}")
        evolved_result = evaluate_single_version(
            op_name=op_name,
            install_path=evolved_path,
            reference_py=reference_py,
            custom_py=custom_py,
            device_id=device_id,
            task_type=task_type,
            profile_dir=evolved_profile_dir,
            num_trials=num_trials,
            tag="evolved",
        )
    finally:
        if lock_fd is not None:
            _release_eval_lock(lock_fd)
            logging.info("评估锁已释放")

    # 对比
    comparison = compare_versions(baseline_result, evolved_result)

    # 组装最终结果
    final_result = {
        "op_name": op_name,
        "repo_type": repo_type,
        "soc": soc,
        "baseline": baseline_result,
        "evolved": evolved_result,
        "comparison": comparison,
        "eval_backend": "python_npu_event",
    }

    # Baseline sanity check: warn if baseline time seems abnormally high
    # (likely measuring host-side e2e instead of kernel-only)
    baseline_time = baseline_result.get("time_us", -1)
    evolved_time = evolved_result.get("time_us", -1)
    if baseline_time > 0 and evolved_time > 0:
        ratio = baseline_time / evolved_time if evolved_time > 0 else 0
        if baseline_time > 2000 and ratio > 0.8 and ratio < 1.2:
            logging.warning(
                "Baseline time %.1fus is unusually high and close to evolved "
                "time %.1fus. This backend (python_npu_event) measures host-side "
                "end-to-end time including framework overhead. If comparing with "
                "forge (msprof kernel time), results are NOT directly comparable.",
                baseline_time, evolved_time,
            )

    # 输出
    if output_path is None:
        output_path = os.path.join(evolved_path, "evaluation_results.json")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    logging.info(f"评估结果已保存到: {output_path}")

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"评估结果摘要: {op_name}")
    print(f"{'='*60}")
    print(f"  Baseline:")
    print(f"    精度: {'PASS' if baseline_result.get('precision_passed') else 'FAIL'}")
    print(f"    耗时: {baseline_result.get('time_us', -1):.2f} us")
    print(f"    瓶颈: {baseline_result.get('bottleneck', 'unknown')}")
    print(f"  Evolved:")
    print(f"    精度: {'PASS' if evolved_result.get('precision_passed') else 'FAIL'}")
    print(f"    耗时: {evolved_result.get('time_us', -1):.2f} us")
    print(f"    瓶颈: {evolved_result.get('bottleneck', 'unknown')}")
    print(f"  Comparison:")
    print(f"    加速比: {comparison.get('speedup', 0):.3f}x")
    print(f"    耗时差异: {comparison.get('time_delta_us', 0):.2f} us")
    print(f"    瓶颈变化: {comparison.get('bottleneck_change', 'unknown')}")
    print(f"{'='*60}")

    return final_result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ops repository operator: baseline vs evolved"
    )
    parser.add_argument(
        "op_name", type=str,
        help="算子名"
    )
    parser.add_argument(
        "--baseline-path", required=True,
        help="baseline 安装路径"
    )
    parser.add_argument(
        "--evolved-path", required=True,
        help="evolved 安装路径"
    )
    parser.add_argument(
        "--reference-py", required=True,
        help="参考实现 Python 文件路径"
    )
    parser.add_argument(
        "--custom-py", required=True,
        help="自定义算子 Python 文件路径"
    )
    parser.add_argument(
        "--device-id", type=int, default=0,
        help="NPU 设备 ID (default: 0)"
    )
    parser.add_argument(
        "--task-type", type=str, default="vector",
        choices=["vector", "cube", "cv-mix", "unknown"],
        help="算子类型 (default: vector)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="评估结果输出路径 (default: evolved_path/evaluation_results.json)"
    )
    parser.add_argument(
        "--num-trials", type=int, default=50,
        help="profiling 试验次数 (default: 50)"
    )
    parser.add_argument(
        "--soc", type=str, default="",
        help="目标芯片 (如 ascend910b)"
    )
    parser.add_argument(
        "--repo-type", type=str, default="",
        help="仓类型 (nn/cv/math/transformer)"
    )
    parser.add_argument(
        "--eval-lock", type=str, default=None,
        help="评估排队锁文件路径。多个子 agent 共享同一张卡时，通过此锁串行排队评估"
    )
    parser.add_argument(
        "--eval-lock-timeout", type=float, default=300,
        help="评估锁等待超时秒数 (default: 300)"
    )
    parser.add_argument(
        "--baseline-cache", type=str, default=None,
        help="baseline 评估结果缓存文件路径（如 baseline_evaluation.json）。"
             "若指定且文件存在，跳过 baseline 评估直接复用，减少持锁时间"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    evaluate_ops(
        op_name=args.op_name,
        baseline_path=args.baseline_path,
        evolved_path=args.evolved_path,
        reference_py=args.reference_py,
        custom_py=args.custom_py,
        device_id=args.device_id,
        task_type=args.task_type,
        output_path=args.output,
        num_trials=args.num_trials,
        soc=args.soc,
        repo_type=args.repo_type,
        eval_lock=args.eval_lock,
        eval_lock_timeout=args.eval_lock_timeout,
        baseline_cache=args.baseline_cache,
    )


if __name__ == "__main__":
    main()
