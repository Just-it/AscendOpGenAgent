#!/usr/bin/env python3
"""算子验证脚本 — 对比框架实现 (Model) 与生成实现 (ModelNew) 的输出一致性。

用法:
    python verify.py --op_name <算子名> [--verify_dir <验证目录>] [--timeout <超时秒数>]

前置条件（验证目录下需存在以下文件）:
    {op_name}_torch.py            — 包含 Model, get_inputs, get_init_inputs
    {op_name}_triton_ascend_impl.py — 包含 ModelNew
"""
import argparse
import os
import sys
import subprocess


def get_limit(data_type):
    """根据数据类型获取精度阈值"""
    import torch
    if data_type == torch.float16:
        return 0.004
    elif data_type == torch.bfloat16:
        return 0.03
    elif data_type == torch.int8:
        return 0.01
    elif data_type in (torch.int16, torch.int32, torch.int64, torch.uint8):
        # 整型要求精确相等
        return 0.0
    else:
        # float32/float64/complex64/complex128 等
        return 0.02


def resolve_input_provider(torch_module):
    """解析任务文件的输入提供方式。

    支持两种格式：
        - get_inputs(): 旧格式，返回单组输入
        - get_input_groups(): 新格式，返回多组输入列表

    Returns:
        (input_groups, total_cases)
        - input_groups: 输入组列表
        - total_cases: 测试用例总数
    """
    if hasattr(torch_module, "get_input_groups"):
        # 新格式：多组输入（如 26_GELU_.py 有 51 组 shape 用例）
        groups = torch_module.get_input_groups()
        return groups, len(groups)
    elif hasattr(torch_module, "get_inputs"):
        # 旧格式：单组输入（保持向后兼容）
        return [torch_module.get_inputs()], 1
    else:
        raise AttributeError(
            f"模块必须提供 get_inputs() 或 get_input_groups() 方法"
        )


def compare(fw_out, impl_out, limit, data_type):
    """对比框架输出和实现输出"""
    import torch
    fw_flat = fw_out.flatten().detach().cpu()
    impl_flat = impl_out.flatten()
    if isinstance(impl_flat, torch.Tensor):
        impl_flat = impl_flat.detach().cpu()
    else:
        impl_flat = torch.tensor(impl_flat, dtype=fw_flat.dtype)

    size = fw_flat.numel()

    if fw_flat.shape != impl_flat.shape:
        raise AssertionError(
            f"验证失败，输出形状不一致: framework={fw_flat.shape}, impl={impl_flat.shape}"
        )

    fw_nan_mask = torch.isnan(fw_flat)
    impl_nan_mask = torch.isnan(impl_flat)
    if not torch.equal(fw_nan_mask, impl_nan_mask):
        fw_nan_count = fw_nan_mask.sum().item()
        impl_nan_count = impl_nan_mask.sum().item()
        raise AssertionError(
            f"验证失败，NaN 位置不匹配: Framework={fw_nan_count}/{size}, "
            f"Implementation={impl_nan_count}/{size}"
        )

    fw_inf_mask = torch.isinf(fw_flat)
    impl_inf_mask = torch.isinf(impl_flat)
    if not torch.equal(fw_inf_mask, impl_inf_mask):
        fw_inf_count = fw_inf_mask.sum().item()
        impl_inf_count = impl_inf_mask.sum().item()
        raise AssertionError(
            f"验证失败，Inf 位置不匹配: Framework={fw_inf_count}/{size}, "
            f"Implementation={impl_inf_count}/{size}"
        )
    if fw_inf_mask.any():
        if not torch.equal(
            torch.sign(fw_flat[fw_inf_mask]),
            torch.sign(impl_flat[impl_inf_mask]),
        ):
            raise AssertionError("验证失败，Inf 符号不匹配")

    finite_mask = torch.isfinite(fw_flat) & torch.isfinite(impl_flat)
    finite_count = finite_mask.sum().item()
    if finite_count == 0:
        print("警告: 所有值都是非有限值，跳过精度检查")
        return

    fw_finite = fw_flat[finite_mask]
    impl_finite = impl_flat[finite_mask]

    if fw_finite.dtype == torch.bool:
        if not torch.equal(fw_finite, impl_finite):
            raise AssertionError(f"验证失败，布尔值不匹配: dtype={data_type}")
        return

    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.to(fw_finite.dtype)

    # complex 类型：分别比较 real/imag 部分
    if fw_finite.is_complex():
        fw_real = fw_finite.real.float()
        fw_imag = fw_finite.imag.float()
        impl_real = impl_finite.real.float()
        impl_imag = impl_finite.imag.float()
        abs_diff = torch.sqrt(
            (fw_real - impl_real) ** 2 + (fw_imag - impl_imag) ** 2
        )
        abs_ref = torch.sqrt(fw_real ** 2 + fw_imag ** 2)
    else:
        abs_diff = torch.abs(fw_finite.float() - impl_finite.float())
        abs_ref = torch.abs(fw_finite.float())
    eps = 1e-8
    relative_error = torch.where(abs_ref > eps, abs_diff / abs_ref, abs_diff)

    err_cnt = (relative_error > limit).sum().item()
    limit_cnt = int(finite_count * limit)

    if err_cnt > limit_cnt:
        max_error = relative_error.max().item()
        mean_error = relative_error.mean().item()
        mismatch_mask = relative_error > limit
        mismatch_indices = torch.where(mismatch_mask)[0]
        num_to_show = min(10, len(mismatch_indices))

        error_msg = (
            f"验证失败，输出不一致(误差数/最大容忍误差数): "
            f"err_cnt={err_cnt} / {limit_cnt}, dtype={data_type}, limit={limit}\n"
        )
        error_msg += f"最大相对误差: {max_error:.6e}, 平均相对误差: {mean_error:.6e}\n"
        error_msg += f"前 {num_to_show} 个不一致的值:\n"
        for i in range(num_to_show):
            idx = mismatch_indices[i].item()
            error_msg += (
                f"  位置[{idx}]: framework={fw_finite[idx]:.6e}, "
                f"impl={impl_finite[idx]:.6e}, "
                f"相对误差={relative_error[idx]:.6e}\n"
            )
        raise AssertionError(error_msg)


def run_single_case(
    framework_model,
    impl_model,
    inputs,
    device,
    case_idx,
    total_cases
):
    """验证单组输入。

    注意: 此函数依赖 torch，应在已导入 torch 的上下文中调用，
    或由 verify_implementations() 调用（该函数会导入 torch）。

    Args:
        framework_model: 参考实现模型
        impl_model: 生成的实现模型
        inputs: 输入张量/参数列表
        device: NPU 设备
        case_idx: 当前用例序号（从1开始）
        total_cases: 总用例数
    """
    import torch  # 延迟导入：确保 torch 在此作用域可用

    print(f"  测试第 {case_idx}/{total_cases} 组输入...", file=sys.stderr)

    # 将输入移至设备
    inputs_for_impl = [
        x.to(device) if isinstance(x, torch.Tensor) else x 
        for x in inputs
    ]
    inputs_for_framework = [
        x.to(device) if isinstance(x, torch.Tensor) else x 
        for x in inputs
    ]

    # 前向推理
    with torch.no_grad():
        impl_output = impl_model(*inputs_for_impl)
        framework_output = framework_model(*inputs_for_framework)

    # 标准化输出格式
    if not isinstance(framework_output, (list, tuple)):
        framework_output = [framework_output]
    if not isinstance(impl_output, (list, tuple)):
        impl_output = [impl_output]

    # 验证输出数量
    if len(framework_output) != len(impl_output):
        raise AssertionError(
            f"[用例 {case_idx}/{total_cases}] 输出数量不一致: "
            f"framework={len(framework_output)}, impl={len(impl_output)}"
        )

    # 比较每个输出
    for i, (fw_out, impl_out) in enumerate(zip(framework_output, impl_output)):
        if fw_out is None or impl_out is None:
            raise AssertionError(
                f"[用例 {case_idx}/{total_cases}] 输出 {i} 为 None: "
                f"framework={fw_out is None}, impl={impl_out is None}"
            )
        if isinstance(fw_out, torch.Tensor) and isinstance(impl_out, torch.Tensor):
            try:
                data_type = fw_out.dtype
                limit = get_limit(data_type)
                compare(fw_out, impl_out, limit, data_type)
            except AssertionError as e:
                raise AssertionError(f"[用例 {case_idx}/{total_cases}] {str(e)}") from e


def _collect_case_meta(inputs):
    """收集当前 case 的元信息（shape + dtype），供失败时汇报使用。"""
    import torch
    shapes = []
    dtypes = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            shapes.append(list(x.shape))
            dtypes.append(str(x.dtype).replace("torch.", ""))
    return shapes, dtypes


def verify_implementations(op_name, verify_dir, triton_impl_name="triton_ascend_impl",
                           continue_on_error=False, summary_output=None):
    """验证框架实现和生成实现的结果一致性，支持多组输入验证。

    Args:
        continue_on_error: True 时，单个 case 失败不中断循环，继续跑其余 case。
        summary_output: 若指定路径，则将通过/失败明细写入该 JSON 文件。
    Returns:
        int: 0 全部通过；1 全部失败；2 部分通过。
    """
    import json as _json
    import torch
    import torch_npu  # noqa: F401

    sys.path.insert(0, verify_dir)

    # 加载模块
    torch_module = __import__(f"{op_name}_torch")
    impl_module = __import__(f"{op_name}_{triton_impl_name}")

    FrameworkModel = torch_module.Model
    ModelNew = impl_module.ModelNew
    get_init_inputs = torch_module.get_init_inputs

    # 解析输入（支持 get_inputs 或 get_input_groups 格式）
    input_groups, total_cases = resolve_input_provider(torch_module)

    device = torch.device("npu")
    init_params = get_init_inputs()

    passed_cases = []
    failed_cases = []

    for case_idx, inputs in enumerate(input_groups, start=1):
        shapes, dtypes = _collect_case_meta(inputs)
        try:
            # 创建模型（确保权重一致）
            torch.manual_seed(0)
            torch.npu.manual_seed(0)
            framework_model = FrameworkModel(*init_params).to(device)

            torch.manual_seed(0)
            torch.npu.manual_seed(0)
            impl_model = ModelNew(*init_params).to(device)

            # 验证该组输入
            run_single_case(
                framework_model,
                impl_model,
                inputs,
                device,
                case_idx,
                total_cases,
            )
            passed_cases.append(case_idx)
        except Exception as e:
            err_msg = str(e)
            if len(err_msg) > 500:
                err_msg = err_msg[:500] + "...(truncated)"
            failed_cases.append({
                "case_idx": case_idx,
                "shapes": shapes,
                "dtypes": dtypes,
                "error": err_msg,
            })
            print(
                f"  [用例 {case_idx}/{total_cases}] 失败: "
                f"dtypes={dtypes}, shapes={shapes}, error={err_msg[:200]}",
                file=sys.stderr,
            )
            if not continue_on_error:
                if summary_output:
                    _write_summary(summary_output, total_cases, passed_cases, failed_cases)
                raise

    passed = len(passed_cases)
    failed = len(failed_cases)

    if summary_output:
        _write_summary(summary_output, total_cases, passed_cases, failed_cases)

    if failed == 0:
        print(f"验证成功：共 {total_cases} 组测试用例通过")
        return 0
    elif passed == 0:
        print(f"验证失败：共 {total_cases} 组测试用例全部失败", file=sys.stderr)
        return 1
    else:
        print(
            f"验证部分通过：{passed}/{total_cases} 成功，{failed} 失败",
            file=sys.stderr,
        )
        return 2


def _write_summary(path, total_cases, passed_cases, failed_cases):
    """写入 verify 摘要 JSON（供上层脚本读取通过率）。"""
    import json as _json
    pass_rate = len(passed_cases) / total_cases if total_cases > 0 else 0.0
    summary = {
        "total_cases": total_cases,
        "passed_cases": len(passed_cases),
        "failed_cases": len(failed_cases),
        "pass_rate": round(pass_rate, 4),
        "passed_case_indices": passed_cases,
        "failed_case_details": failed_cases,
    }
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="算子验证脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument(
        "--verify_dir", default=".",
        help="验证目录，包含 {op_name}_torch.py 和 {op_name}_triton_ascend_impl.py（默认当前目录）",
    )
    parser.add_argument("--timeout", type=int, default=900, help="超时秒数（默认 900）")
    parser.add_argument(
        "--triton_impl_name", default="triton_ascend_impl",
        help="Triton 实现模块名（不含 op_name 前缀，默认 triton_ascend_impl）",
    )
    parser.add_argument(
        "--_run", action="store_true",
        help=argparse.SUPPRESS,  # 内部参数：子进程模式，直接执行验证
    )
    parser.add_argument(
        "--continue_on_error", action="store_true",
        help="单个 case 失败时继续跑其余 case（终评阶段使用）",
    )
    parser.add_argument(
        "--summary_output", default=None,
        help="将 verify 摘要（通过/失败明细）写入指定 JSON 文件",
    )
    args = parser.parse_args()

    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        print(f"错误: 验证目录不存在: {verify_dir}", file=sys.stderr)
        sys.exit(1)

    if args._run:
        # 子进程模式：直接执行验证逻辑
        try:
            rc = verify_implementations(
                args.op_name, verify_dir, args.triton_impl_name,
                continue_on_error=args.continue_on_error,
                summary_output=args.summary_output,
            )
            sys.exit(rc)
        except Exception as e:
            print(f"{e}", file=sys.stderr)
            sys.exit(1)
    else:
        # 主进程模式：启动子进程执行验证，超时后 kill 整个进程树
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--op_name", args.op_name,
            "--verify_dir", verify_dir,
            "--triton_impl_name", args.triton_impl_name,
            "--_run",
        ]
        if args.continue_on_error:
            cmd.append("--continue_on_error")
        if args.summary_output:
            cmd += ["--summary_output", args.summary_output]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = proc.communicate(timeout=args.timeout)

            sys.stdout.buffer.write(stdout)
            sys.stdout.buffer.flush()
            if proc.returncode != 0:
                sys.stderr.buffer.write(stderr)
                sys.stderr.buffer.flush()
            sys.exit(proc.returncode)

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"验证超时（{args.timeout}秒），已终止子进程", file=sys.stderr)
            sys.exit(1)
