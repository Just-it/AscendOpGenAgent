#!/usr/bin/env python3
"""算子验证脚本 — 对比框架实现 (Model) 与生成实现 (ModelNew) 的输出一致性。

多 shape 模式下：每个 shape 独立 try/except，全部跑完后落盘 verify_result.json。
策略 A：passed < total 即整体判失败（exit 1），同时失败清单记录在 JSON 的 `failures` 字段。

用法:
    python verify.py --op_name <算子名> [--verify_dir <验证目录>] [--timeout <超时秒数>]
"""
import argparse
import gc
import json
import os
import sys
import traceback


ERROR_MSG_LIMIT = 2000

REQUIRED_MATCHED_RATIO = 0.9

# allclose 判定阈值 (atol, rtol)：|actual - golden| <= atol + rtol * |golden|
ALLCLOSE_TOLS_STR = {
    "float32":  (1e-3, 2**(-13)), # 2**(-13)=1.220703125e-4
    "float16":  (9e-2, 2**(-10)), # 2**(-10)=9.765625e-4
    "bfloat16": (1e-1, 2**(-7)),  # 2**(-7)=7.8125e-3
}
ALLCLOSE_DEFAULT_TOLS = ALLCLOSE_TOLS_STR["float32"]


class AccuracyError(AssertionError):
    """精度判定失败异常，附带结构化 metrics 便于下游统计。"""

    def __init__(self, message, metrics):
        super().__init__(message)
        self.metrics = metrics


def truncate_error(msg: str, limit: int = ERROR_MSG_LIMIT) -> str:
    if msg is None:
        return ""
    if len(msg) <= limit:
        return msg
    half = limit // 2
    return f"{msg[:half]}\n... [truncated {len(msg) - limit} chars] ...\n{msg[-half:]}"


def describe_input(inputs):
    """输入列表的结构化描述（用于 JSON）。"""
    try:
        import torch
    except Exception:
        torch = None
    descs = []
    for x in inputs:
        if torch is not None and isinstance(x, torch.Tensor):
            descs.append({
                "type": "tensor",
                "shape": list(x.shape),
                "dtype": str(x.dtype),
            })
        else:
            try:
                val = x if isinstance(x, (int, float, bool, str)) else repr(x)
            except Exception:
                val = "<unrepr>"
            descs.append({"type": "scalar", "value": val})
    return descs


def cleanup_npu_memory():
    try:
        import torch
        import torch_npu  # noqa: F401
        torch.npu.empty_cache()
    except Exception:
        pass
    gc.collect()


def get_limits(data_type):
    """根据数据类型返回精度判定的三元组 (small_value_threshold, small_value_error, rel_threshold)。

    参考 NPU Benchmark 精度对比方法：
    - small_value_threshold：判定元素是否落在"小值域"的阈值
    - small_value_error：小值域元素的绝对误差上限
    - rel_threshold：正常值域元素的相对误差上限，同时也是 MERE 的判定阈值

    阈值表：
    | 数据类型      | small_value_threshold | small_value_error | rel_threshold |
    |--------------|-----------------------|-------------------|---------------|
    | FLOAT16      | 2^{-11}               | 2^{-16}           | 2^{-10}       |
    | BFLOAT16     | 2^{-8}                | 2^{-16}           | 2^{-7}        |
    | FLOAT32      | 2^{-14}               | 2^{-30}           | 2^{-13}       |
    | HiFloat32    | 2^{-12}               | 2^{-28}           | 2^{-11}       |
    | FLOAT8 E4M3  | 2^{-4}                | 2^{-6}            | 2^{-3}        |
    | FLOAT8 E5M2  | 2^{-3}                | 2^{-5}            | 2^{-2}        |

    由于 torch.dtype 中没有直接定义 HiFloat32，可通过字符串传入 "hifloat32" 获取对应阈值。
    """  # noqa: E501
    import torch

    # 字符串映射（用于 HiFloat32 或其他自定义类型）
    str_to_limits = {
        "float16":     (2**(-11), 2**(-16), 2**(-10)),
        "bfloat16":    (2**(-8),  2**(-16), 2**(-7)),
        "float32":     (2**(-14), 2**(-30), 2**(-13)),
        "hifloat32":   (2**(-12), 2**(-28), 2**(-11)),
        "float8_e4m3": (2**(-4),  2**(-6),  2**(-3)),
        "float8_e5m2": (2**(-3),  2**(-5),  2**(-2)),
        "fp8_e4m3":    (2**(-4),  2**(-6),  2**(-3)),
        "fp8_e5m2":    (2**(-3),  2**(-5),  2**(-2)),
    }
    if isinstance(data_type, str):
        return str_to_limits.get(data_type.lower(), (2**(-14), 2**(-30), 2**(-13)))

    # torch.dtype 映射
    dtype_limits_map = {
        torch.float16:  (2**(-11), 2**(-16), 2**(-10)),
        torch.bfloat16: (2**(-8),  2**(-16), 2**(-7)),
        torch.float32:  (2**(-14), 2**(-30), 2**(-13)),
    }

    float8_e4m3 = getattr(torch, 'float8_e4m3fn', None) or getattr(torch, 'float8_e4m3', None)
    if float8_e4m3 is not None:
        dtype_limits_map[float8_e4m3] = (2**(-4), 2**(-6), 2**(-3))

    float8_e5m2 = getattr(torch, 'float8_e5m2fn', None) or getattr(torch, 'float8_e5m2', None)
    if float8_e5m2 is not None:
        dtype_limits_map[float8_e5m2] = (2**(-3), 2**(-5), 2**(-2))

    return dtype_limits_map.get(data_type, (2**(-14), 2**(-30), 2**(-13)))


def get_allclose_tols(data_type):
    """根据数据类型返回 allclose 判定的 (atol, rtol)。

    判定公式：|actual - golden| <= atol + rtol * |golden|

    阈值表：
    | 数据类型  | atol  | rtol            |
    |----------|-------|-----------------|
    | FLOAT32  | 2e-5  | 2**(-13)        |
    | FLOAT16  | 5e-3  | 2**(-10)        |
    | BFLOAT16 | 1e-2  | 2**(-7)         |

    未识别 dtype 走 fp32 默认。
    """
    import torch

    if isinstance(data_type, str):
        return ALLCLOSE_TOLS_STR.get(data_type.lower(), ALLCLOSE_DEFAULT_TOLS)

    dtype_map = {
        torch.float16:  ALLCLOSE_TOLS_STR["float16"],
        torch.bfloat16: ALLCLOSE_TOLS_STR["bfloat16"],
        torch.float32:  ALLCLOSE_TOLS_STR["float32"],
    }
    return dtype_map.get(data_type, ALLCLOSE_DEFAULT_TOLS)


def _is_integer_dtype(dtype):
    """判断 torch.dtype 是否为整数类型（不含 bool / 不含浮点 / 不含复数）。"""
    import torch
    if dtype == torch.bool:
        return False
    return (not dtype.is_floating_point) and (not dtype.is_complex)


def _build_dtype_rank():
    """dtype 精度优先级表：值越大精度越高。
    顺序：fp64 > fp32 > fp16 > bf16 > fp8 > int64 > int32 > int16 > int8 > bool
    """
    import torch
    rank = {
        torch.float64: 100,
        torch.float32: 90,
        torch.float16: 80,
        torch.bfloat16: 70,
        torch.int64:   50,
        torch.int32:   40,
        torch.int16:   30,
        torch.int8:    20,
        torch.uint8:   20,
        torch.bool:    10,
    }
    for name in ("float8_e4m3fn", "float8_e4m3", "float8_e5m2fn", "float8_e5m2"):
        dt = getattr(torch, name, None)
        if dt is not None:
            rank[dt] = 60
    return rank


_DTYPE_RANK = None


def _dtype_rank(dtype):
    global _DTYPE_RANK
    if _DTYPE_RANK is None:
        _DTYPE_RANK = _build_dtype_rank()
    return _DTYPE_RANK.get(dtype, 0)


def _is_int_like_dtype(dtype):
    """判断 dtype 属于"整型类"输入（含 bool；不含浮点/复数）。"""
    import torch
    if dtype is None:
        return False
    if dtype == torch.bool:
        return True
    return (not dtype.is_floating_point) and (not dtype.is_complex)


def _infer_input_type(inputs):
    """从 inputs 推断输入类型，返回 ("float" | "int" | "no_tensor", input_dtype | None)。

    判定优先级（KernelBench / NPUKernelBench 统一处理）：
    1. 若存在 torch.Tensor 输入：取所有 tensor 中最高精度 dtype 作为输入类型
    2. 若不存在 tensor，但存在 list/tuple of Tensor（tensor_list）：取第一个 tensor_list 的首元素 dtype
    3. 其他情况（全为标量 attr / 无输入）：返回 ("no_tensor", None)

    bool 输入归到 "int" 类（按规则：bool 输出单独处理；bool 输入与 int 同等对待）。
    """
    import torch
    tensors = [x for x in inputs if isinstance(x, torch.Tensor)]
    source = None
    candidate_dtypes = []
    if tensors:
        candidate_dtypes = [t.dtype for t in tensors]
        top_dtype = max(candidate_dtypes, key=_dtype_rank)
        source = "tensor"
    else:
        tensor_lists = [
            x for x in inputs
            if isinstance(x, (list, tuple)) and len(x) > 0
            and all(isinstance(e, torch.Tensor) for e in x)
        ]
        if tensor_lists:
            top_dtype = tensor_lists[0][0].dtype
            candidate_dtypes = [top_dtype]
            source = "tensor_list"
        else:
            print(
                "  [输入类型判定] 来源=无 tensor 输入（全 attr 或空），"
                "input_type=no_tensor",
                file=sys.stderr,
            )
            return "no_tensor", None

    input_type = "int" if _is_int_like_dtype(top_dtype) else "float"
    print(
        f"  [输入类型判定] 来源={source}，候选 dtypes={[str(dt) for dt in candidate_dtypes]}，"
        f"最高精度={top_dtype}，input_type={input_type}",
        file=sys.stderr,
    )
    return input_type, top_dtype


def resolve_input_provider(torch_module):
    """解析任务文件的输入提供方式。"""
    if hasattr(torch_module, "get_input_groups"):
        groups = torch_module.get_input_groups()
        return groups, len(groups)
    elif hasattr(torch_module, "get_inputs"):
        return [torch_module.get_inputs()], 1
    else:
        raise AttributeError(
            f"模块必须提供 get_inputs() 或 get_input_groups() 方法"
        )


def _compare_binary_exact(fw_out, impl_out, data_type):
    """非计算类：二进制完全一致比对。

    - 浮点 dtype：通过 view-as-int 比较底层 bit pattern，可识别 NaN payload 差异
    - 整型 / bool：直接 torch.equal
    - 复数：实部/虚部分别 view-as-int 比较
    """
    import torch

    fw = fw_out.contiguous().detach().cpu()
    impl = impl_out.contiguous()
    if isinstance(impl, torch.Tensor):
        impl = impl.detach().cpu()
    else:
        raise AssertionError(f"非计算类实现输出必须是 Tensor，实际为 {type(impl).__name__}")

    if fw.shape != impl.shape:
        raise AssertionError(
            f"非计算类验证失败，输出形状不一致: framework={fw.shape}, impl={impl.shape}"
        )
    if fw.dtype != impl.dtype:
        raise AssertionError(
            f"非计算类验证失败，输出 dtype 不一致: framework={fw.dtype}, impl={impl.dtype}"
        )

    def _view_int_dtype(dt):
        if dt in (torch.float64, torch.complex64):
            return torch.int64
        if dt in (torch.float32,):
            return torch.int32
        if dt in (torch.float16, torch.bfloat16):
            return torch.int16
        for name in ("float8_e4m3fn", "float8_e4m3", "float8_e5m2fn", "float8_e5m2"):
            fp8 = getattr(torch, name, None)
            if fp8 is not None and dt == fp8:
                return torch.int8
        return None

    if fw.dtype.is_complex:
        fw_real_bits = torch.view_as_real(fw)
        impl_real_bits = torch.view_as_real(impl)
        view_dt = _view_int_dtype(torch.float32) if fw.dtype == torch.complex64 else torch.int64
        equal = torch.equal(fw_real_bits.view(view_dt), impl_real_bits.view(view_dt))
    elif fw.dtype.is_floating_point:
        view_dt = _view_int_dtype(fw.dtype)
        if view_dt is None:
            raise AssertionError(f"非计算类不支持的浮点 dtype: {fw.dtype}")
        equal = torch.equal(fw.view(view_dt), impl.view(view_dt))
    else:
        equal = torch.equal(fw, impl)

    if equal:
        return

    if fw.dtype.is_floating_point and not fw.dtype.is_complex:
        view_dt = _view_int_dtype(fw.dtype)
        fw_bits = fw.view(view_dt).flatten()
        impl_bits = impl.view(view_dt).flatten()
        diff_mask = fw_bits != impl_bits
        violation_count = int(diff_mask.sum().item())
        violation_idx = torch.where(diff_mask)[0]
        num_to_show = min(10, len(violation_idx))
        detail = f"前 {num_to_show} 个 bit 不一致位置:\n"
        fw_flat = fw.flatten()
        impl_flat = impl.flatten()
        for i in range(num_to_show):
            idx = violation_idx[i].item()
            detail += (
                f"  位置[{idx}]: framework={fw_flat[idx].item()} "
                f"(bits=0x{fw_bits[idx].item() & ((1 << view_dt.itemsize * 8) - 1):x}), "
                f"impl={impl_flat[idx].item()} "
                f"(bits=0x{impl_bits[idx].item() & ((1 << view_dt.itemsize * 8) - 1):x})\n"
            )
    else:
        fw_flat = fw.flatten()
        impl_flat = impl.flatten()
        diff_mask = fw_flat != impl_flat
        violation_count = int(diff_mask.sum().item())
        violation_idx = torch.where(diff_mask)[0]
        num_to_show = min(10, len(violation_idx))
        detail = f"前 {num_to_show} 个不一致位置:\n"
        for i in range(num_to_show):
            idx = violation_idx[i].item()
            detail += (
                f"  位置[{idx}]: framework={fw_flat[idx].item()}, "
                f"impl={impl_flat[idx].item()}\n"
            )

    metrics = {
        "category": "non_compute",
        "dtype": str(data_type),
        "violation_count": violation_count,
        "total_elements": int(fw.numel()),
    }
    raise AccuracyError(
        f"验证失败 dtype={data_type} (非计算类，要求二进制完全一致): "
        f"{violation_count}/{fw.numel()} 元素不一致\n{detail}",
        metrics,
    )


def compare(fw_out, impl_out, data_type, input_type=None, input_dtype=None, non_compute=False):
    """对比框架输出和实现输出。

    Args:
        fw_out: 框架（金标准）输出 Tensor
        impl_out: 被测实现输出 Tensor
        data_type: 输出 dtype（与 fw_out.dtype 一致）
        input_type: 输入类型 "float" / "int" / "no_tensor" / None
            由 _infer_input_type() 推断得出，参与"输出整型时"的分流。
        input_dtype: 输入最高精度 dtype（由 _infer_input_type() 返回），仅用于诊断打印。
        non_compute: 若 True，强制走二进制完全一致路径（搬移 / Cast 等算子）

    决策矩阵（non_compute=False 时）：
        | 输出 dtype | 输入类型           | 类别           | 判定                |
        |-----------|------------------|---------------|--------------------|
        | bool      | 任意              | bool 输出      | torch.equal         |
        | int       | int               | 整数计算类      | |diff| == 0         |
        | int       | float             | 量化类         | |diff| <= 1         |
        | int       | no_tensor         | 整数计算类     | |diff| == 0（最严）   |
        | float     | 任意              | 浮点计算类     | 三项判定（按输出 dtype）|
    """
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

    # 非计算类：二进制完全一致（先于其他判定，跳过 NaN/Inf/finite 过滤）
    if non_compute:
        print(
            f"  [评测模式] 模式=non_compute（非计算类），"
            f"输入 dtype={input_dtype}（{input_type}），输出 dtype={data_type}；"
            f"误差要求=二进制完全一致（view-as-int bit pattern 全等，含 NaN payload）",
            file=sys.stderr,
        )
        _compare_binary_exact(fw_out, impl_out, data_type)
        return

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

    # bool 输出独立处理：严格相等
    if fw_finite.dtype == torch.bool:
        print(
            f"  [评测模式] 模式=bool_output（bool 输出），"
            f"输入 dtype={input_dtype}（{input_type}），输出 dtype={data_type}；"
            f"误差要求=torch.equal 严格相等（finite={finite_count}/{size}）",
            file=sys.stderr,
        )
        if not torch.equal(fw_finite, impl_finite):
            diff_idx = torch.where(fw_finite != impl_finite)[0]
            violation_count = int(diff_idx.numel())
            num_to_show = min(10, violation_count)
            detail = f"前 {num_to_show} 个不一致位置:\n"
            for i in range(num_to_show):
                idx = diff_idx[i].item()
                detail += (
                    f"  位置[{idx}]: framework={fw_finite[idx].item()}, "
                    f"impl={impl_finite[idx].item()}\n"
                )
            metrics = {
                "category": "bool_output",
                "dtype": str(data_type),
                "violation_count": violation_count,
                "total_finite": int(fw_finite.numel()),
            }
            raise AccuracyError(
                f"验证失败 dtype={data_type} (bool 输出，要求严格相等): "
                f"{violation_count}/{fw_finite.numel()} 元素不一致\n{detail}",
                metrics,
            )
        return

    # 输出整型：按 input_type 分流
    if _is_integer_dtype(fw_finite.dtype):
        # input_type == "float" → 量化类 (|diff|<=1)
        # input_type == "int" 或 "no_tensor" 或 None → 整数计算类 (|diff|==0，最严)
        if input_type == "float":
            print(
                f"  [评测模式] 模式=quant_fp_to_int（量化类 fp→int），"
                f"输入 dtype={input_dtype}（{input_type}），输出 dtype={data_type}；"
                f"误差要求=|actual - golden| <= 1（finite={finite_count}/{size}）",
                file=sys.stderr,
            )
            diff = (fw_finite.to(torch.int64) - impl_finite.to(torch.int64)).abs()
            violation_count = int((diff > 1).sum().item())
            if violation_count > 0:
                max_diff = int(diff.max().item())
                violation_idx = torch.where(diff > 1)[0]
                num_to_show = min(10, len(violation_idx))
                detail = f"前 {num_to_show} 个量化误差超限位置:\n"
                for i in range(num_to_show):
                    idx = violation_idx[i].item()
                    detail += (
                        f"  位置[{idx}]: framework={fw_finite[idx].item()}, "
                        f"impl={impl_finite[idx].item()}, "
                        f"|diff|={diff[idx].item()} (允许<=1)\n"
                    )
                metrics = {
                    "category": "quant_fp_to_int",
                    "dtype": str(data_type),
                    "input_type": input_type,
                    "max_abs_diff": max_diff,
                    "violation_count": violation_count,
                    "total_finite": int(diff.numel()),
                    "tolerance": 1,
                }
                raise AccuracyError(
                    f"验证失败 dtype={data_type} (量化类 fp->int，要求|diff|<=1): "
                    f"{violation_count}/{diff.numel()} 元素超限，max_abs_diff={max_diff}\n"
                    f"{detail}",
                    metrics,
                )
            return
        else:
            # 整数计算类：严格相等
            print(
                f"  [评测模式] 模式=integer_compute（整数计算类），"
                f"输入 dtype={input_dtype}（{input_type}），输出 dtype={data_type}；"
                f"误差要求=|actual - golden| == 0（严格相等，finite={finite_count}/{size}）",
                file=sys.stderr,
            )
            if not torch.equal(fw_finite, impl_finite):
                diff = (fw_finite.to(torch.int64) - impl_finite.to(torch.int64)).abs()
                violation_count = int((diff > 0).sum().item())
                max_diff = int(diff.max().item())
                violation_idx = torch.where(diff > 0)[0]
                num_to_show = min(10, len(violation_idx))
                detail = f"前 {num_to_show} 个不一致位置:\n"
                for i in range(num_to_show):
                    idx = violation_idx[i].item()
                    detail += (
                        f"  位置[{idx}]: framework={fw_finite[idx].item()}, "
                        f"impl={impl_finite[idx].item()}, "
                        f"|diff|={diff[idx].item()}\n"
                    )
                metrics = {
                    "category": "integer_compute",
                    "dtype": str(data_type),
                    "input_type": input_type,
                    "max_abs_diff": max_diff,
                    "violation_count": violation_count,
                    "total_finite": int(diff.numel()),
                    "tolerance": 0,
                }
                raise AccuracyError(
                    f"验证失败 dtype={data_type} (整数计算类，要求严格相等): "
                    f"{violation_count}/{diff.numel()} 元素不一致，max_abs_diff={max_diff}\n"
                    f"{detail}",
                    metrics,
                )
            return

    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.to(fw_finite.dtype)

    # 输出浮点：按浮点精度标准执行（dtype-aware 三项判定）
    sv_thr_pre, sv_err_pre, rel_thr_pre = get_limits(data_type)
    atol_pre, rtol_pre = get_allclose_tols(data_type)
    print(
        f"  [评测模式] 模式=float_compute（浮点计算类），"
        f"输入 dtype={input_dtype}（{input_type}），输出 dtype={data_type}；"
        f"误差要求=三项 AND："
        f"(1)max_error_cap |diff|<=atol+rtol*|golden| "
        f"[atol={atol_pre:.3e}, rtol={rtol_pre:.3e}]，"
        f"(2)matched_ratio>={REQUIRED_MATCHED_RATIO} "
        f"[小值域 sv_thr={sv_thr_pre:.3e}/sv_err={sv_err_pre:.3e}，"
        f"正常域 rel_thr={rel_thr_pre:.3e}]，"
        f"(3)MERE<{rel_thr_pre:.3e}（finite={finite_count}/{size}）",
        file=sys.stderr,
    )
    _check_accuracy_npu_benchmark(fw_finite, impl_finite, data_type)


def _check_accuracy_npu_benchmark(golden, actual, data_type):
    """执行 NPU Benchmark 精度验证（三项判定）。

    元素级 matched 定义（用于 #2 matched_ratio）：
    - |golden| < small_value_threshold（小值域）：|diff| <= small_value_error
    - 否则（正常值域）：|diff| / (|golden| + 1e-7) <= rel_threshold

    通过条件（三项 AND）：
    1. allclose: 所有元素满足 |diff| <= atol + rtol * |golden|（dtype-aware）
    2. matched_ratio = sum(matched) / total_finite >= REQUIRED_MATCHED_RATIO（0.9）
    3. MERE < rel_threshold（对所有 finite 元素计算相对误差再取均值，
       分母统一用 |golden| + 1e-7 防除零）

    Args:
        golden: 参考输出（金标准）
        actual: 被测实现输出
        data_type: 数据类型，用于获取对应阈值

    Raises:
        AccuracyError: 当精度验证未通过时，异常的 metrics 属性携带结构化指标
    """
    import torch

    # 统一升 float32，避免低精度 dtype 自身误差污染计算
    golden_f = golden.float()
    actual_f = actual.float()

    sv_thr, sv_err, rel_thr = get_limits(data_type)
    atol, rtol = get_allclose_tols(data_type)

    abs_diff = (actual_f - golden_f).abs()
    abs_golden = golden_f.abs()

    # 分桶（用于 #2 matched_ratio）
    small_mask = abs_golden < sv_thr
    normal_mask = ~small_mask

    # 元素级 matched（#2 口径）
    small_ok = abs_diff <= sv_err
    rel_err = abs_diff / (abs_golden + 1e-7)
    normal_ok = rel_err <= rel_thr
    matched_mask = torch.where(small_mask, small_ok, normal_ok)

    total_finite = matched_mask.numel()
    matched_count = int(matched_mask.sum().item())
    matched_ratio = matched_count / total_finite if total_finite > 0 else 1.0
    max_abs_diff = abs_diff.max().item() if total_finite > 0 else 0.0

    # #1 allclose：逐元素判定，要求 100% 通过
    allclose_bound = atol + rtol * abs_golden
    allclose_mask = abs_diff <= allclose_bound
    allclose_violation_count = int((~allclose_mask).sum().item()) if total_finite > 0 else 0
    allclose_ok = allclose_violation_count == 0

    # MERE：对所有 finite 元素计算相对误差再取均值（分母统一 |golden| + 1e-7 防除零）
    normal_count = int(normal_mask.sum().item())
    if total_finite > 0:
        MERE = rel_err.mean().item()
        mere_ok = MERE < rel_thr
    else:
        MERE = None
        mere_ok = True

    ratio_ok = matched_ratio >= REQUIRED_MATCHED_RATIO
    is_pass = allclose_ok and ratio_ok and mere_ok

    if is_pass:
        return

    metrics = {
        "matched_ratio": matched_ratio,
        "max_abs_diff": max_abs_diff,
        "MERE": MERE,
        "rel_threshold": rel_thr,
        "small_value_threshold": sv_thr,
        "small_value_error": sv_err,
        "atol": atol,
        "rtol": rtol,
        "max_error_cap_violation_count": allclose_violation_count,
        "required_matched_ratio": REQUIRED_MATCHED_RATIO,
        "total_finite": total_finite,
        "matched_count": matched_count,
        "small_count": int(small_mask.sum().item()),
        "normal_count": normal_count,
        "checks": {
            "max_error_cap": allclose_ok,
            "required_matched_ratio": ratio_ok,
            "MERE": mere_ok,
        },
    }

    mere_str = f"{MERE:.6e}" if MERE is not None else "n/a"
    error_msg = (
        f"验证失败 dtype={data_type}: "
        f"max_error_cap_violations={allclose_violation_count}/{total_finite} "
        f"(atol={atol:.6e}, rtol={rtol:.6e}, max_abs_diff={max_abs_diff:.6e}, ok={allclose_ok}), "
        f"matched_ratio={matched_ratio:.6f} (req>={REQUIRED_MATCHED_RATIO}, ok={ratio_ok}), "
        f"MERE={mere_str} (rel_thr={rel_thr:.6e}, ok={mere_ok}); "
        f"small_count={metrics['small_count']}, normal_count={normal_count}\n"
    )

    # 仅在对应检查失败时打印各自的违例位置（前 N 个）
    if not allclose_ok:
        allclose_violation_indices = torch.where(~allclose_mask)[0]
        num_to_show = min(10, len(allclose_violation_indices))
        error_msg += f"前 {num_to_show} 个 max_error_cap 违例位置:\n"
        for i in range(num_to_show):
            idx = allclose_violation_indices[i].item()
            error_msg += (
                f"  位置[{idx}]: framework={golden[idx]:.6e}, "
                f"impl={actual[idx]:.6e}, |diff|={abs_diff[idx]:.6e} "
                f"(允许<=atol+rtol*|golden|={allclose_bound[idx]:.6e})\n"
            )

    if not ratio_ok:
        unmatched_mask = ~matched_mask
        unmatched_indices = torch.where(unmatched_mask)[0]
        num_to_show = min(10, len(unmatched_indices))
        error_msg += f"前 {num_to_show} 个 matched 未通过位置:\n"
        for i in range(num_to_show):
            idx = unmatched_indices[i].item()
            if small_mask[idx].item():
                error_msg += (
                    f"  位置[{idx}] (小值域): framework={golden[idx]:.6e}, "
                    f"impl={actual[idx]:.6e}, |diff|={abs_diff[idx]:.6e} "
                    f"(允许<={sv_err:.6e})\n"
                )
            else:
                error_msg += (
                    f"  位置[{idx}] (正常域): framework={golden[idx]:.6e}, "
                    f"impl={actual[idx]:.6e}, 相对误差={rel_err[idx]:.6e} "
                    f"(允许<={rel_thr:.6e})\n"
                )
    raise AccuracyError(error_msg, metrics)


def run_single_case(
    framework_model,
    impl_model,
    inputs,
    device,
    case_idx,
    total_cases,
    non_compute=False,
):
    """验证单组输入。失败时抛出 AssertionError。"""
    import torch

    print(f"  测试第 {case_idx}/{total_cases} 组输入...", file=sys.stderr)

    # 推断输入类型（"float" / "int" / "no_tensor"）→ 决定输出整型时走整数计算 vs 量化
    input_type, input_dtype = _infer_input_type(inputs)

    inputs_for_impl = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]
    inputs_for_framework = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]

    with torch.no_grad():
        impl_output = impl_model(*inputs_for_impl)
        framework_output = framework_model(*inputs_for_framework)

    if not isinstance(framework_output, (list, tuple)):
        framework_output = [framework_output]
    if not isinstance(impl_output, (list, tuple)):
        impl_output = [impl_output]

    if len(framework_output) != len(impl_output):
        raise AssertionError(
            f"[用例 {case_idx}/{total_cases}] 输出数量不一致: "
            f"framework={len(framework_output)}, impl={len(impl_output)}"
        )

    print(
        f"  [输出概览] 共 {len(framework_output)} 个输出，non_compute={non_compute}",
        file=sys.stderr,
    )

    for i, (fw_out, impl_out) in enumerate(zip(framework_output, impl_output)):
        if fw_out is None or impl_out is None:
            raise AssertionError(
                f"[用例 {case_idx}/{total_cases}] 输出 {i} 为 None: "
                f"framework={fw_out is None}, impl={impl_out is None}"
            )
        if isinstance(fw_out, torch.Tensor) and isinstance(impl_out, torch.Tensor):
            try:
                data_type = fw_out.dtype
                print(
                    f"  [输出 {i}] shape={list(fw_out.shape)}, dtype={data_type}",
                    file=sys.stderr,
                )
                compare(
                    fw_out, impl_out, data_type,
                    input_type=input_type, input_dtype=input_dtype,
                    non_compute=non_compute,
                )
            except AccuracyError as e:
                raise AccuracyError(
                    f"[用例 {case_idx}/{total_cases}] {str(e)}", e.metrics
                ) from e
            except AssertionError as e:
                raise AssertionError(f"[用例 {case_idx}/{total_cases}] {str(e)}") from e


def verify_implementations(op_name, verify_dir, triton_impl_name="triton_ascend_impl", output_path=None, non_compute=False):
    """验证框架实现和生成实现的结果一致性。

    每个 shape 独立 try/except，全部跑完后写 verify_result.json。

    Args:
        non_compute: 若 True，所有 case 走"非计算类"二进制完全一致判定（搬移/Cast 等算子）

    Returns:
        (passed_cases, total_cases)
    """
    import torch
    import torch_npu  # noqa: F401

    sys.path.insert(0, verify_dir)

    torch_module = __import__(f"{op_name}_torch")
    impl_module = __import__(f"{op_name}_{triton_impl_name}")

    FrameworkModel = torch_module.Model
    ModelNew = impl_module.ModelNew
    get_init_inputs = torch_module.get_init_inputs

    # 在获取输入之前设置种子，确保随机生成的输入可复现
    torch.manual_seed(0)
    torch.npu.manual_seed(0)

    input_groups, total_cases = resolve_input_provider(torch_module)

    device = torch.device("npu")

    failures = []
    passed_cases = 0

    for case_idx, inputs in enumerate(input_groups, start=1):
        input_desc = describe_input(inputs)
        framework_model = None
        impl_model = None
        try:
            init_params = get_init_inputs()
            torch.manual_seed(0)
            torch.npu.manual_seed(0)
            framework_model = FrameworkModel(*init_params).to(device)

            torch.manual_seed(0)
            torch.npu.manual_seed(0)
            impl_model = ModelNew(*init_params).to(device)

            run_single_case(
                framework_model, impl_model, inputs, device, case_idx, total_cases,
                non_compute=non_compute,
            )
            passed_cases += 1
        except Exception as e:
            err_detail = traceback.format_exc()
            print(f"  [用例 {case_idx}/{total_cases}] 失败: {type(e).__name__}: {e}", file=sys.stderr)
            failure_entry = {
                "case_idx": case_idx,
                "input_desc": input_desc,
                "error_type": type(e).__name__,
                "error_msg": truncate_error(err_detail),
            }
            if isinstance(e, AccuracyError):
                failure_entry["metrics"] = e.metrics
            failures.append(failure_entry)
        finally:
            del framework_model
            del impl_model
            cleanup_npu_memory()

    failed_cases = total_cases - passed_cases

    # 落盘 verify_result.json
    if output_path is None:
        output_path = os.path.join(verify_dir, "verify_result.json")
    result = {
        "op_name": op_name,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "failures": failures,
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"验证结果已保存到: {output_path}", file=sys.stderr)
    except Exception as e:
        print(f"警告: 无法写入 verify_result.json: {e}", file=sys.stderr)

    if failed_cases == 0:
        print(f"验证成功：共 {total_cases} 组测试用例全部通过")
    else:
        print(
            f"验证失败：{passed_cases}/{total_cases} 组通过，"
            f"{failed_cases} 组失败（详见 {output_path}）",
            file=sys.stderr,
        )

    return passed_cases, total_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="算子验证脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument(
        "--verify_dir", default=".",
        help="验证目录，包含 {op_name}_torch.py 和 {op_name}_triton_ascend_impl.py（默认当前目录）",
    )
    parser.add_argument("--timeout", type=int, default=900, help="超时秒数（已忽略：当前为同进程串行模式）")
    parser.add_argument(
        "--triton_impl_name", default="triton_ascend_impl",
        help="Triton 实现模块名（不含 op_name 前缀，默认 triton_ascend_impl）",
    )
    parser.add_argument(
        "--output", default=None,
        help="验证结果 JSON 输出路径（默认 {verify_dir}/verify_result.json）",
    )
    parser.add_argument(
        "--non-compute", action="store_true",
        help="非计算类算子（搬移 / Cast 等），所有 case 走二进制完全一致判定",
    )
    args = parser.parse_args()

    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        print(f"错误: 验证目录不存在: {verify_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        passed, total = verify_implementations(
            args.op_name, verify_dir, args.triton_impl_name, args.output,
            non_compute=args.non_compute,
        )
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    # 策略 A：passed < total → exit 1
    sys.exit(0 if passed == total and total > 0 else 1)