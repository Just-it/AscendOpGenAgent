#!/usr/bin/env python3
"""
profiling_runner.py — simulator 调用封装 (msprof op simulator / cannsim)

根据算子的工作目录，自动生成运行脚本，调用 simulator 工具生成
指令级 trace 数据 (trace.json)。

两种 profiling 模式:
  1. msprof op simulator: 在 simulator 上运行，生成指令级 trace.json (用于空泡分析) / cannsim: Ascend950专用仿真工具（CANN 9.0+)，输出trace_coreN.json (用于空泡分析) 
  2. msprof op (on-board): 在真实 NPU 上运行，生成 op_summary CSV (用于性能测量)

本脚本专注于模式 1 (simulator)，为 T1-T6 分析工具提供输入数据。

用法:
    python3 profiling_runner.py --work-dir output/AveragePool2D --op-name AveragePool2D
    python3 profiling_runner.py --work-dir output/AveragePool2D --op-name AveragePool2D --output-dir output/AveragePool2D/simulator_prof
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


def _find_custom_opp_path(work_dir: Path) -> str:
    """检测 vendors 下的自定义算子目录，返回绝对路径或 None。

    支持以下目录命名:
    - vendors/customize/         (标准 CAKE2 DSL Pipeline)
    - vendors/custom_nn/         (ops-nn 仓)
    - vendors/custom_cv/         (ops-cv 仓)
    - vendors/custom_transformer/ (ops-transformer 仓)
    - vendors/omni_custom_*/     (omni-ops 仓)
    """
    vendors_dir = work_dir / "vendors"
    if not vendors_dir.is_dir():
        return None

    # 1. 优先: custom_* 或 omni_custom_* 目录
    for subdir in sorted(vendors_dir.iterdir()):
        if subdir.is_dir() and (subdir.name.startswith("custom")
                                 or subdir.name.startswith("omni_custom")):
            return str(subdir.resolve())

    # 2. 回退: customize (标准 CAKE2)
    customize_path = vendors_dir / "customize"
    if customize_path.is_dir():
        return str(customize_path.resolve())

    return None


def _find_custom_lib_path(work_dir: Path) -> str:
    """检测 vendors 下自定义算子的 op_api/lib/ 目录，返回绝对路径或 None。"""
    opp_path = _find_custom_opp_path(work_dir)
    if opp_path:
        lib_dir = Path(opp_path) / "op_api" / "lib"
        if lib_dir.is_dir():
            return str(lib_dir.resolve())
    return None


def _build_env_block(work_dir: Path) -> str:
    """生成脚本中设置 ASCEND_CUSTOM_OPP_PATH 和 LD_LIBRARY_PATH 的代码块。"""
    custom_opp = _find_custom_opp_path(work_dir)
    if not custom_opp:
        return ""
    lines = [
        f'import os as _os',
        f'_os.environ["ASCEND_CUSTOM_OPP_PATH"] = r"{custom_opp}"',
    ]
    lib_path = _find_custom_lib_path(work_dir)
    if lib_path:
        lines.append(
            f'_ld = _os.environ.get("LD_LIBRARY_PATH", "")\n'
            f'if r"{lib_path}" not in _ld:\n'
            f'    _os.environ["LD_LIBRARY_PATH"] = r"{lib_path}" + ":" + _ld'
        )
    return "\n".join(lines)


def _parse_custom_module(work_dir: Path, op_name: str) -> dict:
    """
    解析 {op_name}_custom.py，检测 forward() 中是否有 permute 调用。

    Returns:
        {
            "has_permute": bool,
            "pre_permute": list|None,   # e.g. [0, 2, 3, 1]
            "post_permute": list|None,  # e.g. [0, 3, 1, 2]
            "custom_call": str|None,    # e.g. "custom_ops_lib.average_pool2d_custom"
            "custom_args_extra": str|None,  # e.g. "self.kernel_size, self.stride, self.padding"
            "module_fn_args": str|None,     # e.g. "kernel_size, stride, padding" (from module_fn signature)
            "has_get_inputs": bool,
            "functional_py": str|None,  # path to _functional.py if get_inputs not in _custom.py
        }
    """
    custom_py = work_dir / f"{op_name}_custom.py"
    if not custom_py.exists():
        return {"has_permute": False, "has_get_inputs": False}

    src = custom_py.read_text(encoding="utf-8")

    result = {
        "has_permute": False,
        "pre_permute": None,
        "post_permute": None,
        "custom_call": None,
        "custom_args_extra": None,
        "module_fn_args": None,
        "has_get_inputs": "def get_inputs" in src,
        "functional_py": None,
    }

    # Check for fallback _functional.py
    if not result["has_get_inputs"]:
        func_py = work_dir / f"{op_name}_functional.py"
        if func_py.exists():
            result["functional_py"] = str(func_py)

    # Detect permute in forward()
    # Pattern: x.permute(0, 2, 3, 1).contiguous() or x.permute([0,2,3,1]).contiguous()
    permute_pat = re.compile(
        r'\.permute\(\s*(\[?[\d,\s]+\]?)\s*\)\.contiguous\(\)'
    )
    permutes = permute_pat.findall(src)
    if len(permutes) >= 1:
        result["has_permute"] = True
        # Parse first permute (pre-call) and second (post-call)
        def _parse_perm(s):
            s = s.strip().strip("[]")
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        result["pre_permute"] = _parse_perm(permutes[0])
        if len(permutes) >= 2:
            result["post_permute"] = _parse_perm(permutes[1])

    # Detect custom_ops_lib.xxx_custom() call
    call_pat = re.compile(r'(custom_ops_lib\.\w+)\(')
    call_match = call_pat.search(src)
    if call_match:
        result["custom_call"] = call_match.group(1)

    # Extract module_fn signature args (after x)
    fn_pat = re.compile(r'def module_fn\(x:\s*torch\.Tensor,?\s*(.*?)\)\s*->')
    fn_match = fn_pat.search(src)
    if fn_match:
        raw_args = fn_match.group(1).strip().rstrip(",")
        # Strip type annotations and defaults: "kernel_size: int, stride: int = 2" -> "kernel_size, stride"
        arg_names = []
        for arg in raw_args.split(","):
            arg = arg.strip()
            if not arg:
                continue
            name = arg.split(":")[0].split("=")[0].strip()
            arg_names.append(name)
        result["module_fn_args"] = ", ".join(arg_names)

    return result


def _parse_test_case_csv(csv_path: str, case_id: str = None) -> dict:
    """
    从 test_cases.csv 中读取指定 case 的输入 shape 和 dtype。

    CSV 格式:
        case_id, var0_shape, var0_dtype, var1_shape, var1_dtype, ...
        case_0, "[1, 128, 768]", float16, "[768]", float16, ...

    如果 case_id 未指定，选择总元素数最大的 case（代表最重的负载，
    profiling 结果最能反映真实场景）。

    Returns:
        {
            "case_id": str,
            "inputs": [
                {"shape": [1, 128, 768], "dtype": "float16"},
                {"shape": [768], "dtype": "float16"},
                ...
            ],
            "init_params": {key: value}  # 非 var*_shape/dtype 的列
        }
        解析失败时返回 None。
    """
    import csv
    import ast

    if not os.path.isfile(csv_path):
        return None

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        for row in reader:
            rows.append(row)

    if not rows:
        return None

    # 选择 case
    target_row = None
    if case_id:
        for row in rows:
            if row.get("case_id", "").strip() == case_id:
                target_row = row
                break
        if target_row is None:
            # case_id not found, fall back to largest
            target_row = None

    if target_row is None:
        # 选择总元素数最大的 case
        def _total_elements(row):
            total = 0
            for k, v in row.items():
                if k.endswith("_shape") and v:
                    try:
                        shape = ast.literal_eval(v.strip())
                        elems = 1
                        for d in shape:
                            elems *= d
                        total += elems
                    except Exception:
                        pass
            return total
        target_row = max(rows, key=_total_elements)

    # 解析 var*_shape 和 var*_dtype
    inputs = []
    i = 0
    while True:
        shape_key = f"var{i}_shape"
        dtype_key = f"var{i}_dtype"
        if shape_key not in target_row:
            break
        shape_val = target_row.get(shape_key, "").strip()
        dtype_val = target_row.get(dtype_key, "float16").strip()
        if not shape_val:
            break
        try:
            shape = ast.literal_eval(shape_val)
        except Exception:
            break
        inputs.append({"shape": shape, "dtype": dtype_val})
        i += 1

    if not inputs:
        return None

    # 收集非 var*_shape/dtype 的列作为 init_params
    init_params = {}
    for k, v in target_row.items():
        if k == "case_id" or k.startswith("var") and ("_shape" in k or "_dtype" in k):
            continue
        if v and v.strip():
            try:
                init_params[k] = ast.literal_eval(v.strip())
            except Exception:
                init_params[k] = v.strip()

    return {
        "case_id": target_row.get("case_id", "unknown"),
        "inputs": inputs,
        "init_params": init_params,
    }


def _dtype_str_to_torch(dtype_str: str) -> str:
    """将 CSV 中的 dtype 字符串转换为 torch.xxx 格式。"""
    mapping = {
        "float16": "torch.float16",
        "float32": "torch.float32",
        "float64": "torch.float64",
        "bfloat16": "torch.bfloat16",
        "int8": "torch.int8",
        "int16": "torch.int16",
        "int32": "torch.int32",
        "int64": "torch.int64",
        "bool": "torch.bool",
        "uint8": "torch.uint8",
    }
    return mapping.get(dtype_str.lower(), f"torch.{dtype_str}")


def _generate_csv_inputs_code(test_case: dict) -> str:
    """
    生成从 CSV test case 构造输入张量的代码片段（替代 get_inputs()）。

    Returns:
        Python 代码字符串，执行后 `inputs` 变量包含输入张量列表。
    """
    lines = ["    # Inputs from test_cases.csv (case: {})".format(test_case["case_id"])]
    lines.append("    inputs = []")
    for i, inp in enumerate(test_case["inputs"]):
        torch_dtype = _dtype_str_to_torch(inp["dtype"])
        lines.append(
            f"    inputs.append(torch.randn({inp['shape']}, dtype={torch_dtype}))"
        )
    return "\n".join(lines)


def _generate_run_script(
    work_dir: Path,
    op_name: str,
    script_path: Path,
    device_id: int = 0,
    test_case: dict = None,
) -> bool:
    """
    生成一个独立的 Python 脚本，用于在 msprof op simulator 下运行算子。

    自动检测 forward() 中的 permute 调用:
    - Path A (有 permute): 在 CPU 上执行 permute，直接调用 custom_ops_lib.xxx_custom()
    - Path B (无 permute): 直接调用 module_fn()

    同时自动设置 ASCEND_CUSTOM_OPP_PATH (如果 vendors/ 下存在自定义算子目录)。

    Args:
        work_dir: 算子工作目录 (含 {op_name}_custom.py, pybind_lib/ 等)
        op_name: 算子名称
        script_path: 生成脚本的输出路径
        device_id: NPU 设备 ID
        test_case: 从 test_cases.csv 解析的测试用例（覆盖 get_inputs()）

    Returns:
        True if script generated successfully
    """
    custom_py = work_dir / f"{op_name}_custom.py"
    if not custom_py.exists():
        return False

    info = _parse_custom_module(work_dir, op_name)

    pybind_lib = work_dir / "pybind_lib"
    pybind_insert = ""
    if pybind_lib.is_dir():
        pybind_insert = f"""
# pybind_lib isolation
import sys as _sys
_pbl = r"{pybind_lib}"
if _pbl not in _sys.path:
    _sys.path.insert(0, _pbl)
if "custom_ops_lib" in _sys.modules:
    del _sys.modules["custom_ops_lib"]
"""

    env_block = _build_env_block(work_dir)

    # Determine where get_inputs / get_init_inputs come from
    if info["has_get_inputs"]:
        inputs_module_path = str(custom_py)
        inputs_module_name = "op_module"
    elif info.get("functional_py"):
        inputs_module_path = info["functional_py"]
        inputs_module_name = "func_module"
    else:
        inputs_module_path = str(custom_py)
        inputs_module_name = "op_module"

    if info["has_permute"] and info["custom_call"]:
        script_content = _generate_direct_call_script(
            work_dir, custom_py, info, pybind_insert, env_block,
            inputs_module_path, inputs_module_name, device_id,
            test_case=test_case,
        )
    else:
        script_content = _generate_module_fn_script(
            work_dir, custom_py, info, pybind_insert, env_block,
            inputs_module_path, inputs_module_name, device_id,
            test_case=test_case,
        )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_content, encoding="utf-8")
    return True


def _generate_direct_call_script(
    work_dir, custom_py, info, pybind_insert, env_block,
    inputs_module_path, inputs_module_name, device_id,
    test_case=None,
) -> str:
    """
    Path A: forward() 中有 permute — 在 CPU 上执行 permute，直接调用 custom_ops_lib。
    避免 permute().contiguous() 在 NPU 上触发 Transpose 内核。
    """
    pre_perm = info["pre_permute"]
    custom_call = info["custom_call"]
    module_fn_args = info.get("module_fn_args", "")

    # Build the extra-args loading from get_init_inputs
    if module_fn_args:
        arg_names = [a.strip() for a in module_fn_args.split(",")]
        init_unpack = ", ".join(arg_names) + f" = {inputs_module_name}.get_init_inputs()"
        extra_args = ", " + ", ".join(arg_names)
    else:
        init_unpack = f"_ = {inputs_module_name}.get_init_inputs()  # no extra args"
        extra_args = ""

    # Load func_module if different from op_module
    func_import = ""
    if inputs_module_name != "op_module":
        func_import = (
            f'func_spec = importlib.util.spec_from_file_location("func_module", r"{inputs_module_path}")\n'
            f'func_module = importlib.util.module_from_spec(func_spec)\n'
            f'func_spec.loader.exec_module(func_module)\n'
        )

    # Input generation: use test_case CSV shapes if provided, else get_inputs()
    if test_case:
        csv_inputs_code = _generate_csv_inputs_code(test_case)
        inputs_block = (
            f'{csv_inputs_code}\n'
        )
    else:
        inputs_block = (
            f'    inputs = {inputs_module_name}.get_inputs()\n'
        )

    return (
        f'#!/usr/bin/env python3\n'
        f'"""Auto-generated script for msprof op simulator profiling (direct call, no permute on NPU)."""\n'
        f'import sys\n'
        f'sys.path.insert(0, r"{work_dir}")\n'
        f'{pybind_insert}\n'
        f'{env_block}\n'
        f'import torch\n'
        f'import torch_npu\n'
        f'torch_npu.npu.set_device({device_id})\n'
        f'\n'
        f'import importlib.util\n'
        f'spec = importlib.util.spec_from_file_location("op_module", r"{custom_py}")\n'
        f'op_module = importlib.util.module_from_spec(spec)\n'
        f'spec.loader.exec_module(op_module)\n'
        f'{func_import}\n'
        f'import custom_ops_lib\n'
        f'\n'
        f'# Run: permute on CPU, then direct custom call on NPU\n'
        f'with torch.no_grad():\n'
        f'    {init_unpack}\n'
        f'{inputs_block}\n'
        f'    # Pre-permute on CPU to avoid Transpose kernel on NPU\n'
        f'    x = inputs[0].permute({pre_perm}).contiguous()\n'
        f'    x = x.npu()\n'
        f'\n'
        f'    # Direct custom op call — only this kernel is captured by msprof\n'
        f'    output = {custom_call}(x{extra_args})\n'
        f'    torch_npu.npu.synchronize()\n'
    )


def _generate_module_fn_script(
    work_dir, custom_py, info, pybind_insert, env_block,
    inputs_module_path, inputs_module_name, device_id,
    test_case=None,
) -> str:
    """
    Path B: forward() 中无 permute — 通过 ModelNew 实例化并执行前向推理。
    """
    func_import = ""
    if inputs_module_name != "op_module":
        func_import = (
            f'func_spec = importlib.util.spec_from_file_location("func_module", r"{inputs_module_path}")\n'
            f'func_module = importlib.util.module_from_spec(func_spec)\n'
            f'func_spec.loader.exec_module(func_module)\n'
        )

    # Input generation: use test_case CSV shapes if provided, else get_inputs()
    if test_case:
        csv_inputs_code = _generate_csv_inputs_code(test_case)
        inputs_block = (
            f'{csv_inputs_code}\n'
        )
    else:
        inputs_block = (
            f'    inputs = {inputs_module_name}.get_inputs()\n'
        )

    return (
        f'#!/usr/bin/env python3\n'
        f'"""Auto-generated script for msprof op simulator profiling (ModelNew forward)."""\n'
        f'import sys\n'
        f'sys.path.insert(0, r"{work_dir}")\n'
        f'{pybind_insert}\n'
        f'{env_block}\n'
        f'import torch\n'
        f'import torch_npu\n'
        f'torch_npu.npu.set_device({device_id})\n'
        f'\n'
        f'import importlib.util\n'
        f'spec = importlib.util.spec_from_file_location("op_module", r"{custom_py}")\n'
        f'op_module = importlib.util.module_from_spec(spec)\n'
        f'spec.loader.exec_module(op_module)\n'
        f'{func_import}\n'
        f'# Instantiate and run\n'
        f'with torch.no_grad():\n'
        f'    init_inputs = {inputs_module_name}.get_init_inputs()\n'
        f'    model = op_module.ModelNew(*init_inputs)\n'
        f'    model = model.npu()\n'
        f'    model.eval()\n'
        f'\n'
        f'{inputs_block}'
        f'    inputs = [x.npu() if isinstance(x, torch.Tensor) else x for x in inputs]\n'
        f'\n'
        f'    # Single forward pass for simulator trace\n'
        f'    output = model(*inputs)\n'
        f'    torch_npu.npu.synchronize()\n'
    )

def _detect_simulator_backend() -> str:
    """
    检测可用的simulator后端，返回"cannsim"或"msprof"或None。

    优先级：msprof op simulator > cannsim
    cannsim 是Ascend950专用工具，msprof op simulator依赖msopprof。
    """
    # 检查 msprof op simulator （需要msopprof）
    msprof_path = shutil.which("msprof")
    if msprof_path:
        # msprof op simulator 需要 msopprof 二进制
        ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
        msopprof = os.path.join(ascend_home, "tools", "msopprof", "bin", "msopprof")
        if os.path.isfile(msopprof):
            return "msprof"
    # 检查 cannsim 
    cannsim_path = shutil.which("cannsim")
    if cannsim_path:
        return "cannsim"
    return None

def find_cannsim_output(output_dir: str) -> str:
    """
    在 output_dir 中查找cannsim 生成的trace文件目录。

    cannsim输出结构：
        output_dir/cannsim_{timestamp}_{app}/report/trace_core0.json
    
    为了与下游T1/T2/T3 分析工具兼容（它们期望 coreN.veccoreM/trace.json)，
    本函数会创建兼容的符号链接目录结构：
      output_dir/simulator/core0.veccore0/trace.json -> ../../cannsim_.../report/trace_core0.json

    Returns:
        兼容目录路径（output_dir/simulator/)，未找到则返回 None
    """
    if not os.path.isdir(output_dir):
        return None

    # 查找 cannsim_* 目录
    entries = sorted(
        [e for e in os.listdir(output_dir) if e.startswith("cannsim_")],
        reverse=True,
    )

    for entry in entries:
        cannsim_dir = os.path.join(output_dir, entry)
        if not os.path.isdir(cannsim_dir):
            continue

        # 查找 report/ 子目录中的trace_coreN.json文件
        report_dir = os.path.join(cannsim_dir, "report")
        if not os.path.isdir(report_dir):
            # cannsim --gen-report 可能还没生成report，尝试手动生成
            continue

        trace_files = sorted(glob.glob(os.path.join(report_dir, "trace_core*.json")))
        if not trace_files:
            continue
        
        #创建兼容目录结构：simulator/coreN.veccore0/trace.json
        compat_dir = os.path.join(output_dir, "simulator")
        os.makedirs(compat_dir, exist_ok=True)

        for trace_file in trace_files:
            fname = os.path.basename(trace_file)  # trace_core0.json
            # 提取核号：trace_core0.json -> 0
            m = re.match(r"trace_core(\d+)\.json", fname)
            if not m:
                continue
            core_id = m.group(1)
            core_dir_name = f"core{core_id}.veccore0"
            core_dir = os.path.join(compat_dir, core_dir_name)
            os.makedirs(core_dir, exist_ok=True)

            target_trace = os.path.join(core_dir, "trace.json")
            if os.path.exists(target_trace):
                os.remove(target_trace)
            # 复制文件 （而非符号链接，避免跨目录问题）
            shutil.copy2(trace_file, target_trace)
        # 验证至少有一个core目录
        for d in os.listdir(compat_dir):
            if d.startswith("core"):
                trace = os.path.join(compat_dir, d, "trace.json")
                if os.path.isfile(trace):
                    return compat_dir

    return None

def find_simulator_output(output_dir: str) -> str:
    """
    在 output_dir 中查找 simulator trace 输出目录。

    支持两种后端的输出格式:
      1. msprof：output_dir/OPPROF_*/simulator/coreN.veccoreM/trace.json
      2. cannsim：output_dir/cannsim_*/report/trace_coreN.json
         (自动转换为兼容格式：output_dir/simulator/coreN.veccore0/trace.json)

    Returns:
        simulator 目录路径，未找到则返回 None
    """
    if not os.path.isdir(output_dir):
        return None

    # 策略1 ：检查已有的兼容目录（simulator/）
    compat_dir = os.path.join(output_dir, "simulator")
    if os.path.isdir(compat_dir):
        for d in os.listdir(compat_dir):
            if d.startswith("core"):
                trace = os.path.join(compat_dir, d, "trace.json")
                if os.path.isfile(trace):
                    return compat_dir

    # 策略2 ：msprof格式（OPPROF_*/simulator)
    entries = sorted(
        [e for e in os.listdir(output_dir) if e.startswith("OPPROF_")],
        reverse=True,
    )

    for entry in entries:
        opprof_dir = os.path.join(output_dir, entry)
        if not os.path.isdir(opprof_dir):
            continue

        # 递归查找 simulator/ 子目录
        for root, dirs, files in os.walk(opprof_dir):
            if "simulator" in dirs:
                sim_dir = os.path.join(root, "simulator")
                # 验证有 core 子目录含 trace.json
                for d in os.listdir(sim_dir):
                    if d.startswith("core"):
                        trace = os.path.join(sim_dir, d, "trace.json")
                        if os.path.isfile(trace):
                            return sim_dir

    # 策略3：cannsim 格式（cannsim_*/report/trace_coreN.json)
    cannsim_result = find_cannsim_output(output_dir)
    if cannsim_result:
        return cannsim_result

    return None


def _detect_soc_version() -> str:
    """通过 npu-smi 自动检测芯片型号，返回 msprof 可用的 soc-version 字符串。

    npu-smi info 输出格式 (每个 NPU 两行):
      | 0     910B3               | OK            | ...
      | 0                         | 0000:C1:00.0  | ...
    第一行的第一个 cell 包含 "NPU_ID  芯片名"（如 "0     910B3"），
    用正则从中提取芯片型号。
    """
    try:
        out = subprocess.run(
            "npu-smi info", shell=True, capture_output=True, text=True, timeout=10
        )
        for line in out.stdout.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            # 在整行中搜索芯片型号: 3位数字 + 可选字母 + 可选数字 (910B3, 310P3, 610)
            m = re.search(r'\b(\d{3}[A-Za-z]\w*)\b', line)
            if m:
                return f"Ascend{m.group(1)}"
    except Exception:
        pass
    return None


def _find_simulator_lib(soc_version: str = None) -> str:
    """
    查找 simulator 动态库目录。

    CANN 的 set_env.sh 默认不会将 simulator lib 加入 LD_LIBRARY_PATH，
    导致 msprof op simulator 报 "Failed to load simulator so"。
    本函数自动定位该路径以便注入环境变量。

    搜索顺序:
      1. $ASCEND_HOME_PATH/tools/simulator/{soc_version}/lib/
      2. $ASCEND_TOOLKIT_HOME/tools/simulator/{soc_version}/lib/
      3. 遍历 /usr/local/Ascend/ascend-toolkit/latest/ 等常见路径
    """
    candidates = []
    for env_var in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        base = os.environ.get(env_var)
        if base:
            candidates.append(Path(base) / "tools" / "simulator")

    # 常见安装路径兜底
    candidates.append(Path("/usr/local/Ascend/ascend-toolkit/latest/tools/simulator"))

    for sim_root in candidates:
        if not sim_root.is_dir():
            continue
        # 优先匹配指定的 soc_version
        if soc_version:
            lib_dir = sim_root / soc_version / "lib"
            if lib_dir.is_dir():
                return str(lib_dir)
        # 未指定时，尝试自动检测
        detected = _detect_soc_version()
        if detected:
            lib_dir = sim_root / detected / "lib"
            if lib_dir.is_dir():
                return str(lib_dir)
        # 最后兜底：找任何含 libruntime_camodel.so 的子目录
        for child in sorted(sim_root.iterdir()):
            lib_dir = child / "lib"
            if (lib_dir / "libruntime_camodel.so").is_file():
                return str(lib_dir)

    return None


def run_simulator_profiling(
    work_dir: str,
    op_name: str,
    output_dir: str = None,
    device_id: int = 0,
    soc_version: str = None,
    aic_metrics: str = "PipeUtilization",
    timeout: int = 3600,
    test_case_csv: str = None,
    case_id: str = None,
) -> dict:
    """
    运行 simulator profiling（自动选择cannsim或者msprof op simulator）。

    流程:
    1. 检测可用后端（msprof op simulator优先）
    2. 在 work_dir 下生成临时运行脚本
    3. 调用simulator工具（msporf： msprof op simulator --application="python3 {script}" --output={output_dir} 或者 cannsim）
    4. 查找生成的 trace 目录（自动转换为兼容格式）

    Args:
        work_dir: 算子工作目录 (含 {op_name}_custom.py)
        op_name: 算子名称
        output_dir: profiling 输出目录 (默认: work_dir/simulator_prof)
        device_id: NPU 设备 ID
        soc_version: SoC 版本 (如 "Ascend910B3" 或者 "Ascend950"，默认自动检测)
        aic_metrics: 采集指标 (默认 PipeUtilization, 仅msprof使用)
        timeout: 超时秒数 (默认 3600，即 1 小时；最大 21600 即 6 小时)
        test_case_csv: test_cases.csv 路径，提供时使用其中的 shape 替代 get_inputs()
        case_id: CSV 中的 case_id，未指定时选元素数最大的 case

    Returns:
        {"success": bool, "simulator_dir": str|None, "error": str|None, "command": str, "backend": str}
    """
    # 限制 timeout 范围: 最低 60 秒, 最高 21600 秒 (6 小时)
    timeout = max(60, min(timeout, 21600))

    work_dir = Path(work_dir).resolve()
    if output_dir is None:
        output_dir = str(work_dir / "simulator_prof")
    output_dir = str(Path(output_dir).resolve())

    # step 0: 检测 simulator 后端
    backend = _detect_simulator_backend()
    if not backend:
        return {
            "success": False,
            "simulator_dir": None,
            "error": "No simulator backend found. Need cannsim (Ascend950) or msprof op simulator (with msopprof)",
            "backend": None,
        }
    # Step 0.5: 解析 test_case_csv（若提供）
    test_case = None
    if test_case_csv:
        test_case = _parse_test_case_csv(test_case_csv, case_id)
        if test_case:
            from functools import reduce
            import operator
            total_elems = sum(
                reduce(operator.mul, inp["shape"], 1)
                if inp["shape"] else 0
                for inp in test_case["inputs"]
            )
            print(f"[profiling_runner] Using test case '{test_case['case_id']}' "
                  f"from {test_case_csv} (total elements: {total_elems})")
        else:
            print(f"[profiling_runner] WARNING: Failed to parse {test_case_csv}, "
                  f"falling back to get_inputs()")

    # Step 1: 生成运行脚本
    script_path = work_dir / "_profiling_sim_run.py"
    if not _generate_run_script(work_dir, op_name, script_path, device_id,
                                test_case=test_case):
        return {
            "success": False,
            "simulator_dir": None,
            "error": f"{op_name}_custom.py not found in {work_dir}",
            "backend": backend,
        }

    # Step 2: 构建命令和环境变量
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    # 设置 ASCEND_CUSTOM_OPP_PATH
    custom_opp = _find_custom_opp_path(work_dir)
    if custom_opp:
        env["ASCEND_CUSTOM_OPP_PATH"] = custom_opp
        lib_path = _find_custom_lib_path(work_dir)
        if lib_path:
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            if lib_path not in existing_ld:
                env["LD_LIBRARY_PATH"] = f"{lib_path}:{existing_ld}".rstrip(":")

    if backend == "cannsim":
        cmd = _build_cannsim_command(script_path, output_dir, soc_version)
    else:
        # msprof op simulator
        if not soc_version:
            soc_version = _detect_soc_version()
        cmd = _build_msprof_command(script_path, output_dir, soc_version, aic_metrics)
        # 注入simulator动态库路径
        sim_lib = _find_simulator_lib(soc_version)
        if sim_lib:
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            if sim_lib not in existing_ld:
                env["LD_LIBRARY_PATH"] = f"{sim_lib}:{existing_ld}".rstrip(":")

    print(f"[profiling_runner] Backend: {backend}, Command: {cmd}")

    # Step 3: 执行
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_dir),
            env=env,
        )

        # 清理临时脚本
        if script_path.exists():
            script_path.unlink()
        wrapper_path = script_path.parent / "_cannsim_wrapper.sh"
        if wrapper_path.exists():
            wrapper_path.unlink()
        
        if result.returncode != 0:
            return {
                "success": False,
                "simulator_dir": None,
                "error": f"{backend} failed (rc={result.returncode}): {result.stderr[-500:]}",
                "command": cmd,
                "backend": backend,
                "stdout": result.stdout[-500:] if result.stdout else "",
            }

        # Step 3.5: cannsim 需要额外的report步骤（如果没有 --gen--report）
        if backend == "cannsim":
            _cannsim_ensure_report(output_dir, soc_version)
        
        # Step 4: 查找生成的 simulator 目录
        simulator_dir = find_simulator_output(output_dir)

        if not simulator_dir:
            return {
                "success": False,
                "simulator_dir": None,
                "error": f"{backend} completed but no trace found in {output_dir}",
                "command": cmd,
                "backend": backend,
                "stdout": result.stdout[-500:] if result.stdout else "",
            }

        return {
            "success": True,
            "simulator_dir": simulator_dir,
            "command": cmd,
            "backend": backend,
            "stdout": result.stdout[-200:] if result.stdout else "",
        }

    except FileNotFoundError:
        if script_path.exists():
            script_path.unlink()
        wrapper_path = script_path.parent / "_cannsim_wrapper.sh"
        if wrapper_path.exists():
            wrapper_path.unlink()
        return {
            "success": False,
            "simulator_dir": None,
            "error": f"{backend} not found in PATH. Ensure CANN is installed and sourced.",
            "backend": backend,
        }
    except subprocess.TimeoutExpired:
        if script_path.exists():
            script_path.unlink()
        wrapper_path = script_path.parent / "_cannsim_wrapper.sh"
        if wrapper_path.exists():
            wrapper_path.unlink()
        return {
            "success": False,
            "simulator_dir": None,
            "error": f"{backend} timed out after {timeout} seconds",
            "command": cmd,
            "backend": backend,
        }

def _build_cannsim_command(script_path: Path, output_dir:str, soc_version: str = None) -> str:
    """构建cannsim record 命令。
    cannsim 的 user_app参数必须是一个可执行文件路径（非shell命令）。
    因此需要先创建一个可执行的bash wrapper 脚本， 内部调用 python3 运行
    实际的profiling脚本。
    """
    soc = soc_version or "Ascend950"
    # 创建 bash wrapper脚本
    wrapper_path = script_path.parent / "_cannsim_wrapper.sh"
    wrapper_content = (
        f'#!/bin/bash\n'
        f'exec python3 {script_path}\n'
    )
    wrapper_path.write_text(wrapper_content, encoding="utf-8")
    os.chmod(str(wrapper_path), 0o755)
    cmd = (
        f'cannsim record'
        f' {wrapper_path}"'
        f' -s {soc}'
        f' -o {output_dir}'
        f' --gen-report'
    )
    return cmd


def _build_msprof_command(script_path: Path, output_dir:str, soc_version: str, aic_metrics: str) -> str:
    """构建msprof op simulator 命令。"""
    cmd = (
        f'msprof op simulator'
        f' --application="python3 {script_path}"'
        f' --output={output_dir}'
        f' --aic-metrics={aic_metrics}'
    )
    if soc_version:
        cmd += f' --soc-version={soc_version}'
    return cmd

def _cannsim_ensure_report(output_dir: str, soc_version: str = None) -> None:
    """
    确保 cannsim 输出目录中有 report/ 子目录
    如果 --gen-report 已生成则跳过，否则手动调用 cannsim report。
    """
    entries = sorted(
        [e for e in os.listdir(output_dir) if e.startswith("cannsim_")],
        reverse=True,
    )
    for entry in entries:
        cannsim_dir = os.path.join(output_dir, entry)
        report_dir = os.path.join(cannsim_dir, "report")
        if os.path.isdir(report_dir) and glob.glob(os.path.join(report_dir, "trace_core*.json")):
            return # report 已存在
        # 手动生成 report
        cmd = f'cannsim report -e {cannsim_dir} -o {report_dir} -n all'
        try:
            subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        except Exception:
            pass
        return 

def main():
    parser = argparse.ArgumentParser(
        description="Run msprof op simulator to generate instruction-level trace data"
    )
    parser.add_argument("--work-dir", required=True,
                        help="Operator work directory (containing {op_name}_custom.py)")
    parser.add_argument("--op-name", required=True,
                        help="Operator name")
    parser.add_argument("--output-dir", default=None,
                        help="Profiling output directory (default: work_dir/simulator_prof)")
    parser.add_argument("--device-id", type=int, default=0,
                        help="NPU device ID (default: 0)")
    parser.add_argument("--soc-version", default=None,
                        help="SoC version (e.g., Ascend910B3)")
    parser.add_argument("--aic-metrics", default="PipeUtilization",
                        help="AIC metrics to collect (default: PipeUtilization)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Timeout in seconds (default: 3600, i.e. 1 hour; max: 21600, i.e. 6 hours)")
    parser.add_argument("--test-case-csv", default=None,
                        help="Path to test_cases.csv; when provided, use its shapes instead of get_inputs()")
    parser.add_argument("--case-id", default=None,
                        help="Specific case_id from test_cases.csv (default: largest case by element count)")
    args = parser.parse_args()

    result = run_simulator_profiling(
        work_dir=args.work_dir,
        op_name=args.op_name,
        output_dir=args.output_dir,
        device_id=args.device_id,
        soc_version=args.soc_version,
        aic_metrics=args.aic_metrics,
        timeout=args.timeout,
        test_case_csv=args.test_case_csv,
        case_id=args.case_id,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
