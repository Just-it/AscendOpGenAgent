---
name: ascendc-evaluation
description: Deploy and evaluate AscendC operators
---

## What I do

Generate PyBind bindings for AscendC operator, install the operate run file and evaluate operator correctness against reference implementations.

## When to use me

Use this when you need to evaluate operator correctness against reference implementations.

## Input Parameters

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `op_name` | str | 是 | 算子名称 |
| `output_dir` | str | 否 | 输出目录路径，默认为 `output/{op_name}` |
| `warmup` | int | 否 | 性能测试 warmup 次数，默认 5 |
| `repeats` | int | 否 | 性能测试重复次数，默认 50 |

## Workflow
1. install `custom_opp_ubuntu_aarch64.run` file.
2. generate pybind and install the whl(run generate_pybind.py)
3. evaluate operator correctness

### install run file
 It usually locates in `{output_dir}/{op_name}Custom/build_out/` dir (default: `output/{op_name}/{op_name}Custom/build_out/`).
 You must only use --install-path=`{output_dir}` to run it. And you must use absolute path for --install-path argument.

### generate pybind
You need to run `generate_pybind.py`, it will generate pybind whl and use pip to install this whl. Get the console output of the executed script and print it.

Usage:
```shell
python3 .opencode/skills/ascendc_evalution/scripts/generate_pybind.py <op_name> [--output_dir <目录>]
```

### evaluate operator correctness
You need to run `evaluate.py` to evaluate the correctness of the operator. It sets environment variables `ASCEND_CUSTOM_OPP_PATH`, adds `.so` file path in `LD_LIBRARY_PATH`, and evaluates the correctness.

**evaluate.py输出很长，需要重定向并对文件尾部分析。**

Usage:
```shell
python3 .opencode/skills/ascendc_evalution/scripts/evaluate.py <op_name> [--output_dir <目录>] &> evaluate.log
```

Console output:
- result of correctness
    - pass: result of performance
    - fail: correctness fail message

### evaluate performance
Perform the operation only after the correctness is successful. You need to run `benchmark.py` to evaluate the performance. It sets environment variables `ASCEND_CUSTOM_OPP_PATH`, adds `.so` file path in `LD_LIBRARY_PATH`, and evaluates the performance.

Usage:
```shell
python3 .opencode/skills/ascendc_evalution/scripts/performance.py  \
    --op_name <op_name> \
    --verify_dir <input path> \
    --warmup <number of warmup> \
    --repeats <number of repeats> \
    --output <ouput path>
```

performance.py输出很长，需要重定向并对文件分析。

Console output:
- result of performance
    - pass: result of performance
    - fail: correctness fail message