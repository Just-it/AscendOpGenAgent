---
name: ascendc-evaluation
description: Deploy and evaluate AscendC operators
---

## What I do

Generate PyBind bindings for AscendC operator, install the operate run file and evaluate operator correctness against reference implementations.

## When to use me

Use this when you need to evaluate operator correctness against reference implementations.

## Workflow
1. install `custom_opp_ubuntu_aarch64.run` or `custom_opp_ubuntu_x86_x64.run`file.
2. generate pybind and install the whl(run generate_pybind.py)
3. evaluate operator correctness and performance

### install run file
 It usually locates in `output/{op_name}/{op_name}Custom/build_out/` dir.
 You must only use --install-path=`output/{op_name}` to run it. And you must use absolute path for --install-path argument.

### generate pybind
You need to run `generate_pybind.py`, it will generate pybind whl and use pip to install this whl. Get the console output of the executed script and print it.

Usage:
```shell
python3 skills/ascendc/ascendc-evaluation/scripts/generate_pybind.py <op_name>
```

### evaluate operator correctness
You need to run `evaluate.py` to evaluate the correctness of the operator. It sets environment variables `ASCEND_CUSTOM_OPP_PATH`, adds `.so` file path in `LD_LIBRARY_PATH`, and evaluates the correctness and performance.

Usage:
```shell
python3 skills/ascendc/ascendc-evaluation/scripts/evaluate.py <op_name>
```

Console output:
- result of evaluation
    - pass: result of performance
    - fail: correctness fail message

