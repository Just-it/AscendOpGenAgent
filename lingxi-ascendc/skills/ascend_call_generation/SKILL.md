---
name: ascend-call-generation
description: Generate Ascend function call code and Ascend C project
---

## What I do

Generate Ascend operator invocation code and project json configuration file from functional PyTorch code and then create initial Ascend C project.

## When to use me

Use this after functional conversion to create operator metadata for DSL generation.

## Workflow
1. read input functional pytorch code `{op_name}_functional.py`
2. read three example output files in `references/{op_name}` dir by op catagory
    - For pool ops: `average_pooling2d`
    - For reduction ops: `sum_reduction_over_a_dimension`
    - For loss ops: `mse_loss`
    - Other: `layer_norm`
3. generate code and save in file.
    - project json code for msopgen to create the custom Ascend operate project
    - python bind code for C -> python API interface
    - python code to call the custom Ascend C operate
4. save all file in `output/{op_name}/` directory.
5. create Ascend C project(run gen_project.py)

### Generation Task
Construct operator name as `{op_name}_custom` and convert to PascalCase for operator definition.

#### 1. project_json code
JSON schema defining the custom AscendC operator:
- **inputs**: ND tensors mapping to [module_fn] tensor arguments
- **outputs**: ND tensors mapping to [module_fn] return values
- **attributes**: scalar/config parameters from [module_fn]

**Critical**: Ensure 1-to-1 correspondence with [module_fn] arguments. Do NOT add attributes that don't appear in [module_fn] signature.

Save json code in `{op_name}_project.json`

#### 2. python_bind code
C++ pybind11 code connecting AscendC operator to PyTorch:
- Include headers: `torch/library.h`, `pytorch_npu_helper.hpp`
- Implement `*_impl_npu` function calling `EXEC_NPU_CMD`
- Register with `TORCH_LIBRARY_IMPL`
- Export via `PYBIND11_MODULE`

**Requirements**:
- Function name must match `{op_name}_custom`
- Handle negative dimension parameters (dim) properly
- Expose function with **exact same signature** as [module_fn]

Save C code in `{op_name}.cpp`


#### 3. model code
`ModelNew(nn.Module)` class that:
- Replicates original `Model` functionality
- Calls custom AscendC operator via `custom_ops_lib.{op_name}_custom`
- Maintains same `__init__` and `forward` signatures
- Imports: `torch`, `torch.nn`, `torch_npu`, `custom_ops_lib`

**Note**: Only implement specific behavior used by original Model. For example, if Model always uses `dim=1`, don't implement arbitrary dim cases.

Save python code in `{op_name}_custom.py`

### create AscendC project
You should use `gen_project.py` to create the Ascend C project.

Usage:
```shell
python3 .opencode/skills/ascend_call_generation/scripts/gen_project.py <op_name> <json_file_path>
```
Check if the project generate success in `output` dir.
If not, analysis the error and try to use right parameters re-run the script.
