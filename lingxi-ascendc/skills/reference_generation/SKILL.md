---
name: reference-generation
description: Generate reference PyTorch code from operator description JSON
---

## What I do

Generate functional reference PyTorch code from operator description JSON files. Output uses `torch.nn` with `nn.Module` classes and proper type hints.

## When to use me

Use this when you need to create a golden reference implementation for an operator from its description.

## Workflow

1. Read operator description json file `{op_name}_op_desc.json`
2. Extract operator name, category, description, shape information, parameters
3. Read output example code @references/layer_norm.py
4. Generate PyTorch code:
   - Imports: specify using `torch` and `torch.nn`
   - Model Definition: A `Model(nn.Module)` class with proper `__init__` and `forward` methods.
   - Configurations: hyper-parameters and two helpers `get_inputs()` and `get_init_inputs()` consumed by the model.
5. Save to `output/{op_name}/{op_name}_reference.py`

**key Points**:
1. The `get_inputs()` function should generate input tensors according to the shape_info provided:
   - Use the specified shapes and dtypes from shape_info
   - If dtype is "float32", use `torch.rand()`
   - If dtype is "int32" or "int64", use `torch.randint()`
   - If dtype is "float16" or "bfloat16", use `torch.rand()` and cast to the dtype
2. The `get_init_inputs()` function should return the initialization parameters specified in the parameters field
3. The `Model` class should implement the operator according to the description
4. If the operator is to compute the gradient, you should:
   - Please manually implement the computation in the `forward` function instead of using the builtin autograd
   - If the gradient computation requires the output of the original forward pass, compute this output in `get_inputs()` and pass it as an input to the `forward` function

