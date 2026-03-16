---
name: dsl-baseline-generation
description: Generate baseline AscendDSL code from functional PyTorch
---

## What I do

Generate initial AscendDSL code with proper class structure, compute methods, and tiling logic for NPU architecture.

## When to use me

Use this after ascend call generation to create initial DSL implementation.

## Workflow

1. Read input functional PyTorch code `{op_name}_functional.py`
2. Select appropriate example:
   - read example input file in dir: `references/input_example/`
   - read example output file in dir: `references/output_example/`
3. read ascend dsl knowledge @references/ascendDSL.py
4. generate Ascend DSL code
5. Save to `output/{op_name}/{op_name}_dsl.py`

### input explain
- **[module_fn]**: a pure PyTorch functional implementation  
- **[Model Definition]**: a `Model(nn.Module)` calling [module_fn]  
- **[Configurations]**: hyper-parameters and input helper functions  

### generate requirment
Your task is to generate an **Ascend DSL** that replicates the computation in [module_fn], optimized for the input shape specified in Configurations.  
You only can launch a kernel once.
Follow the implementation patterns demonstrated in the example, use the **same number of launched cores** as in the example, and adopt a similar core partitioning and tiling strategy where applicable.
Note that the Ascend DSL host side must use the same argument list as module_fn.
Therefore, the inputs passed from the host to module_fn must already match the exact shapes expected by module_fn, including any shapes that result from transposed dimensions or similar transformations.
Natural-language comments are mandatory and are part of the DSL.
