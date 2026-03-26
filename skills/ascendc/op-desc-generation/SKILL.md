---
name: op-desc-generation
description: Generate operator description JSON from template
---

## What I do

Generate operator description JSON files from the template for Ascend operator development. This is the first step in the operator generation pipeline.

## When to use me

Use this when you need to create an operator description JSON file for a new operator before generating PyTorch reference code, AscendDSL, or AscendC implementations.

## Workflow

1. Read the template file @references/op_desc_template.json and example file @references/layer_norm_op_desc.json
2. Analyze the operator requirements and extract necessary information. If not mentioned, refer to the API definition in PyTorch.
   - Operator name (e.g., average_pooling2d, conv2d, softmax)
   - Category (choose from: pooling, activation, convolution, reduction, normalization, matmul, loss, optimizer, math)
   - Mathematical description and behavior
   - Input tensor shapes and data types
   - Output tensor shapes and data types
   - Attributes (kernel_size, stride, padding, eps, etc.)
3. Create operator description JSON with appropriate fields:
   - `op_name`: Unique identifier for the operator
   - `category`: Operator classification for organization
   - `description`: Clear mathematical definition and behavior
   - `shape_info`: Information related to tensors in the parameters with names, shapes, dtypes.
   - `attributes`: The parameters other than the tensor parameters which are generally kwargs and have default values.
4. Save the JSON file in current project `output/{op_name}/{op_name}_op_desc.json`