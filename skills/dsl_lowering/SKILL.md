---
name: dsl-lowering
description: Translate the operator DSL into AscendC code through multiple passes.
subagent:
  enabled: true
  agent_type: Build
  reason: "DSL lowering involves 4 complex transformation passes (tiling, init, process, nonaligned), each requiring detailed code analysis and generation. Using a subagent allows autonomous multi-step execution with error recovery."
  timeout: 900  # 15 minutes
  max_iterations: 15
  model: "myprovider/glm-5"
  console_output: true
  silent: false 
---


## What I do

Translate the operator DSL into AscendC code through four passes: `tiling_pass`, `init_pass`, `process_pass`, and `process_nonaligned_pass`.
The input AscendC code may already include modifications from earlier passes. Apply **only** the transformation described in the task.


## When to use me

Use this after project creation to lower DSL code to AscendC code

## Input Parameters

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `op_name` | str | 是 | 算子名称 |
| `output_dir` | str | 否 | 输出目录路径，默认为 `output/{op_name}` |

## Workflow

Sub-Agent Strategy

This skill uses a subagent to handle the complex multi-pass transformation:
- The subagent autonomously executes all 4 passes sequentially
- Handles compilation errors with automatic retry (up to 3 attempts)
- Manages intermediate file state and error recovery
- Reports progress after each pass

1. Read the operator DSL file `{output_dir}/{op_name}_dsl.py` (default: `output/{op_name}/{op_name}_dsl.py`) and the directory created by the previous AscendC project generation step

2. Based on the operator DSL file, sequentially execute four transformation passes. After each pass generates AscendC code, invoke the `build.sh` script in the AscendC project to perform compilation. If a compilation error occurs, refer to the repair examples in `references/error_correction/` for guidance. Specifically, the four transformation passes are as follows:

2.1 `tiling_pass`'s task description is below：

"""
### Tiling Instruction for DSL → AscendC Conversion

When converting AscendDSL kernels into AscendC, the **tiling logic must be moved to the host side**.  
The purpose of tiling is to compute all scalar parameters, attributes, and workspace requirements on the host and store them in a tiling struct that will be passed to the AscendC kernel.

AscendC kernels follow the SPMD execution model: each core handles one data block.  
Thus, tiling must determine how data is partitioned across cores.

---

### Required Components in Tiling

#### 2.1.1. Define a Tiling Struct
Create a tiling data structure.
This struct stores:
- All scalar values originally computed in DSL.
- Attributes (if any).

#### 2.1.2. Compute and Set Tiling Parameters
All scalar computations that were inside DSL must be moved to the host:
- Core partitioning logic (e.g. rows per core, tiles per core)
- UB tiling parameters (tile sizes, loop counts)
- Shape-derived constants

#### 2.1.3. Set Kernel Attributes
If used attributes, add corresponding fields into the tiling struct.
Access attributes via context->GetAttrs()->GetAttrPointer<float>(index).

#### 2.1.4. Set Workspace Sizes

Workspace should be allocated in AscendC **only when the original DSL host code explicitly allocates a GM buffer for intermediate results**.

In other words:

- If the DSL contains statements like:
```python
workspace = torch.empty(n, dtype=torch.float32, device=...)
```
In such cases:
```cpp
#include "tiling/platform/platform_ascendc.h" // must include for using PlatformAscendC

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = requiredWorkspaceBytes + sysWorkspaceSize;
```
"""

2.2 Refer to the existing knowledge of AscendC API in `references/ascend_api/`, example usages of specific APIs are as follows:

**API Name:** `GetAttrPointer`  

**Function Prototype:**  
```cpp
template<typename T>
const T* GetAttrPointer(size_t index) const
```

**Usage Example:** 
To obtain the first attribute (negative_slope):
```cpp
namespace ops {
class LeakyReluCustom : public OpDef {
public:
    LeakyReluCustom(const char *name) : OpDef(name)
    {
        // Define the attribute with default value
        this->Attr("negative_slope").AttrType(OPTIONAL).Float(0.0);
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    gert::TilingContext context;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();

    // Access the attribute value by index
    const float* negativeSlope = attrs->GetAttrPointer<float>(0);
}
}
```
'''

2.3 Based on the operator category, refer to the most appropriate `tiling` template for the corresponding operator under the `references/lowering_examples/` directory and perform the transformation accordingly.


3. `init_pass`' task description is below：

"""
Implement kernel code init part in AscendC kernels based on the AscendDSL code. 

### DSL to AscendC Kernel Key Principles
#### 3.1. Parameter and Core Setup

| DSL Concept | AscendC Implementation | Principle |
| :--- | :--- | :--- |
| **DSL Tiling Parameters** (`rows_per_core`, `tile_length`, etc.) | Store these as **`uint32_t` member variables** within the AscendC class. | These parameters determine the loop bounds and data access size within the `Process()` method. |
| **program_id** | Retrieve using **`GetBlockIdx()`** in the `Init()` method. | This value is used to calculate the starting GM address for the core's data partition. |
| **Global Memory (GM) Access Ranges** | Calculated and configured using **`SetGlobalBuffer`**. | Configure the **`In`** and **`Out`** Tensors (member variables) to ensure the AI Core accesses only its assigned block of data from Global Memory (GM). |

---

#### 3.2. Buffer and Queue Initialization

You must strictly select the type of variable, either TBuf (Calculation Buffer) or TQue (Data Queue), based on its purpose in the data flow. This is a core requirement for AscendC kernel design.

| DSL Buffer Type | AscendC Object Type | VEC TPosition | Initialization and Purpose |
| :--- | :--- | :--- | :--- |
| **Input Buffers** (e.g., ub used in copyin) | **`TQue`** (Tensor Queue) | **VECIN** (Vector Input) | **Purpose:** Used for synchronized data transfer into the `Compute` function. **Initialization:** `pipe.InitBuffer(TENSOR_QUEUE, slot_count, SIZE_IN_BYTES)`. |
| **Output Buffers** (e.g., ub used in copyout) | **`TQue`** (Tensor Queue) | **VECOUT** (Vector Output) | **Purpose:** Used for synchronized data transfer out of the `Compute` function to `CopyOut`. **Initialization:** `pipe.InitBuffer(TENSOR_QUEUE, slot_count, SIZE_IN_BYTES)`. |
| **Intermediate/Working Buffers** (`tmp_ub`, `shared_ub`) | **`TBuf`** (Tensor Buffer) | **VECCALC** (Vector Calculation) | **Purpose:** Used for temporary, short-term storage during a single `Compute` phase (e.g., reduction workspace, intermediate results). **Initialization:** `pipe.InitBuffer(TENSOR_BUFFER, SIZE_IN_BYTES)`. |

"""

3.3 Based on the operator category, refer to the most appropriate `init` template for the corresponding operator under the `references/lowering_examples/` directory and perform the transformation accordingly.


3.4 Refer to the existing knowledge of AscendC API in `references/ascend_api/`.

4. `process_pass`'s task description is below：

"""
Implement the kernel process part in AscendC kernels based on AscendDSL code.

### DSL to AscendC Kernel Key Principles
#### 4.1. Overall Code Structure and Control Flow

The AscendC code must map the DSL's overall workload execution model onto the AI Core's execution environment.

| DSL Concept | AscendC Implementation | Principle |
| :--- | :--- | :--- |
| **Workload Loop** (Iterating over partitioned data/tiles) | Encapsulated in the **`Process()`** method. | `Process()` manages the execution loop over the data units assigned to the core. |
| **Computational Phases** | `Process()` must call dedicated functions: **`CopyInX`**, **`ComputeX`**, and **`CopyOutX`**. |  Use numbering (e.g., `CopyIn1`, `Compute2`) if the core logic involves multiple distinct passes or data movement steps. |
| **Function Definition** | Each stage function must be defined as `__aicore__ inline` and accept the current loop index (`uint32_t idx`) if needed for global memory (GM) address calculation. | Standard kernel function attributes and structure. |

---

#### 4.2. Data and Buffer Management

This section defines how DSL's memory allocation and movement map to AscendC's Queue and Tensor Buffer (`TBuf`) system for managing the Unified Buffer (UB).

##### 4.2.1. Data Loading (`CopyInX` Functions)

1.  **Allocate UB Space:** Use the appropriate input queue's **`AllocTensor<T>()`** method to reserve a local tensor in the UB.
2.  **Move Data (GM to UB):** Use **`AscendC::DataCopy`** to transfer the data tile from Global Memory (DSL's input tensor) to the local UB tensor.
3.  **Transfer to Compute:** Use the input queue's **`EnQue()`** method to pass the loaded local tensor to the next stage.

##### 4.2.2. Computation (`ComputeX` Functions)

1.  **Acquire All Tensors (Mandatory Start):** At the **very beginning** of every `ComputeX` function, obtain all tensors required for that calculation phase:
    * **Input Tensors:** Dequeue from the input queue(s) (`inQueue.DeQue<T>()`).
    * **Working Buffers:** For internal temporary buffers, use the pre-defined member Tensor Buffers' (`TBuf`) **`Get<T>()`** method (e.g., `sharedBuf.Get<float>()`).
2.  **Execute Logic:** Translate DSL operations to AscendC APIs.
3.  **Flow Control:**
    * If the result is needed by a subsequent stage (`Compute` or `CopyOut`), use the output queue's **`EnQue()`** method.
    * Once an input tensor is processed and no longer needed, use **`FreeTensor()`** on its originating queue.
4.  **Global synchronization**
When different kernels operate on the same global memory and potential data dependency issues arise, synchronization statements `AscendC::SyncAll()` are inserted.

##### 4.2.3. Data Storing (`CopyOutX` Functions)

1.  **Acquire Result:** Use the output queue's **`DeQue<T>()`** method to retrieve the final result tensor from the previous stage.
2.  **Move Data (UB to GM):** Use **`AscendC::DataCopy`** to transfer the result from the local UB tensor to the DSL's output Global Tensor.
3.  **Release UB Space:** Use **`FreeTensor()`** on the output queue to release the local tensor buffer.

---
"""

4.3 Refer to the existing knowledge of AscendC API in `references/ascend_api/`, example usages of specific APIs are as follows:

'''
API Name: Compare
API Description: Performs element-wise comparison between two tensors. For each element, if the comparison result is true, the corresponding bit in the output tensor is set to 1; otherwise, it is set to 0.
Parameter List:
  - dstLocal: Destination operand. Type: LocalTensor<uint8_t>
  - src0Local, src1Local: Source operands. Type: LocalTensor
  - cmpMode: Comparison mode of type CMPMODE, which includes the following options: EQ, NE, GE, LE, GT, LT.
  - count: Number of input data elements.
Example:
```cpp
AscendC::Compare(dstLocal, src0Local, src1Local, AscendC::CMPMODE::LT, srcDataSize);
```

API Name: CompareScalar
API Description: Performs element-wise comparison between a tensor and a scalar. For each element, if the comparison result is true, the corresponding bit in the output tensor is set to 1; otherwise, it is set to 0.
Parameter List:
  - dstLocal: Destination operand. Type: LocalTensor<uint8_t>
  - src0Local: Source operand. Type: LocalTensor
  - src1Scalar: Scalar operand.
  - cmpMode: Comparison mode of type CMPMODE, which includes EQ, NE, GE, LE, GT, LT.
  - count: Number of input data elements

API Name: Select
API Description: Generates the destination tensor dst by selecting elements from two source operands (src0 and src1) according to the bit values in selMask.
Parameter List:
  - dstLocal: Destination operand. Type: LocalTensor
  - selMask: Selection mask tensor. Type: LocalTensor<uint8_t>
  - src0Local: Source operand. Type: LocalTensor
  - src1Local/src1Scalar: Source operand, can be a LocalTensor or a scalar.
  - selMode: VSEL_TENSOR_SCALAR_MODE or VSEL_TENSOR_TENSOR_MODE
  - count: Number of input data elements
Example:
```cpp
AscendC::Select(dstLocal, maskLocal, src0Local, src1Local, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, dataSize);
```

API Name: LocalTensor.GetValue and GlobalTensor.GetValue
API Description: Retrieves the value at a specific offset in a LocalTensor or GlobalTensor.
Parameter List:
  - offset: Offset by this number of elements
Return Value:
  - Returns an immediate value of type T

API Name: Duplicate
API Description: Duplicates a variable or an immediate value multiple times and fills them into a vector.
Parameter List:
  - dstLocal: Destination operand. Type: LocalTensor.
  - scalarValue: Source operand to be duplicated. Supports both variables and immediate values. The data type must match the element data type in dstLocal.
  - calCount: Number of data elements to be filled.

API Name: DataCopy
API Description: Data transfer interface for aligned data. 
Parameter List:
  - dstLocal or dstGlobal: Destination operand. Type: LocalTensor or GlobalTensor.
  - srcGlobal or srcLocal: Source operand. Type: GlobalTensor or LocalTensor.
  - calCount: Number of elements involved in the transfer. The amount of data transferred must be a multiple of 32 bytes.

API Name: TBuf.Get
API Description: Retrieves a LocalTensor from a TBuf object. The tensor can span the entire allocated buffer or a specified portion of it.
Example:
```cpp
// Create a vector calculation buffer and initialize with 1024 bytes
AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
uint32_t byteLen = 1024;
pipe.InitBuffer(calcBuf, byteLen);

// Get a tensor spanning the entire buffer (1024 bytes = 256 int32_t elements)
AscendC::LocalTensor<int32_t> fullTensor = calcBuf.Get<int32_t>();

// Get a tensor with 128 int32_t elements (512 bytes)
AscendC::LocalTensor<int32_t> partialTensor = calcBuf.Get<int32_t>(128);
```
'''


4.4 Based on the operator category, refer to the most appropriate `process` template for the corresponding operator under the `references/lowering_examples/` directory and perform the transformation accordingly.



5. `proess_nonaligned_pass`'s task description is below：

"""
Process non-32-byte–aligned data transfers in AscendC kernels and **output the full, complete kernel code**.

For all data transfers in the `CopyIn` and `CopyOut` stages:
- If the **data size being transferred** is 32-byte aligned, keep the original `AscendC::DataCopy` implementation.
- If the **data size is not 32-byte aligned**, replace `AscendC::DataCopy` with `AscendC::DataCopyPad`.

⚠️ Important:
- Do NOT output partial code or diffs.
"""

5.1  Refer to the existing knowledge of AscendC API in `references/ascend_api/`, example usages of specific APIs are as follows:

'''
API Name: DataCopy
API Description: Data transfer interface for aligned data. 
Parameter List:
  - dstLocal or dstGlobal: Destination operand. Type: LocalTensor or GlobalTensor.
  - srcGlobal or srcLocal: Source operand. Type: GlobalTensor or LocalTensor.
  - calCount: Number of elements involved in the transfer. The amount of data transferred must be a multiple of 32 bytes.

API Name: DataCopyPad
API Description: Data transfer interface for non-aligned data.
Parameter List:
  - dstLocal or dstGlobal: Destination operand. Type: LocalTensor or GlobalTensor.
  - srcGlobal or srcLocal: Source operand. Type: GlobalTensor or LocalTensor.
  - dataCopyParams, needed for all situations, typically set to {1, count * sizeof(dtype), 0, 0} for contiguous data.
  - padParams, only needed for Global Memory->Local Memory, usually {false, 0, 0, 0} when padding is not required.
Example:
```cpp
AscendC::DataCopyPad(dstGlobal, srcLocal, {1, static_cast<uint16_t>(1 * sizeof(float)), 0, 0, 0}); // four parameters for dataCopyParams: blockCount, blockLen, srcStride, dstStride
AscendC::DataCopyPad(dstLocal, srcGlobal, {1, static_cast<uint16_t>(20 * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0}); // Global Memory->Local Memory, needs four parameters
```
'''

5.2  Refer to the examples in the `references/lowering_examples/non_aligned/` directory for the transformation.

## Note

- Apply **only** the transformation described in the task.  
- Follow the logic strictly as defined in the AscendDSL code.  Do **not** introduce extra logic (e.g., conditional checks).  
- Maintain the original code formatting and structure.
- The AscendC code generated by each pass overwrites the existing files directly in the AscendC project.
- In case of compilation failure, attempt up to **three** rounds of automatic repair.
- Provide clear status messages after each pass transformation and after each compilation attempt.