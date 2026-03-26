### Example input dsl
```python
import torch
import tile.language as tl

@ascend_kernel
def softmax_kernel(input_ptr, output_ptr, rows_per_core, tile_length):
    pid = tl.program_id(0)
    row_start_idx = pid * rows_per_core
    row_end_idx = row_start_idx + rows_per_core

    # Allocate UB Buffers 
    row_ub      = tl.alloc_ub(tile_length, dtype=tl.float32) 
    exp_ub      = tl.alloc_ub(tile_length, dtype=tl.float32) 
    shared_ub   = tl.alloc_ub(tile_length, dtype=tl.float32) 
    out_ub      = tl.alloc_ub(tile_length, dtype=tl.float32) 
    
    # Computation Logic
    for row_idx in range(row_start_idx, row_end_idx):
        offsets = row_idx * tile_length + tl.arange(0, tile_length)

        with tl.copyin():
            tl.load(input_ptr + offsets, row_ub)

        with tl.compute():
            tl.reduce_max(shared_ub, row_ub, shared_ub) 
            row_max = extract_scalar(shared_ub, 0)
            tl.vsub_scalar(exp_ub, row_ub, row_max) 
            tl.vexp(exp_ub, exp_ub) 
            tl.reduce_sum(shared_ub, exp_ub, shared_ub)
            row_sum = extract_scalar(shared_ub, 0)
            tl.vdiv_scalar(out_ub, exp_ub, row_sum)

        with tl.copyout():
            tl.store(output_ptr + offsets, out_ub)

def softmax_host(x: torch.Tensor, output: torch.Tensor):
    rows = x.shape[0]
    cols = x.shape[1]
    
    # Core Partitioning
    n_cores = 16 
    rows_per_core = rows // n_cores 

    # Tiling Strategy
    # Entire row fits into UB.
    tile_length = cols  
    
    softmax_kernel[n_cores](x, output, rows_per_core, tile_length)

```

### Example input AscendC
```
host_tiling_src="""

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SoftmaxCustom, SoftmaxCustomTilingData)
}

"""

host_operator_src="""

#include "softmax_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 16;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    SoftmaxCustomTilingData tiling;
    const gert::StorageShape* input_shape = context->GetInputShape(0);
    
    // Extract shape dimensions for 2D tensor
    uint32_t nRows = input_shape->GetStorageShape().GetDim(0);
    uint32_t nCols = input_shape->GetStorageShape().GetDim(1);
    
    context->SetBlockDim(BLOCK_DIM);
    
    // Calculate rows per core: (n_rows + n_cores - 1) // n_cores
    uint32_t rowsPerCore = (nRows + BLOCK_DIM - 1) / BLOCK_DIM;
    
    tiling.set_tileLength(nCols);
    tiling.set_rowsPerCore(rowsPerCore);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class SoftmaxCustom : public OpDef {
public:
    explicit SoftmaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(SoftmaxCustom);
}

"""

kernel_src="""
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
"""
```

### Example output AscendC
```
host_tiling_src="""

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SoftmaxCustom, SoftmaxCustomTilingData)
}

"""

host_operator_src="""

#include "softmax_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 16;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    SoftmaxCustomTilingData tiling;
    const gert::StorageShape* input_shape = context->GetInputShape(0);
    
    // Extract shape dimensions for 2D tensor
    uint32_t nRows = input_shape->GetStorageShape().GetDim(0);
    uint32_t nCols = input_shape->GetStorageShape().GetDim(1);
    
    context->SetBlockDim(BLOCK_DIM);
    
    // Calculate rows per core: (n_rows + n_cores - 1) // n_cores
    uint32_t rowsPerCore = (nRows + BLOCK_DIM - 1) / BLOCK_DIM;
    
    tiling.set_tileLength(nCols);
    tiling.set_rowsPerCore(rowsPerCore);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class SoftmaxCustom : public OpDef {
public:
    explicit SoftmaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(SoftmaxCustom);
}

"""

kernel_src="""
#include "kernel_operator.h"

class KernelSoftmax {
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expBuf, sharedBuf;
    AscendC::GlobalTensor<float> inputGm;
    AscendC::GlobalTensor<float> outputGm;
    uint32_t tileLength;
    uint32_t rowsPerCore;

public:
    __aicore__ inline KernelSoftmax() {}
    __aicore__ inline void Init(GM_ADDR input_ptr, GM_ADDR output_ptr, uint32_t tileLength, uint32_t rowsPerCore)
    {
        this->tileLength = tileLength;
        this->rowsPerCore = rowsPerCore;

        // Set global memory buffer. Offset is calculated based on block index
        uint32_t rowStart = rowsPerCore * AscendC::GetBlockIdx();
        uint32_t totalElements = rowsPerCore * tileLength;
        
        inputGm.SetGlobalBuffer((__gm__ float *)input_ptr + rowStart * tileLength, totalElements);
        outputGm.SetGlobalBuffer((__gm__ float *)output_ptr + rowStart * tileLength, totalElements);

        // Initialize pipe buffer queues with one slot, each holding tileLength floats
        pipe.InitBuffer(inQueue, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueue, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(expBuf, this->tileLength * sizeof(float));
        pipe.InitBuffer(sharedBuf, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // TODO implemented
    }
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmax op;
    op.Init(x, y, tiling_data.tileLength, tiling_data.rowsPerCore);
    op.Process();
}
"""