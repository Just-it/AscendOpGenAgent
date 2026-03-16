### Example input dsl
```python
import tile.language as tl

@ascend_kernel
def mse_loss_kernel(
    pred_ptr,                 # [N]
    target_ptr,               # [N]
    workspace,               # [n_cores]
    output_ptr,               # final scalar
    elems_per_core,
    tile_size,
    inner_loops,
    total_elems               # for computing mean
):
    pid = tl.program_id(0)
    start = pid * elems_per_core

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    pred_ub    = tl.alloc_ub(tile_size, dtype=tl.float32)
    target_ub  = tl.alloc_ub(tile_size, dtype=tl.float32)
    diff_ub    = tl.alloc_ub(tile_size, dtype=tl.float32)
    sq_ub      = tl.alloc_ub(tile_size, dtype=tl.float32)
    shared_ub  = tl.alloc_ub(tile_size, dtype=tl.float32)
    workspace_out_ub = tl.alloc_ub(tile_size, dtype=tl.float32)
    workspace_in_ub   = tl.alloc_ub(tile_size, dtype=tl.float32)
    output_ub   = tl.alloc_ub(tile_size, dtype=tl.float32)

    

    # ------------------------------------------------------------
    # Phase 1: per-core partial reduction of squared errors
    # ------------------------------------------------------------
    partial_sum = 0.0

    for i in range(inner_loops):
        tile_start = start + i * tile_size
        offsets = tile_start + tl.arange(0, tile_size)

        # -------------------------------
        # COPYIN
        # -------------------------------
        with tl.copyin():
            tl.load(pred_ptr   + offsets, pred_ub)
            tl.load(target_ptr + offsets, target_ub)

        # -------------------------------
        # COMPUTE ((pred - target)^2)
        # -------------------------------
        with tl.compute():
            tl.vsub(diff_ub, pred_ub, target_ub)
            tl.vmul(sq_ub, diff_ub, diff_ub)
            tl.reduce_sum(sq_ub, sq_ub, shared_ub)
            tile_sum = extract_scalar(sq_ub, 0)
            partial_sum = partial_sum + tile_sum

    # ------------------------------------------------------------
    # Write per-core partial sum
    # ------------------------------------------------------------
    with tl.copyout():
        tl.set_scalar(workspace_out_ub, 0, partial_sum)
        tl.store(workspace+tl.arange(pid,pid+1), workspace_out_ub)

    # ------------------------------------------------------------
    # Phase 2: Core 0 performs final reduce + mean
    # ------------------------------------------------------------
    if pid == 0:

        # ---------------------------
        # Load all partial sums
        # ---------------------------
        with tl.copyin():
            tl.load(workspace + tl.arange(0, tl.num_programs(0)), workspace_in_ub)

        # ---------------------------
        # Reduce
        # ---------------------------
        with tl.compute():
            tl.reduce_sum(shared_ub, workspace_in_ub, shared_ub)
            sum_sq = extract_scalar(shared_ub, 0)
            mse = sum_sq / float(total_elems)
            tl.set_scalar(output_ub, 0, mse)

        # ---------------------------
        # Copyout final scalar
        # ---------------------------
        with tl.copyout():
           tl.store(output_ptr+tl.arange(0, 1), output_ub)

def mse_loss_host(pred: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    total_elems = pred.numel()

    # ------------------------------------------------------------
    # Core Partitioning
    # ------------------------------------------------------------
    n_cores = 16
    elems_per_core = total_elems // n_cores

    # GM buffer for per-core partial results
    workspace = torch.empty(n_cores, dtype=torch.float32, device=pred.device)

    # ------------------------------------------------------------
    # Tiling Strategy
    # ------------------------------------------------------------
    tile_size = 2048
    inner_loops = elems_per_core // tile_size

    # ------------------------------------------------------------
    # Launch kernel
    # ------------------------------------------------------------
    mse_loss_kernel[n_cores](
        pred,
        target,
        workspace,
        output,
        elems_per_core,
        tile_size,
        inner_loops,
        total_elems
    )


```
### Example input AscendC
```
host_tiling_src="""

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MseLossCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MseLossCustom, MseLossCustomTilingData)
}

"""

host_operator_src="""

#include "mse_loss_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  MseLossCustomTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  tiling.set_size(data_sz);
  context->SetBlockDim(8);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

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
class MseLossCustom : public OpDef {
public:
    explicit MseLossCustom(const char* name) : OpDef(name)
    {
        this->Input("predictions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("targets")
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

OP_ADD(MseLossCustom);
}

"""

kernel_src="""
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void mse_loss_custom(GM_ADDR predictions, GM_ADDR targets, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
"""
```

### Example output AscendC
```python
host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MseLossCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalElems);
  TILING_DATA_FIELD_DEF(uint32_t, elemsPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
  TILING_DATA_FIELD_DEF(uint32_t, innerLoops);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MseLossCustom, MseLossCustomTilingData)
}
"""

host_operator_src="""
#include "mse_loss_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
const uint32_t N_CORES = 16;
const uint32_t TILE_SIZE = 2048;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  MseLossCustomTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  
  // Calculate total number of elements
  uint32_t totalElems = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
    totalElems *= x1_shape->GetStorageShape().GetDim(i);
  }
  
  // Core partitioning
  uint32_t elemsPerCore = totalElems / N_CORES;
  
  // Tiling strategy
  uint32_t innerLoops = elemsPerCore / TILE_SIZE;
  
  context->SetBlockDim(N_CORES);
  
  tiling.set_totalElems(totalElems);
  tiling.set_elemsPerCore(elemsPerCore);
  tiling.set_tileSize(TILE_SIZE);
  tiling.set_innerLoops(innerLoops);
  
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  
  // Set workspace size for per-core partial results
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t requiredWorkspaceBytes = N_CORES * sizeof(float);
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = requiredWorkspaceBytes + sysWorkspaceSize;

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
class MseLossCustom : public OpDef {
public:
    explicit MseLossCustom(const char* name) : OpDef(name)
    {
        this->Input("predictions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("targets")
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

OP_ADD(MseLossCustom);
}

"""

kernel_src="""
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void mse_loss_custom(GM_ADDR predictions, GM_ADDR targets, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
"""