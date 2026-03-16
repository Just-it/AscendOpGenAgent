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
```cpp
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
```

### Example output AscendC
```cpp
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
        for (uint32_t i = 0; i < this->rowsPerCore; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
        AscendC::DataCopy(inputLocal, inputGm[rowIdx * this->tileLength], this->tileLength);
        inQueue.EnQue(inputLocal);
    }
    
    __aicore__ inline void Compute(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> inputLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();
        
        AscendC::LocalTensor<float> expLocalTensor = expBuf.Get<float>(); 
        AscendC::LocalTensor<float> sharedLocalTensor = sharedBuf.Get<float>();
        
        // Find max value in the row 
        AscendC::ReduceMax(sharedLocalTensor, inputLocal, sharedLocalTensor, this->tileLength);
        float maxVal = sharedLocalTensor.GetValue(0);
        
        // Subtract max from all elements
        AscendC::Adds(expLocalTensor, inputLocal, -maxVal, this->tileLength);
        
        // Compute exponential
        AscendC::Exp(expLocalTensor, expLocalTensor, this->tileLength);
        
        // Compute sum of exponentials 
        AscendC::ReduceSum(sharedLocalTensor, expLocalTensor, sharedLocalTensor, this->tileLength);

        // Divide by sum
        AscendC::Reciprocal(sharedLocalTensor, sharedLocalTensor, 1);
        AscendC::Muls(outputLocal, expLocalTensor, sharedLocalTensor.GetValue(0), this->tileLength);
        
        outQueue.EnQue<float>(outputLocal);
        inQueue.FreeTensor(inputLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> outputLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(outputGm[rowIdx * this->tileLength], outputLocal, this->tileLength);
        outQueue.FreeTensor(outputLocal);
    }
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmax op;
    op.Init(x, y, tiling_data.tileLength, tiling_data.rowsPerCore);
    op.Process();
}
```