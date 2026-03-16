### Example input dsl
```python
import tile.language as tl

@ascend_kernel
def leaky_relu_kernel(input_ptr, output_ptr,
                      elements_per_core, tile_size, inner_loops, negative_slope):

    pid = tl.program_id(0)
    start = pid * elements_per_core

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    x_ub          = tl.alloc_ub(tile_size, dtype=tl.float32)
    pos_ub        = tl.alloc_ub(tile_size, dtype=tl.float32)
    neg_ub        = tl.alloc_ub(tile_size, dtype=tl.float32)
    out_ub        = tl.alloc_ub(tile_size, dtype=tl.float32)

    # ------------------------------------------------------------
    # Tile loop
    # ------------------------------------------------------------
    for i in range(inner_loops):
        tile_start = start + i * tile_size
        offsets = tile_start + tl.arange(0, tile_size)

        # --------------------------------------------------------
        # COPYIN
        # --------------------------------------------------------
        with tl.copyin():
            tl.load(input_ptr + offsets, x_ub)

        # --------------------------------------------------------
        # COMPUTE
        # --------------------------------------------------------
        with tl.compute():
            # pos = max(x, 0)
            tl.vmax(pos_ub, x_ub, 0.0)

            # neg = min(x, 0)
            tl.vmin(neg_ub, x_ub, 0.0)

            # neg_scaled = neg * negative_slope
            tl.vmul_scalar(neg_ub, neg_ub, negative_slope)

            # out = pos + neg_scaled
            tl.vadd(out_ub, pos_ub, neg_ub)

        # --------------------------------------------------------
        # COPYOUT
        # --------------------------------------------------------
        with tl.copyout():
            tl.store(output_ptr + offsets, out_ub)


def leaky_relu_host(x: torch.Tensor, output: torch.Tensor, negative_slope: float):
    total_elems = x.numel()

    # ------------------------------------------------------------
    # Core Partitioning
    # ------------------------------------------------------------
    n_cores = 16
    elements_per_core = total_elems // n_cores

    # ------------------------------------------------------------
    # Tiling Strategy
    # ------------------------------------------------------------
    tile_size = 2048
    inner_loops = elements_per_core // tile_size

    # ------------------------------------------------------------
    # Launch kernel
    # ------------------------------------------------------------
    leaky_relu_kernel[n_cores](
        x, output,
        elements_per_core,
        tile_size,
        inner_loops,
        negative_slope
    )


```
### Example input AscendC
```
#include "kernel_operator.h"

class KernelLeakyRelu {
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> posBuf, negBuf;
    AscendC::GlobalTensor<float> inputGm;
    AscendC::GlobalTensor<float> outputGm;
    uint32_t elementsPerCore;
    uint32_t tileSize;
    uint32_t innerLoops;
    float alpha;

public:
    __aicore__ inline KernelLeakyRelu() {}
    __aicore__ inline void Init(GM_ADDR input_ptr, GM_ADDR output_ptr, uint32_t elementsPerCore, uint32_t tileSize, uint32_t innerLoops, float alpha)
    {
        this->elementsPerCore = elementsPerCore;
        this->tileSize = tileSize;
        this->innerLoops = innerLoops;
        this->alpha = alpha;

        // Set global memory buffer. Offset is calculated based on block index
        uint32_t start = elementsPerCore * AscendC::GetBlockIdx();
        uint32_t totalElements = elementsPerCore;
        
        inputGm.SetGlobalBuffer((__gm__ float *)input_ptr + start, totalElements);
        outputGm.SetGlobalBuffer((__gm__ float *)output_ptr + start, totalElements);

        // Initialize pipe buffer queues with one slot, each holding tileSize floats
        pipe.InitBuffer(inQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(outQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(posBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(negBuf, this->tileSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // TODO implemented
    }
};

extern "C" __global__ __aicore__ void leaky_relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLeakyRelu op;
    op.Init(x, y, tiling_data.elementsPerCore, tiling_data.tileSize, tiling_data.innerLoops, tiling_data.alpha);
    op.Process();
}
```

### Example output AscendC
```
#include "kernel_operator.h"

class KernelLeakyRelu {
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> posBuf, negBuf;
    AscendC::GlobalTensor<float> inputGm;
    AscendC::GlobalTensor<float> outputGm;
    uint32_t elementsPerCore;
    uint32_t tileSize;
    uint32_t innerLoops;
    float alpha;

public:
    __aicore__ inline KernelLeakyRelu() {}
    __aicore__ inline void Init(GM_ADDR input_ptr, GM_ADDR output_ptr, uint32_t elementsPerCore, uint32_t tileSize, uint32_t innerLoops, float alpha)
    {
        this->elementsPerCore = elementsPerCore;
        this->tileSize = tileSize;
        this->innerLoops = innerLoops;
        this->alpha = alpha;

        // Set global memory buffer. Offset is calculated based on block index
        uint32_t start = elementsPerCore * AscendC::GetBlockIdx();
        uint32_t totalElements = elementsPerCore;
        
        inputGm.SetGlobalBuffer((__gm__ float *)input_ptr + start, totalElements);
        outputGm.SetGlobalBuffer((__gm__ float *)output_ptr + start, totalElements);

        // Initialize pipe buffer queues with one slot, each holding tileSize floats
        pipe.InitBuffer(inQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(outQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(posBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(negBuf, this->tileSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->innerLoops; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx)
    {
        AscendC::LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
        AscendC::DataCopy(inputLocal, inputGm[idx * this->tileSize], this->tileSize);
        inQueue.EnQue(inputLocal);
    }
    
    __aicore__ inline void Compute(uint32_t idx)
    {
        AscendC::LocalTensor<float> inputLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();
        
        AscendC::LocalTensor<float> posLocal = posBuf.Get<float>();
        AscendC::LocalTensor<float> negLocal = negBuf.Get<float>();
        
        // pos = max(x, 0)
        AscendC::Maxs(posLocal, inputLocal, 0.0f, this->tileSize);
        
        // neg = min(x, 0)
        AscendC::Mins(negLocal, inputLocal, 0.0f, this->tileSize);
        
        // neg_scaled = neg * alpha
        AscendC::Muls(negLocal, negLocal, this->alpha, this->tileSize);
        
        // out = pos + neg_scaled
        AscendC::Add(outputLocal, posLocal, negLocal, this->tileSize);
        
        outQueue.EnQue<float>(outputLocal);
        inQueue.FreeTensor(inputLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t idx)
    {
        AscendC::LocalTensor<float> outputLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(outputGm[idx * this->tileSize], outputLocal, this->tileSize);
        outQueue.FreeTensor(outputLocal);
    }
};

extern "C" __global__ __aicore__ void leaky_relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLeakyRelu op;
    op.Init(x, y, tiling_data.elementsPerCore, tiling_data.tileSize, tiling_data.innerLoops, tiling_data.alpha);
    op.Process();
}
```