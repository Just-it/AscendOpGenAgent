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
```cpp
#include "kernel_operator.h"

class KernelMseLoss {
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> predQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> targetQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> workspaceInQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> workspaceOutQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> diffBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sqBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sharedBuf;
    AscendC::GlobalTensor<float> predGm;
    AscendC::GlobalTensor<float> targetGm;
    AscendC::GlobalTensor<float> outputGm;
    AscendC::GlobalTensor<float> workspaceGm;
    uint32_t totalElems;
    uint32_t elemsPerCore;
    uint32_t tileSize;
    uint32_t innerLoops;
    uint32_t programId;

public:
    __aicore__ inline KernelMseLoss() {}
    __aicore__ inline void Init(GM_ADDR pred_ptr, GM_ADDR target_ptr, GM_ADDR output_ptr, GM_ADDR workspace_ptr, 
                                 uint32_t totalElems, uint32_t elemsPerCore, uint32_t tileSize, uint32_t innerLoops)
    {
        this->totalElems = totalElems;
        this->elemsPerCore = elemsPerCore;
        this->tileSize = tileSize;
        this->innerLoops = innerLoops;
        this->programId = AscendC::GetBlockIdx();

        uint32_t start = programId * elemsPerCore;

        // Set global memory buffers
        predGm.SetGlobalBuffer((__gm__ float *)pred_ptr + start, elemsPerCore);
        targetGm.SetGlobalBuffer((__gm__ float *)target_ptr + start, elemsPerCore);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace_ptr, 16);  // N_CORES = 16
        outputGm.SetGlobalBuffer((__gm__ float *)output_ptr, 1);

        // Initialize input queues for copyin
        pipe.InitBuffer(predQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(targetQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(workspaceInQueue, 1, 16 * sizeof(float));  // for loading workspace in core 0

        // Initialize output queues for copyout
        pipe.InitBuffer(workspaceOutQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(outputQueue, 1, this->tileSize * sizeof(float));

        // Initialize intermediate/working buffers for compute
        pipe.InitBuffer(diffBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(sqBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(sharedBuf, this->tileSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // TODO
    }

};

extern "C" __global__ __aicore__ void mse_loss_custom(GM_ADDR predictions, GM_ADDR targets, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMseLoss op;
    op.Init(predictions, targets, y, workspace, tiling_data.totalElems, tiling_data.elemsPerCore, tiling_data.tileSize, tiling_data.innerLoops);
    op.Process();
}
```

### Example output AscendC
```cpp
#include "kernel_operator.h"

class KernelMseLoss {
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> predQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> targetQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> workspaceInQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> workspaceOutQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> diffBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sqBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sharedBuf;
    AscendC::GlobalTensor<float> predGm;
    AscendC::GlobalTensor<float> targetGm;
    AscendC::GlobalTensor<float> outputGm;
    AscendC::GlobalTensor<float> workspaceGm;
    uint32_t totalElems;
    uint32_t elemsPerCore;
    uint32_t tileSize;
    uint32_t innerLoops;
    uint32_t programId;

public:
    __aicore__ inline KernelMseLoss() {}
    __aicore__ inline void Init(GM_ADDR pred_ptr, GM_ADDR target_ptr, GM_ADDR output_ptr, GM_ADDR workspace_ptr, 
                                 uint32_t totalElems, uint32_t elemsPerCore, uint32_t tileSize, uint32_t innerLoops)
    {
        this->totalElems = totalElems;
        this->elemsPerCore = elemsPerCore;
        this->tileSize = tileSize;
        this->innerLoops = innerLoops;
        this->programId = AscendC::GetBlockIdx();

        uint32_t start = programId * elemsPerCore;

        // Set global memory buffers
        predGm.SetGlobalBuffer((__gm__ float *)pred_ptr + start, elemsPerCore);
        targetGm.SetGlobalBuffer((__gm__ float *)target_ptr + start, elemsPerCore);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace_ptr, 16);  // N_CORES = 16
        outputGm.SetGlobalBuffer((__gm__ float *)output_ptr, 1);

        // Initialize input queues for copyin
        pipe.InitBuffer(predQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(targetQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(workspaceInQueue, 1, 16 * sizeof(float));  // for loading workspace in core 0

        // Initialize output queues for copyout
        pipe.InitBuffer(workspaceOutQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(outputQueue, 1, this->tileSize * sizeof(float));

        // Initialize intermediate/working buffers for compute
        pipe.InitBuffer(diffBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(sqBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(sharedBuf, this->tileSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // Phase 1: per-core partial reduction of squared errors
        float partialSum = 0.0f;

        for (uint32_t i = 0; i < this->innerLoops; i++) {
            CopyIn1(i);
            Compute1(i, partialSum);
        }

        CopyOut1(partialSum);
        AscendC::SyncAll();

        // Phase 2: Core 0 performs final reduce + mean
        if (programId == 0) {
            CopyIn2();
            Compute2();
            CopyOut2();
        }
    }

private:
    __aicore__ inline void CopyIn1(uint32_t i)
    {
        AscendC::LocalTensor<float> predLocal = predQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> targetLocal = targetQueue.AllocTensor<float>();
        
        uint32_t tileStart = i * this->tileSize;
        AscendC::DataCopy(predLocal, predGm[tileStart], this->tileSize);
        AscendC::DataCopy(targetLocal, targetGm[tileStart], this->tileSize);
        
        predQueue.EnQue(predLocal);
        targetQueue.EnQue(targetLocal);
    }

    __aicore__ inline void Compute1(uint32_t i, float& partialSum)
    {
        AscendC::LocalTensor<float> predLocal = predQueue.DeQue<float>();
        AscendC::LocalTensor<float> targetLocal = targetQueue.DeQue<float>();
        AscendC::LocalTensor<float> diffLocal = diffBuf.Get<float>();
        AscendC::LocalTensor<float> sqLocal = sqBuf.Get<float>();
        AscendC::LocalTensor<float> sharedLocal = sharedBuf.Get<float>();

        // diff = pred - target
        AscendC::Sub(diffLocal, predLocal, targetLocal, this->tileSize);
        
        // sq = diff * diff
        AscendC::Mul(sqLocal, diffLocal, diffLocal, this->tileSize);
        
        // Reduce sum within tile
        AscendC::ReduceSum(sqLocal, sqLocal, sharedLocal, this->tileSize);
        float tileSum = sqLocal.GetValue(0);
        
        // Accumulate
        partialSum = partialSum + tileSum;

        predQueue.FreeTensor(predLocal);
        targetQueue.FreeTensor(targetLocal);
    }

    __aicore__ inline void CopyOut1(float partialSum)
    {
        AscendC::LocalTensor<float> uploadWorkspaceLocal = workspaceOutQueue.AllocTensor<float>();
        
        uploadWorkspaceLocal.SetValue(0, partialSum);
        AscendC::DataCopy(workspaceGm[programId], uploadWorkspaceLocal, 1);
        
        workspaceOutQueue.FreeTensor(uploadWorkspaceLocal);
    }

    __aicore__ inline void CopyIn2()
    {
        AscendC::LocalTensor<float> workspaceLocal = workspaceInQueue.AllocTensor<float>();
        
        AscendC::DataCopy(workspaceLocal, workspaceGm[0], 16);
        
        workspaceInQueue.EnQue(workspaceLocal);
    }

    __aicore__ inline void Compute2()
    {
        AscendC::LocalTensor<float> workspaceLocal = workspaceInQueue.DeQue<float>();
        AscendC::LocalTensor<float> sharedLocal = sharedBuf.Get<float>();
        AscendC::LocalTensor<float> outputLocal = outputQueue.AllocTensor<float>();

        // Reduce sum across all cores
        AscendC::ReduceSum(sharedLocal, workspaceLocal, sharedLocal, 16);
        float sumSq = sharedLocal.GetValue(0);
        
        // Compute mean - remove static_cast as it's not allowed between float and uint
        float mse = sumSq / this->totalElems;
        
        outputLocal.SetValue(0, mse);
        
        workspaceInQueue.FreeTensor(workspaceLocal);
        outputQueue.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut2()
    {
        AscendC::LocalTensor<float> outputLocal = outputQueue.DeQue<float>();
        
        AscendC::DataCopy(outputGm[0], outputLocal, 1);
        
        outputQueue.FreeTensor(outputLocal);
    }
};

extern "C" __global__ __aicore__ void mse_loss_custom(GM_ADDR predictions, GM_ADDR targets, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMseLoss op;
    op.Init(predictions, targets, y, workspace, tiling_data.totalElems, tiling_data.elemsPerCore, tiling_data.tileSize, tiling_data.innerLoops);
    op.Process();
}
```