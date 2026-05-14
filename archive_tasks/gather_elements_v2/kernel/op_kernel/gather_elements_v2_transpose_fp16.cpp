#include "gather_elements_v2_transpose_kernel.h"

extern "C" __global__ __aicore__ void gather_elements_v2_transpose_fp16(
    GM_ADDR x,
    GM_ADDR index,
    GM_ADDR rowMap,
    GM_ADDR y,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    GatherElementsV2TransposeKernel<half> kernel;
    kernel.Init(x, index, rowMap, y, tiling, &pipe);
    kernel.Process();
}

