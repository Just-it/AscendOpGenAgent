#include "concat_dim0_4_kernel.h"

extern "C" __global__ __aicore__ void concat_dim0_4(
    GM_ADDR x0,
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR x3,
    GM_ADDR y,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    ConcatDim0_4Kernel kernel;
    kernel.Init(x0, x1, x2, x3, y, tiling, &pipe);
    kernel.Process();
}
