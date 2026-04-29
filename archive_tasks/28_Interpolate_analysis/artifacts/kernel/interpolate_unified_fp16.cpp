#include "kernel_operator.h"
#include "interpolate_unified_kernel.h"

extern "C" __global__ __aicore__ void interpolate_unified_fp16(
    GM_ADDR x, GM_ADDR h_idx, GM_ADDR w_idx, GM_ADDR h_w, GM_ADDR w_w,
    GM_ADDR y, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    InterpolateUnifiedKernel<half> kernel;
    kernel.Init(x, h_idx, w_idx, h_w, w_w, y, tiling, &pipe);
    kernel.Process();
}

extern "C" void interpolate_unified_fp16_do(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *h_idx, uint8_t *w_idx,
    uint8_t *h_w, uint8_t *w_w, uint8_t *y, uint8_t *tiling)
{
    interpolate_unified_fp16<<<blockDim, nullptr, stream>>>(
        x, h_idx, w_idx, h_w, w_w, y, tiling);
}
