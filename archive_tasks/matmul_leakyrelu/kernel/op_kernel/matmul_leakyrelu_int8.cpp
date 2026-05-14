/**
 * @file matmul_leakyrelu_int8.cpp
 *
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "matmul_leakyrelu.h"

extern "C" __global__ __aicore__ void matmul_leakyrelu_int8(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    MatmulLeakyKernel<int8_t, int8_t, int32_t, float> matmulLeakyKernel;
    matmulLeakyKernel.Init(a, b, c, workspace, tiling, &pipe);
    matmulLeakyKernel.Process();
}

