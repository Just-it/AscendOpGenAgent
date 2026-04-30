#ifndef INTERPOLATE_TILING_H
#define INTERPOLATE_TILING_H

#include <cstdint>

constexpr int32_t INTERP_NUM_PHYSICAL_CORES = 20;
constexpr int32_t INTERP_K_MAX = 8;  // max K_h or K_w (covers up to 8x area downsample)

struct InterpolateTiling {
    int32_t NC;
    int32_t H_in;
    int32_t W_in;
    int32_t H_out;
    int32_t W_out;
    int32_t K_h;
    int32_t K_w;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t totalTasks;
    // 1 → kernel computes bicubic weights from t_h/t_w in NPU fp32 (matches
    //     PyTorch NPU's fp32 polynomial rounding); ignores precomputed h_w/w_w.
    // 0 → use precomputed h_w/w_w from host (default, used for nearest /
    //     bilinear / area).
    int32_t bicubic_in_kernel;
};

#endif  // INTERPOLATE_TILING_H
