#ifndef INTERPOLATE_TILING_H
#define INTERPOLATE_TILING_H

#include <cstdint>

constexpr int32_t INTERP_NUM_PHYSICAL_CORES = 20;
constexpr int32_t INTERP_K_MAX = 8;  // max K_h or K_w (covers up to 8x area downsample)

struct InterpolateTiling {
    int32_t NC;       // = N * C
    int32_t H_in;
    int32_t W_in;
    int32_t H_out;
    int32_t W_out;
    int32_t K_h;      // K_h ≤ INTERP_K_MAX
    int32_t K_w;      // K_w ≤ INTERP_K_MAX
    int32_t usedCoreNum;
    int32_t tasksPerCore;   // tasks per core, where one task = one (nc, h_out) pair
    int32_t totalTasks;     // = NC * H_out
    int32_t reserved0;
};

#endif  // INTERPOLATE_TILING_H
