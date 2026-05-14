#ifndef CURRENT_TASK_CONCAT_DIM0_TILING_H
#define CURRENT_TASK_CONCAT_DIM0_TILING_H

#include <cstdint>

struct ConcatDim0Tiling {
    int32_t M0;
    int32_t M1;
    int32_t M2;
    int32_t M3;
    int32_t inputCount;
    int32_t N;
    int32_t totalM;
    int32_t blockM;
    int32_t subBlockM;
    int32_t blockNum;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
};

#endif
