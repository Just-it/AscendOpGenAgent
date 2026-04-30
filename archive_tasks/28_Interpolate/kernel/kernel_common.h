#ifndef INTERP_KERNEL_COMMON_H
#define INTERP_KERNEL_COMMON_H

#include "kernel_operator.h"

template <typename T>
__aicore__ inline int32_t CeilDivT(T a, T b) {
    return static_cast<int32_t>((a + b - 1) / b);
}

template <typename T>
__aicore__ inline T MinT(T a, T b) { return a < b ? a : b; }

template <typename T>
__aicore__ inline T MaxT(T a, T b) { return a > b ? a : b; }

#endif  // INTERP_KERNEL_COMMON_H
