#ifndef KV_RMSNORM_ROPE_CACHE_TILING_H
#define KV_RMSNORM_ROPE_CACHE_TILING_H

#include <cstdint>

constexpr uint32_t DEFAULT_BLOCK_SIZE = 64;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct KvRmsnormRopeCacheTiling {
    int32_t B;
    int32_t N;
    int32_t S;
    int32_t rms_size;
    int32_t rope_size;
    int32_t hidden_size;
    int32_t total;
    int32_t block_size;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    float eps;
    float invRmsSize;
    int32_t cache_mode;      // 0=Norm, 1=PA, 2=PA_BNSD, 3=PA_NZ, 4=PA_BLK_BNSD, 5=PA_BLK_NZ
    int32_t is_output_kv;
    int32_t k_cache_dim0;
    int32_t k_cache_dim1;
    int32_t k_cache_dim2;
    int32_t k_cache_dim3;
    int32_t ckv_cache_dim0;
    int32_t ckv_cache_dim1;
    int32_t ckv_cache_dim2;
    int32_t ckv_cache_dim3;
    int32_t index_numel;
    int32_t quant_enabled;   // 0=no quant, 1=quant
};

#endif
