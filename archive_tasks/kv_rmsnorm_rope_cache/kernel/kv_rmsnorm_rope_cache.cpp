#include "kernel_operator.h"
#include "kv_rmsnorm_rope_cache_kernel.h"
#include "kv_rmsnorm_rope_cache_tiling.h"

extern "C" __global__ __aicore__ void kv_rmsnorm_rope_cache_custom_fp16(
    GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR index, GM_ADDR k_cache, GM_ADDR ckv_cache,
    GM_ADDR k_cache_out, GM_ADDR ckv_cache_out,
    GM_ADDR k_embed_out, GM_ADDR v_out,
    GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KvRmsnormRopeCacheKernel<half> kernel;
    kernel.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                k_cache_out, ckv_cache_out, k_embed_out, v_out,
                tiling, &pipe);
    kernel.Process();
}

extern "C" void kv_rmsnorm_rope_cache_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *kv, uint8_t *gamma, uint8_t *cos, uint8_t *sin,
    uint8_t *index, uint8_t *k_cache, uint8_t *ckv_cache,
    uint8_t *k_cache_out, uint8_t *ckv_cache_out,
    uint8_t *k_embed_out, uint8_t *v_out,
    uint8_t *tiling)
{
    kv_rmsnorm_rope_cache_custom_fp16<<<blockDim, nullptr, stream>>>(
        kv, gamma, cos, sin, index, k_cache, ckv_cache,
        k_cache_out, ckv_cache_out, k_embed_out, v_out, tiling);
}

extern "C" __global__ __aicore__ void kv_rmsnorm_rope_cache_custom_bf16(
    GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR index, GM_ADDR k_cache, GM_ADDR ckv_cache,
    GM_ADDR k_cache_out, GM_ADDR ckv_cache_out,
    GM_ADDR k_embed_out, GM_ADDR v_out,
    GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KvRmsnormRopeCacheKernel<bfloat16_t> kernel;
    kernel.Init(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                k_cache_out, ckv_cache_out, k_embed_out, v_out,
                tiling, &pipe);
    kernel.Process();
}

extern "C" void kv_rmsnorm_rope_cache_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *kv, uint8_t *gamma, uint8_t *cos, uint8_t *sin,
    uint8_t *index, uint8_t *k_cache, uint8_t *ckv_cache,
    uint8_t *k_cache_out, uint8_t *ckv_cache_out,
    uint8_t *k_embed_out, uint8_t *v_out,
    uint8_t *tiling)
{
    kv_rmsnorm_rope_cache_custom_bf16<<<blockDim, nullptr, stream>>>(
        kv, gamma, cos, sin, index, k_cache, ckv_cache,
        k_cache_out, ckv_cache_out, k_embed_out, v_out, tiling);
}
