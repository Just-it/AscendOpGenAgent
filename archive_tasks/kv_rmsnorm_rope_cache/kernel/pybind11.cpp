#include <algorithm>
#include <string>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "kv_rmsnorm_rope_cache_tiling.h"

extern "C" void kv_rmsnorm_rope_cache_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *kv, uint8_t *gamma, uint8_t *cos, uint8_t *sin,
    uint8_t *index, uint8_t *k_cache, uint8_t *ckv_cache,
    uint8_t *k_cache_out, uint8_t *ckv_cache_out,
    uint8_t *k_embed_out, uint8_t *v_out,
    uint8_t *tiling);

extern "C" void kv_rmsnorm_rope_cache_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *kv, uint8_t *gamma, uint8_t *cos, uint8_t *sin,
    uint8_t *index, uint8_t *k_cache, uint8_t *ckv_cache,
    uint8_t *k_cache_out, uint8_t *ckv_cache_out,
    uint8_t *k_embed_out, uint8_t *v_out,
    uint8_t *tiling);

namespace kv_rmsnorm_rope_cache_ext {

using LaunchFn = void (*)(uint32_t, void *,
    uint8_t *, uint8_t *, uint8_t *, uint8_t *,
    uint8_t *, uint8_t *, uint8_t *,
    uint8_t *, uint8_t *, uint8_t *, uint8_t *,
    uint8_t *);

int GetCacheModeId(const std::string &mode)
{
    if (mode == "Norm") return 0;
    if (mode == "PA") return 1;
    if (mode == "PA_BNSD") return 2;
    if (mode == "PA_NZ") return 3;
    if (mode == "PA_BLK_BNSD") return 4;
    if (mode == "PA_BLK_NZ") return 5;
    return 0;
}

pybind11::tuple run_kv_rmsnorm_rope_cache(
    const at::Tensor &kv, const at::Tensor &gamma,
    const at::Tensor &cos, const at::Tensor &sin,
    const at::Tensor &index, const at::Tensor &k_cache, const at::Tensor &ckv_cache,
    double epsilon, const std::string &cache_mode, bool is_output_kv)
{
    TORCH_CHECK(kv.dim() == 4, "kv must be 4D [B, N, S, hidden_size]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(cos.dim() == 4, "cos must be 4D");
    TORCH_CHECK(sin.dim() == 4, "sin must be 4D");
    TORCH_CHECK(index.dtype() == at::kLong, "index must be int64");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be 4D");
    TORCH_CHECK(ckv_cache.dim() == 4, "ckv_cache must be 4D");
    TORCH_CHECK(kv.is_contiguous(), "kv must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(cos.is_contiguous(), "cos must be contiguous");
    TORCH_CHECK(sin.is_contiguous(), "sin must be contiguous");
    TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
    TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");
    TORCH_CHECK(ckv_cache.is_contiguous(), "ckv_cache must be contiguous");

    const int32_t B = static_cast<int32_t>(kv.sizes()[0]);
    const int32_t N = static_cast<int32_t>(kv.sizes()[1]);
    const int32_t S = static_cast<int32_t>(kv.sizes()[2]);
    const int32_t hidden_size = static_cast<int32_t>(kv.sizes()[3]);
    const int32_t rms_size = static_cast<int32_t>(gamma.sizes()[0]);
    const int32_t rope_size = hidden_size - rms_size;
    const int32_t total = B * S * N;

    // Rearrange kv, cos, sin from BNSD to BSND on host
    at::Tensor kv_bsnd = kv.permute({0, 2, 1, 3}).contiguous();
    at::Tensor cos_bsnd = cos.permute({0, 2, 1, 3}).contiguous();
    at::Tensor sin_bsnd = sin.permute({0, 2, 1, 3}).contiguous();

    // Extract rms_in and rope_in, then interleave rope_in
    at::Tensor rms_in = kv_bsnd.narrow(3, 0, rms_size).contiguous();
    at::Tensor rope_in = kv_bsnd.narrow(3, rms_size, rope_size).contiguous();
    at::Tensor k_input = rope_in.reshape({B, S, N, rope_size / 2, 2})
                            .permute({0, 1, 2, 4, 3})
                            .reshape({B, S, N, rope_size})
                            .contiguous();

    at::Tensor kv_input = at::cat({rms_in.reshape({total, rms_size}),
                                   k_input.reshape({total, rope_size})}, 1).contiguous();

    at::Tensor k_cache_out = k_cache.clone();
    at::Tensor ckv_cache_out = ckv_cache.clone();
    at::Tensor k_embed_out = at::empty({B, N, S, rope_size}, kv.options());
    at::Tensor v_out = at::empty({B, N, S, rms_size}, kv.options());

    const int32_t mNum = (total + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, mNum);
    const int32_t tasksPerCore = (mNum + usedCoreNum - 1) / usedCoreNum;

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(KvRmsnormRopeCacheTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<KvRmsnormRopeCacheTiling *>(tilingCpu.data_ptr());
    tiling->B = B;
    tiling->N = N;
    tiling->S = S;
    tiling->rms_size = rms_size;
    tiling->rope_size = rope_size;
    tiling->hidden_size = hidden_size;
    tiling->total = total;
    tiling->block_size = DEFAULT_BLOCK_SIZE;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->eps = static_cast<float>(epsilon);
    tiling->invRmsSize = 1.0f / static_cast<float>(rms_size);
    tiling->cache_mode = GetCacheModeId(cache_mode);
    tiling->is_output_kv = is_output_kv ? 1 : 0;
    tiling->k_cache_dim0 = static_cast<int32_t>(k_cache.sizes()[0]);
    tiling->k_cache_dim1 = static_cast<int32_t>(k_cache.sizes()[1]);
    tiling->k_cache_dim2 = static_cast<int32_t>(k_cache.sizes()[2]);
    tiling->k_cache_dim3 = static_cast<int32_t>(k_cache.sizes()[3]);
    tiling->ckv_cache_dim0 = static_cast<int32_t>(ckv_cache.sizes()[0]);
    tiling->ckv_cache_dim1 = static_cast<int32_t>(ckv_cache.sizes()[1]);
    tiling->ckv_cache_dim2 = static_cast<int32_t>(ckv_cache.sizes()[2]);
    tiling->ckv_cache_dim3 = static_cast<int32_t>(ckv_cache.sizes()[3]);
    tiling->index_numel = static_cast<int32_t>(index.numel());
    tiling->quant_enabled = 0;

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;
    if (kv.scalar_type() == at::kHalf) {
        launch = kv_rmsnorm_rope_cache_do_fp16;
    } else if (kv.scalar_type() == at::kBFloat16) {
        launch = kv_rmsnorm_rope_cache_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype, only float16 and bfloat16 are supported");
    }

    launch(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t *>(kv_input.data_ptr()),
        static_cast<uint8_t *>(gamma.data_ptr()),
        static_cast<uint8_t *>(cos_bsnd.data_ptr()),
        static_cast<uint8_t *>(sin_bsnd.data_ptr()),
        static_cast<uint8_t *>(index.data_ptr()),
        static_cast<uint8_t *>(k_cache.data_ptr()),
        static_cast<uint8_t *>(ckv_cache.data_ptr()),
        static_cast<uint8_t *>(k_cache_out.data_ptr()),
        static_cast<uint8_t *>(ckv_cache_out.data_ptr()),
        static_cast<uint8_t *>(k_embed_out.data_ptr()),
        static_cast<uint8_t *>(v_out.data_ptr()),
        static_cast<uint8_t *>(tilingNpu.data_ptr()));

    if (is_output_kv) {
        return pybind11::make_tuple(k_cache_out, ckv_cache_out, k_embed_out, v_out);
    }
    return pybind11::make_tuple(k_cache_out, ckv_cache_out, pybind11::none(), pybind11::none());
}

}  // namespace kv_rmsnorm_rope_cache_ext

PYBIND11_MODULE(_kv_rmsnorm_rope_cache_ext, m)
{
    m.doc() = "kv_rmsnorm_rope_cache extension";
    m.def("run_kv_rmsnorm_rope_cache", &kv_rmsnorm_rope_cache_ext::run_kv_rmsnorm_rope_cache, "");
}
