#include <algorithm>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "interpolate_tiling.h"

extern "C" void interpolate_unified_fp32_do(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *h_idx, uint8_t *w_idx,
    uint8_t *h_w, uint8_t *w_w, uint8_t *y, uint8_t *tiling);

extern "C" void interpolate_unified_fp16_do(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *h_idx, uint8_t *w_idx,
    uint8_t *h_w, uint8_t *w_w, uint8_t *y, uint8_t *tiling);

extern "C" void interpolate_unified_bf16_do(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *h_idx, uint8_t *w_idx,
    uint8_t *h_w, uint8_t *w_w, uint8_t *y, uint8_t *tiling);

namespace interpolate_ext {

using LaunchFn = void (*)(uint32_t, void *,
                          uint8_t *, uint8_t *, uint8_t *,
                          uint8_t *, uint8_t *, uint8_t *, uint8_t *);

inline int32_t CeilDivI32(int32_t a, int32_t b) { return (a + b - 1) / b; }

at::Tensor run_interpolate(
    const at::Tensor &xFlat,        // [NC, H_in, W_in], dtype = T_IN
    const at::Tensor &h_idx,        // [H_out, K_h] int32
    const at::Tensor &w_idx,        // [W_out, K_w] int32
    const at::Tensor &h_w,          // [H_out, K_h] fp32
    const at::Tensor &w_w,          // [W_out, K_w] fp32
    int64_t NC, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int64_t K_h, int64_t K_w)
{
    TORCH_CHECK(xFlat.dim() == 3, "xFlat must be [NC, H_in, W_in]");
    TORCH_CHECK(xFlat.is_contiguous(), "xFlat must be contiguous");
    TORCH_CHECK(h_idx.is_contiguous() && w_idx.is_contiguous(),
                "idx tensors must be contiguous");
    TORCH_CHECK(h_w.is_contiguous() && w_w.is_contiguous(),
                "weight tensors must be contiguous");
    TORCH_CHECK(h_idx.scalar_type() == at::kInt &&
                w_idx.scalar_type() == at::kInt,
                "h_idx/w_idx must be int32");
    TORCH_CHECK(h_w.scalar_type() == at::kFloat &&
                w_w.scalar_type() == at::kFloat,
                "h_w/w_w must be float32");

    const int32_t NC32    = static_cast<int32_t>(NC);
    const int32_t H_in32  = static_cast<int32_t>(H_in);
    const int32_t W_in32  = static_cast<int32_t>(W_in);
    const int32_t H_out32 = static_cast<int32_t>(H_out);
    const int32_t W_out32 = static_cast<int32_t>(W_out);
    const int32_t K_h32   = static_cast<int32_t>(K_h);
    const int32_t K_w32   = static_cast<int32_t>(K_w);

    const int32_t totalTasks = NC32 * H_out32;
    const int32_t usedCoreNum = std::min<int32_t>(
        INTERP_NUM_PHYSICAL_CORES, totalTasks);
    const int32_t tasksPerCore = CeilDivI32(totalTasks, usedCoreNum);

    auto opts = xFlat.options();
    at::Tensor yFlat = at::empty({NC32, H_out32, W_out32}, opts);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(InterpolateTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *t = reinterpret_cast<InterpolateTiling *>(tilingCpu.data_ptr());
    t->NC          = NC32;
    t->H_in        = H_in32;
    t->W_in        = W_in32;
    t->H_out       = H_out32;
    t->W_out       = W_out32;
    t->K_h         = K_h32;
    t->K_w         = K_w32;
    t->usedCoreNum = usedCoreNum;
    t->tasksPerCore= tasksPerCore;
    t->totalTasks  = totalTasks;
    t->reserved0   = 0;

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    LaunchFn launch = nullptr;
    auto dt = xFlat.scalar_type();
    if (dt == at::kFloat)        launch = interpolate_unified_fp32_do;
    else if (dt == at::kHalf)    launch = interpolate_unified_fp16_do;
    else if (dt == at::kBFloat16) launch = interpolate_unified_bf16_do;
    else TORCH_CHECK(false, "Unsupported dtype for interpolate kernel");

    launch(
        static_cast<uint32_t>(usedCoreNum),
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(xFlat.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(h_idx.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(w_idx.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(h_w.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(w_w.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(yFlat.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));

    return yFlat;
}

} // namespace interpolate_ext

PYBIND11_MODULE(_interpolate_ext, m)
{
    m.doc() = "28_Interpolate AscendC extension";
    m.def("run_interpolate", &interpolate_ext::run_interpolate, "");
}
