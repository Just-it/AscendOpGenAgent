// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_kernel_helper.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

#include "avg_pool3_d_tiling.h"

#include "aclrtlaunch_avg_pool3_d_generic.h"
#include "aclrtlaunch_avg_pool3_d_reduce_d.h"
#include "aclrtlaunch_avg_pool3_d_split_c.h"
#include "aclrtlaunch_avg_pool3_d_split_w.h"
#include "aclrtlaunch_avg_pool3_d_multi_w.h"

namespace ascend_kernel {

namespace {

constexpr int32_t SPLIT_MODE_C = 1;
constexpr int32_t SPLIT_MODE_W = 2;
constexpr int32_t SPLIT_MODE_MULTI_W = 3;

constexpr int32_t IMPL_GENERIC = 0;
constexpr int32_t IMPL_REDUCE_D = 1;
constexpr int32_t IMPL_SPLIT_C = 2;
constexpr int32_t IMPL_SPLIT_W = 3;
constexpr int32_t IMPL_MULTI_W = 4;

inline int32_t CeilDivI32(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

int64_t CeilDiv(int64_t value, int64_t factor)
{
    if (factor == 0) {
        return 0;
    }
    if (value % factor == 0) {
        return value / factor;
    }
    return value / factor + 1;
}

c10::SmallVector<int64_t, 3> avg_pool3d_output_size(const at::Tensor &self, c10::IntArrayRef kernel_size,
                                                    c10::IntArrayRef stride, c10::IntArrayRef padding,
                                                    bool ceil_mode)
{
    int self_d = self.size(-3);
    int self_h = self.size(-2);
    int self_w = self.size(-1);

    int64_t kernel_d = ceil_mode ? (CeilDiv(self_d + 2 * padding[0] - kernel_size[0], stride[0]) + 1)
                                 : ((self_d + 2 * padding[0] - kernel_size[0]) / stride[0] + 1);
    int64_t kernel_h = ceil_mode ? (CeilDiv(self_h + 2 * padding[1] - kernel_size[1], stride[1]) + 1)
                                 : ((self_h + 2 * padding[1] - kernel_size[1]) / stride[1] + 1);
    int64_t kernel_w = ceil_mode ? (CeilDiv(self_w + 2 * padding[2] - kernel_size[2], stride[2]) + 1)
                                 : ((self_w + 2 * padding[2] - kernel_size[2]) / stride[2] + 1);

    if (ceil_mode) {
        if ((kernel_d - 1) * stride[0] >= self_d + padding[0]) {
            --kernel_d;
        }
        if ((kernel_h - 1) * stride[1] >= self_h + padding[1]) {
            --kernel_h;
        }
        if ((kernel_w - 1) * stride[2] >= self_w + padding[2]) {
            --kernel_w;
        }
    }

    c10::SmallVector<int64_t, 3> output_size;
    if (self.dim() == 4) {
        output_size = {self.size(0), kernel_d, kernel_h, kernel_w};
    } else {
        output_size = {self.size(0), self.size(1), kernel_d, kernel_h, kernel_w};
    }
    return output_size;
}

int32_t ChooseBlockM(int32_t mOut)
{
    for (int32_t candidate : {64, 32, 16, 8, 4, 2}) {
        if (candidate <= mOut && mOut % candidate == 0) {
            return candidate;
        }
    }
    TORCH_CHECK(false, "Unsupported output spatial size: M_out=", mOut);
}

int32_t ChooseBlockC(int32_t c)
{
    for (int32_t candidate : {256, 128, 64, 32}) {
        if (candidate <= c && c % candidate == 0) {
            return candidate;
        }
    }
    return 0;
}

int32_t ResolveImplMode(int32_t splitMode, int32_t kH, int32_t kW, int32_t sH, int32_t sW,
                        int32_t pH, int32_t pW, int32_t c)
{
    if (kH == 1 && kW == 1 && sH == 1 && sW == 1 && pH == 0 && pW == 0) {
        return IMPL_REDUCE_D;
    }
    if (splitMode == SPLIT_MODE_C) {
        int32_t blockC = ChooseBlockC(c);
        if (blockC > 0) {
            return IMPL_SPLIT_C;
        }
        TORCH_CHECK(false, "split_c scenario requires divisible channel tiles, got C=", c);
    }
    if (splitMode == SPLIT_MODE_W) {
        return IMPL_SPLIT_W;
    }
    if (splitMode == SPLIT_MODE_MULTI_W) {
        return IMPL_MULTI_W;
    }
    return IMPL_GENERIC;
}

} // namespace

at::Tensor avg_pool3d(const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                      at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                      c10::optional<int64_t> divisor_override)
{
    TORCH_CHECK(self.dim() == 4 || self.dim() == 5, "avg_pool3d: input must be 4D or 5D");
    TORCH_CHECK(!kernel_size.empty(), "avg_pool3d: kernel_size must not be empty");

    int64_t k_d = kernel_size[0];
    int64_t k_h = kernel_size.size() == 1 ? k_d : kernel_size[1];
    int64_t k_w = kernel_size.size() == 1 ? k_d : kernel_size[2];

    int64_t s_d = stride.empty() ? k_d : stride[0];
    int64_t s_h = stride.empty() ? k_h : stride.size() == 1 ? s_d : stride[1];
    int64_t s_w = stride.empty() ? k_w : stride.size() == 1 ? s_d : stride[2];
    TORCH_CHECK(s_d != 0 && s_h != 0 && s_w != 0, "avg_pool3d: stride should not contain zero");

    int64_t p_d = padding[0];
    int64_t p_h = padding.size() == 1 ? p_d : padding[1];
    int64_t p_w = padding.size() == 1 ? p_d : padding[2];

    c10::SmallVector<int64_t, 3> kernel_sizes = {k_d, k_h, k_w};
    c10::SmallVector<int64_t, 3> stride_sizes = {s_d, s_h, s_w};
    c10::SmallVector<int64_t, 3> padding_sizes = {p_d, p_h, p_w};

    auto output_size = avg_pool3d_output_size(self, at::IntArrayRef(kernel_sizes),
                                               at::IntArrayRef(stride_sizes),
                                               at::IntArrayRef(padding_sizes), ceil_mode);
    at::Tensor result = at_npu::native::empty_with_format(
        output_size,
        self.options(),
        at_npu::native::get_npu_format(self));

    int64_t n = self.size(0);
    int64_t c = (self.dim() == 5) ? self.size(1) : 1;
    int64_t d = self.size(-3);
    int64_t h = self.size(-2);
    int64_t w = self.size(-1);

    int64_t o_d = output_size[self.dim() == 5 ? 2 : 1];
    int64_t o_h = output_size[self.dim() == 5 ? 3 : 2];
    int64_t o_w = output_size[self.dim() == 5 ? 4 : 3];

    int32_t k_d32 = static_cast<int32_t>(k_d);
    int32_t k_h32 = static_cast<int32_t>(k_h);
    int32_t k_w32 = static_cast<int32_t>(k_w);

    int32_t s_d32 = static_cast<int32_t>(s_d);
    int32_t s_h32 = static_cast<int32_t>(s_h);
    int32_t s_w32 = static_cast<int32_t>(s_w);

    int32_t p_d32 = static_cast<int32_t>(p_d);
    int32_t p_h32 = static_cast<int32_t>(p_h);
    int32_t p_w32 = static_cast<int32_t>(p_w);

    int32_t countIncludePad = count_include_pad ? 1 : 0;
    int32_t divisorOverride = divisor_override.has_value() ? static_cast<int32_t>(divisor_override.value()) : 0;

    // Prepare input in NHWC flat format: (n*d*h*w, c)
    at::Tensor x_input;
    if (self.dim() == 5) {
        x_input = self.permute({0, 2, 3, 4, 1}).contiguous();
    } else {
        x_input = self.permute({0, 2, 3, 1}).contiguous();
    }
    at::Tensor x_flat = x_input.reshape({n * d * h * w, c});

    TORCH_CHECK(x_flat.is_contiguous(), "avg_pool3d: x_flat must be contiguous");
    TORCH_CHECK(x_flat.scalar_type() == at::kFloat, "avg_pool3d: only float32 is supported");

    int32_t n32 = static_cast<int32_t>(n);
    int32_t c32 = static_cast<int32_t>(c);
    int32_t d32 = static_cast<int32_t>(d);
    int32_t h32 = static_cast<int32_t>(h);
    int32_t w32 = static_cast<int32_t>(w);
    int32_t od32 = static_cast<int32_t>(o_d);
    int32_t oh32 = static_cast<int32_t>(o_h);
    int32_t ow32 = static_cast<int32_t>(o_w);

    int32_t inSpatial = d32 * h32 * w32;
    int32_t outSpatial = od32 * oh32 * ow32;
    int32_t mOut = n32 * outSpatial;

    int32_t blockM = ChooseBlockM(mOut);
    int32_t subBlockM = blockM / DEFAULT_VEC_NUM;
    int32_t mNum = mOut / blockM;
    int32_t usedCoreNum = std::min(DEFAULT_NUM_PHYSICAL_CORES, mNum);
    int32_t tasksPerCore = CeilDivI32(mNum, usedCoreNum);

    int32_t splitMode = 0;
    int32_t blockC = 0;
    int32_t splitWTileKw = 0;
    int32_t multiWWindow = 1;

    int32_t implMode = ResolveImplMode(splitMode, k_h32, k_w32, s_h32, s_w32, p_h32, p_w32, c32);

    at::Tensor y_flat = at::empty({mOut, c32}, at::device(at::kPrivateUse1).dtype(at::kFloat));

    at::Tensor tilingCpu = at::empty({static_cast<long>(sizeof(AvgPool3DKernelTiling))},
                                      at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<AvgPool3DKernelTiling *>(tilingCpu.data_ptr());

    tiling->N = n32;
    tiling->C = c32;
    tiling->D = d32;
    tiling->H = h32;
    tiling->W = w32;

    tiling->OD = od32;
    tiling->OH = oh32;
    tiling->OW = ow32;

    tiling->kD = k_d32;
    tiling->kH = k_h32;
    tiling->kW = k_w32;

    tiling->sD = s_d32;
    tiling->sH = s_h32;
    tiling->sW = s_w32;

    tiling->pD = p_d32;
    tiling->pH = p_h32;
    tiling->pW = p_w32;

    tiling->countIncludePad = countIncludePad;
    tiling->divisorOverride = divisorOverride;

    tiling->splitMode = splitMode;
    tiling->blockC = blockC;
    tiling->splitWTileKw = splitWTileKw;
    tiling->multiWWindow = multiWWindow;

    tiling->blockM = blockM;
    tiling->subBlockM = subBlockM;
    tiling->mNum = mNum;
    tiling->outSpatial = outSpatial;
    tiling->inSpatial = inSpatial;
    tiling->hw = h32 * w32;

    tiling->cNum = 1;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;

    tiling->vectorLen = std::min<int32_t>(c32, 256);
    tiling->reserved0 = implMode;
    tiling->reserved1 = 0;

    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    tiling->launchBlocks = static_cast<int32_t>(blockDim);

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    switch (implMode) {
        case IMPL_REDUCE_D:
            EXEC_KERNEL_CMD(avg_pool3_d_reduce_d, blockDim, x_flat, y_flat, tilingNpu);
            break;
        case IMPL_SPLIT_C:
            EXEC_KERNEL_CMD(avg_pool3_d_split_c, blockDim, x_flat, y_flat, tilingNpu);
            break;
        case IMPL_SPLIT_W:
            EXEC_KERNEL_CMD(avg_pool3_d_split_w, blockDim, x_flat, y_flat, tilingNpu);
            break;
        case IMPL_MULTI_W:
            EXEC_KERNEL_CMD(avg_pool3_d_multi_w, blockDim, x_flat, y_flat, tilingNpu);
            break;
        case IMPL_GENERIC:
        default:
            EXEC_KERNEL_CMD(avg_pool3_d_generic, blockDim, x_flat, y_flat, tilingNpu);
            break;
    }

    // Reshape output back to NCDHW/NDHWC and then permute
    at::Tensor y_ndhwc = y_flat.reshape({n, o_d, o_h, o_w, c});
    at::Tensor y_ncdhw = y_ndhwc.permute({0, 4, 1, 2, 3}).contiguous();

    // For 4D input, remove the channel dimension
    if (self.dim() == 4) {
        y_ncdhw = y_ncdhw.squeeze(1).contiguous();
    }

    // Copy to result tensor if needed
    if (!y_ncdhw.is_same(result)) {
        result.copy_(y_ncdhw);
    }

    return result;
}

} // namespace ascend_kernel
