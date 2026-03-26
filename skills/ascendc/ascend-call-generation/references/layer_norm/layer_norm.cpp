#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor layer_norm_custom_impl_npu(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, double eps) {
    // float argument not supported now, so use double negative_slope
    at::Tensor result = at::empty_like(input);
    EXEC_NPU_CMD(aclnnLayerNormCustom, input, weight, bias, eps, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("layer_norm_custom", &layer_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm_custom", &layer_norm_custom_impl_npu, "Layer Normalization");
}