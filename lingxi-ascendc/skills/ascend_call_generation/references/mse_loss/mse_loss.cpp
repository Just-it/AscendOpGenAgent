#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor mse_loss_custom_impl_npu(const at::Tensor& predictions, const at::Tensor& targets) {
    at::Tensor result = at::empty({}, predictions.options());
    EXEC_NPU_CMD(aclnnMseLossCustom, predictions, targets, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("mse_loss_custom", &mse_loss_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mse_loss_custom", &mse_loss_custom_impl_npu, "Mean Squared Error Loss");
}