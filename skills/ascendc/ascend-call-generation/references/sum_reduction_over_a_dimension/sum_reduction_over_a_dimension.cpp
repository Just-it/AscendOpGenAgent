#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor sum_reduction_over_a_dimension_custom_impl_npu(const at::Tensor& input, int64_t dim) {
    int64_t ndim = input.dim();
    int64_t adjusted_dim = dim;
    if (dim < 0) {
        adjusted_dim = dim + ndim;
    }

    auto input_shape = input.sizes().vec();
    auto output_shape = input_shape;
    output_shape[adjusted_dim] = 1;
    
    at::Tensor result = at::empty(output_shape, input.options());
    EXEC_NPU_CMD(aclnnSumReductionOverADimensionCustom, input, adjusted_dim, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("sum_reduction_over_a_dimension_custom", &sum_reduction_over_a_dimension_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_reduction_over_a_dimension_custom", &sum_reduction_over_a_dimension_custom_impl_npu, "Sum Reduction Over A Dimension");
}