
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

// 为NPU设备注册前向实现
at::Tensor elementwise_sub_impl_npu(const at::Tensor& self, const at::Tensor& other) {
    // 创建输出内存
    at::Tensor result = at::empty_like(self);

    // 调用aclnn接口计算
    EXEC_NPU_CMD(aclnnElementwiseSub, self, other, result);
    return result;
}


// 为NPU设备注册前反向实现
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("elementwise_sub", &elementwise_sub_impl_npu);
}

// // 通过pybind将c++接口和python接口绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elementwise_sub", &elementwise_sub_impl_npu, "x - y");
}
