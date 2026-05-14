#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor matmul_leakyrelu(const at::Tensor &a, const at::Tensor &b);

} // namespace ascend_kernel

#endif // OPS_H
