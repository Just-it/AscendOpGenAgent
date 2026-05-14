#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor concat_dim0_1(const at::Tensor &x0);
at::Tensor concat_dim0_2(const at::Tensor &x0, const at::Tensor &x1);
at::Tensor concat_dim0_3(const at::Tensor &x0, const at::Tensor &x1, const at::Tensor &x2);
at::Tensor concat_dim0_4(const at::Tensor &x0, const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &x3);

} // namespace ascend_kernel

#endif // OPS_H
