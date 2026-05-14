#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor gather_elements_v2(
    const at::Tensor &x,
    const at::Tensor &index,
    const at::Tensor &row_map,
    int64_t ig,
    int64_t mode);

} // namespace ascend_kernel

#endif // OPS_H
