#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("gather_elements_v2(Tensor x, Tensor index, Tensor row_map, int ig, int mode) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("gather_elements_v2", TORCH_FN(ascend_kernel::gather_elements_v2));
}

}  // namespace
