#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("matmul_leakyrelu(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("matmul_leakyrelu", TORCH_FN(ascend_kernel::matmul_leakyrelu));
}

}  // namespace
