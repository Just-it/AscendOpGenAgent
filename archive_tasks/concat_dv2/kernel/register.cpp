// Copyright (c) 2026 Huawei Technologies Co., Ltd
// Licensed under the BSD 3-Clause License

#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("concat_dim0_1(Tensor x0) -> Tensor");
    m.def("concat_dim0_2(Tensor x0, Tensor x1) -> Tensor");
    m.def("concat_dim0_3(Tensor x0, Tensor x1, Tensor x2) -> Tensor");
    m.def("concat_dim0_4(Tensor x0, Tensor x1, Tensor x2, Tensor x3) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("concat_dim0_1", TORCH_FN(ascend_kernel::concat_dim0_1));
    m.impl("concat_dim0_2", TORCH_FN(ascend_kernel::concat_dim0_2));
    m.impl("concat_dim0_3", TORCH_FN(ascend_kernel::concat_dim0_3));
    m.impl("concat_dim0_4", TORCH_FN(ascend_kernel::concat_dim0_4));
}

} // namespace
