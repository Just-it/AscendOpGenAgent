# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test__kernel_scale_fp8_row_npu.py
# Main kernel: _kernel_scale_fp8_row
# PT file: test__kernel_scale_fp8_row_v2.pt

import triton
import triton.language as tl


# === _kernel_scale_fp8_row ===
@triton.jit
def _kernel_scale_fp8_row(
    A,
    x_scale,
    w_scale,
    scaled_out,
    M,
    N,
    stride_am,
    stride_an,
    stride_om,
    stride_on,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    Scale each row of A by x_scale and each column of A by w_scale.

    Args:
        A (Tensor): [m, n] Input tensor to scale.
        x_scale (Tensor): [m] Row-wise scale tensor.
        w_scale (Tensor): [n] Col-wise scale tensor.
        scaled_out (Tensor): [m, n] Output tensor.
        M (int): Number of rows.
        N (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_an (int): Stride of n dimension of A.
        stride_om (int): Stride of m dimension of output.
        stride_on (int): Stride of n dimension of output.
        BLOCK_SIZE (int): Block size for data loads.
    """
    pid = tl.program_id(0)
    n_offset = tl.arange(0, BLOCK_SIZE)
    # Load activation scale for this row.
    row_scale = tl.load(x_scale + pid)

    # Iterate over chunks of the row and apply scales.
    for _k in range(0, tl.cdiv(N, BLOCK_SIZE)):
        a = tl.load(
            A + pid * stride_am + n_offset * stride_an, mask=n_offset < N, other=0.0
        )
        col_scale = tl.load(w_scale + n_offset)
        scaled_a = a * row_scale * col_scale
        tl.store(
            scaled_out + pid * stride_om + n_offset * stride_on,
            scaled_a,
            mask=n_offset < N,
        )
        n_offset += BLOCK_SIZE

