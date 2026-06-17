# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/fbgemm_operator_cases/newtest_cases/test__kernel_quantize_fp8_group_npu.py
# Main kernel: _kernel_quantize_fp8_group
# PT file: test__kernel_quantize_fp8_group_v2.pt

import triton
import triton.language as tl


# === _kernel_quantize_fp8_group ===
@triton.jit
def _kernel_quantize_fp8_group(
    A,
    A_scale,
    A_fp8,
    scale_ub,
    m_sizes,
    M,
    K,
    stride_am,
    stride_ak,
    stride_om,
    stride_ok,
    stride_a_scale_m,
    stride_a_scale_k,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    USE_INT64: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    USE_M_MAJOR: tl.constexpr,
    G: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
):
    """Quantize and scale each GROUP_SIZE chunk of each row.

    Scale per group i is computed as 1 / (MAX_FP8 / max(abs(A[i:i+GROUP_SIZE])))

    Each kernel thread is responsible for one row and loads and processes a tunable
    number of groups at once.

    Args:
        A (Tensor): [M, K] higher precision input tensor.
        A_scale (Tensor): [M, cdiv(K, GROUP_SIZE)] reciprocal scale tensor per group.
        A_fp8 (Tensor): [M, K] fp8 scaled tensor. A_fp8 = A * a
        scale_ub (Tensor): [1] Maximum allowed value for scale.
        m_sizes (Optional[Tensor]): [G] Number of rows in each group.
        M (int): Number of rows.
        K (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_om (int): Stride of m dimension of output.
        stride_ok (int): Stride of k dimension of output.
        stride_a_scale_m (int): Stride of m dimension of A_scale.
        stride_a_scale_k (int): Stride of k dimension of A_scale.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maxmimum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whether to apply scale_ub.
        USE_INT64 (bool): Whether to index using int64, which may be needed for large tensors.
        GROUP_SIZE (int): Group size for K dimension of A_scale and kernel.
        USE_M_MAJOR (bool): Whether to use grouped M-major layout for A_scale.
        G (int): Number of groups in A_scale, only relevant when m_sizes is provided.
        GROUP_LOAD (int): Number of groups to load and process simultaneously.
    """
    pid = tl.program_id(0)
    if USE_INT64:
        pid = pid.to(tl.int64)
    # We load group_size * group_load chunks at a time.
    row_offset = pid * stride_am
    out_offset = pid * stride_om
    scale_row_offset = pid * stride_a_scale_m
    k_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE)
    scale_k_offset = tl.arange(0, GROUP_LOAD)
    NUM_GROUPS: tl.constexpr = K // GROUP_SIZE

    # When dealing with an M-major grouped gemm, we need to figure out
    # which group this thread corresponds to and figure out the corresponding
    # scale offset.
    group_offset = 0
    group_cumsum = 0
    group_M = 0
    stop = False
    if USE_M_MAJOR and G > 0:
        # Iterate over groups to both compute the cumulative sum and find which group we are in.
        for i in range(G):
            if not stop:
                group_M = tl.cast(tl.load(m_sizes + i), pid.dtype)
                if (group_cumsum + group_M) <= pid:
                    group_cumsum += group_M
                else:
                    # Indicate we are finished computing cumsum.
                    stop = True

        group_offset = group_cumsum * NUM_GROUPS

    for k in range(0, tl.cdiv(K, (GROUP_LOAD * GROUP_SIZE))):
        # Load groups of the input.
        chunk_offset = k_offset + k * GROUP_LOAD * GROUP_SIZE
        a = tl.load(
            A + row_offset + chunk_offset * stride_ak, mask=chunk_offset < K, other=0.0
        )
        # View loaded chunk as a set of groups.
        a_grouped = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE])
        # Reduce over groups.
        group_max = tl.max(tl.abs(a_grouped), axis=1)
        # Apply clamping if specified.
        if CLAMP_MAX:
            ub = tl.load(scale_ub)
            group_max = tl.clamp(group_max, EPS, ub)
        else:
            group_max = tl.maximum(group_max, EPS)
        # Scale and quantize.
        a_scale = MAX_FP8 / group_max
        scale_chunk_offset = scale_k_offset + k * GROUP_LOAD

        if USE_M_MAJOR and G > 0:
            tl.store(
                A_scale
                + group_offset
                + (pid - group_cumsum) * stride_a_scale_k
                + (scale_chunk_offset * group_M),
                1.0 / a_scale,
                mask=scale_chunk_offset < NUM_GROUPS,
            )
        else:
            if USE_M_MAJOR:
                tl.store(
                    A_scale
                    + pid * stride_a_scale_k
                    + scale_chunk_offset * stride_a_scale_m,
                    1.0 / a_scale,
                    mask=scale_chunk_offset < NUM_GROUPS,
                )
            else:
                tl.store(
                    A_scale + scale_row_offset + scale_chunk_offset * stride_a_scale_k,
                    1.0 / a_scale,
                    mask=scale_chunk_offset < NUM_GROUPS,
                )
        # Apply scale to input.
        a_fp8 = a_grouped * a_scale[:, None]
        # Clamp to FP8 range to avoid overflow
        a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8).to(TL_FP8_DTYPE)
        # Write to output.
        tl.store(
            A_fp8 + out_offset + chunk_offset * stride_ok,
            tl.ravel(a_fp8),
            mask=chunk_offset < K,
        )

