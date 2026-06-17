# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/sglang_operator_cases/newtest_cases/test_merge_16x16_to_64x64_inverse_kernel.py
# Main kernel: merge_16x16_to_64x64_inverse_kernel
# PT file: test_merge_16x16_to_64x64_inverse_kernel_v2_uint2.pt

import triton
import triton.language as tl


# === merge_16x16_to_64x64_inverse_kernel ===
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * 64
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 64

    p_A_21 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0)
    )
    p_A_32 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0)
    )
    p_A_31 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0)
    )
    p_A_43 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0)
    )
    p_A_42 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0)
    )
    p_A_41 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0)
    )
    p_Ad_11 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 64, 0), (16, 16), (1, 0)
    )
    p_Ad_22 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0)
    )
    p_Ad_33 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0)
    )
    p_Ad_44 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0)
    )

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)
    A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)

    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_33 = tl.load(p_Ad_33, boundary_check=(0, 1)).to(tl.float32)
    Ai_44 = tl.load(p_Ad_44, boundary_check=(0, 1)).to(tl.float32)

    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
    )
    Ai_32 = -tl.dot(
        tl.dot(Ai_33, A_32, input_precision="ieee"), Ai_22, input_precision="ieee"
    )
    Ai_43 = -tl.dot(
        tl.dot(Ai_44, A_43, input_precision="ieee"), Ai_33, input_precision="ieee"
    )

    Ai_31 = -tl.dot(
        Ai_33,
        tl.dot(A_31, Ai_11, input_precision="ieee")
        + tl.dot(A_32, Ai_21, input_precision="ieee"),
        input_precision="ieee",
    )
    Ai_42 = -tl.dot(
        Ai_44,
        tl.dot(A_42, Ai_22, input_precision="ieee")
        + tl.dot(A_43, Ai_32, input_precision="ieee"),
        input_precision="ieee",
    )
    Ai_41 = -tl.dot(
        Ai_44,
        tl.dot(A_41, Ai_11, input_precision="ieee")
        + tl.dot(A_42, Ai_21, input_precision="ieee")
        + tl.dot(A_43, Ai_31, input_precision="ieee"),
        input_precision="ieee",
    )

    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64, 0), (16, 16), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 16), (16, 16), (1, 0)
    )
    p_Ai_33 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 32), (16, 16), (1, 0)
    )
    p_Ai_44 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 48), (16, 16), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0)
    )
    p_Ai_31 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0)
    )
    p_Ai_32 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0)
    )
    p_Ai_41 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0)
    )
    p_Ai_42 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0)
    )
    p_Ai_43 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0)
    )
    tl.store(
        p_Ai_11,
        Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_33,
        Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_44,
        Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_31,
        Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_32,
        Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_41,
        Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_42,
        Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_43,
        Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )

    fill_zeros = tl.zeros((16, 16), dtype=tl.float32)
    p_Ai_12 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64, 16), (16, 16), (1, 0)
    )
    p_Ai_13 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64, 32), (16, 16), (1, 0)
    )
    p_Ai_14 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64, 48), (16, 16), (1, 0)
    )
    p_Ai_23 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 32), (16, 16), (1, 0)
    )
    p_Ai_24 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 48), (16, 16), (1, 0)
    )
    p_Ai_34 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 48), (16, 16), (1, 0)
    )
    tl.store(
        p_Ai_12,
        fill_zeros.to(p_Ai_12.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_13,
        fill_zeros.to(p_Ai_13.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_14,
        fill_zeros.to(p_Ai_14.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_23,
        fill_zeros.to(p_Ai_23.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_24,
        fill_zeros.to(p_Ai_24.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_34,
        fill_zeros.to(p_Ai_34.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )

