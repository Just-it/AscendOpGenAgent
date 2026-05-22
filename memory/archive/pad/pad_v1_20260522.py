import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_npu


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=[],
)
@triton.jit
def pad_kernel(
    in_ptr, out_ptr,
    in_d0: tl.constexpr, in_d1: tl.constexpr, in_d2: tl.constexpr, in_d3: tl.constexpr,
    out_d0: tl.constexpr, out_d1: tl.constexpr, out_d2: tl.constexpr, out_d3: tl.constexpr,
    pad_l0: tl.constexpr, pad_r0: tl.constexpr, pad_l1: tl.constexpr, pad_r1: tl.constexpr,
    pad_l2: tl.constexpr, pad_r2: tl.constexpr, pad_l3: tl.constexpr, pad_r3: tl.constexpr,
    in_s0: tl.constexpr, in_s1: tl.constexpr, in_s2: tl.constexpr, in_s3: tl.constexpr,
    out_s0: tl.constexpr, out_s1: tl.constexpr, out_s2: tl.constexpr, out_s3: tl.constexpr,
    mode: tl.constexpr,
    fill_value: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    out_numel = out_d0 * out_d1 * out_d2 * out_d3
    elements_per_core = tl.cdiv(out_numel, num_cores)
    core_start = pid * elements_per_core
    core_end = core_start + elements_per_core
    core_end = tl.minimum(core_end, out_numel)
    num_blocks_per_core = tl.cdiv(core_end - core_start, BLOCK_SIZE)

    for block_idx in range(num_blocks_per_core):
        block_start = core_start + block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < core_end

        c0 = offsets // out_s0
        rem0 = offsets - c0 * out_s0
        c1 = rem0 // out_s1
        rem1 = rem0 - c1 * out_s1
        c2 = rem1 // out_s2
        rem2 = rem1 - c2 * out_s2
        c3 = rem2 // out_s3

        in_c0 = c0 - pad_l0
        in_c1 = c1 - pad_l1
        in_c2 = c2 - pad_l2
        in_c3 = c3 - pad_l3

        if mode == 0:
            c0_f = c0.to(tl.float32)
            c1_f = c1.to(tl.float32)
            c2_f = c2.to(tl.float32)
            c3_f = c3.to(tl.float32)

            valid0 = (c0_f >= pad_l0) & (c0_f < pad_l0 + in_d0)
            valid1 = (c1_f >= pad_l1) & (c1_f < pad_l1 + in_d1)
            valid2 = (c2_f >= pad_l2) & (c2_f < pad_l2 + in_d2)
            valid3 = (c3_f >= pad_l3) & (c3_f < pad_l3 + in_d3)
            valid = valid0 & valid1 & valid2 & valid3

            safe_c0 = tl.where(valid0, in_c0, 0)
            safe_c1 = tl.where(valid1, in_c1, 0)
            safe_c2 = tl.where(valid2, in_c2, 0)
            safe_c3 = tl.where(valid3, in_c3, 0)

            in_idx = safe_c0 * in_s0 + safe_c1 * in_s1 + safe_c2 * in_s2 + safe_c3 * in_s3
            data = tl.load(in_ptr + in_idx, mask=valid & mask, other=fill_value)
        elif mode == 1:
            in_c0_f = in_c0.to(tl.float32)
            in_c1_f = in_c1.to(tl.float32)
            in_c2_f = in_c2.to(tl.float32)
            in_c3_f = in_c3.to(tl.float32)

            in_c0 = tl.where(in_c0_f < 0.0, -in_c0, in_c0)
            in_c0 = tl.where(in_c0_f >= in_d0, 2 * (in_d0 - 1) - in_c0, in_c0)
            in_c1 = tl.where(in_c1_f < 0.0, -in_c1, in_c1)
            in_c1 = tl.where(in_c1_f >= in_d1, 2 * (in_d1 - 1) - in_c1, in_c1)
            in_c2 = tl.where(in_c2_f < 0.0, -in_c2, in_c2)
            in_c2 = tl.where(in_c2_f >= in_d2, 2 * (in_d2 - 1) - in_c2, in_c2)
            in_c3 = tl.where(in_c3_f < 0.0, -in_c3, in_c3)
            in_c3 = tl.where(in_c3_f >= in_d3, 2 * (in_d3 - 1) - in_c3, in_c3)

            in_idx = in_c0 * in_s0 + in_c1 * in_s1 + in_c2 * in_s2 + in_c3 * in_s3
            data = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
        elif mode == 2:
            in_c0_f = in_c0.to(tl.float32)
            in_c1_f = in_c1.to(tl.float32)
            in_c2_f = in_c2.to(tl.float32)
            in_c3_f = in_c3.to(tl.float32)

            in_c0 = tl.where(in_c0_f < 0.0, 0, in_c0)
            in_c0 = tl.where(in_c0_f >= in_d0, in_d0 - 1, in_c0)
            in_c1 = tl.where(in_c1_f < 0.0, 0, in_c1)
            in_c1 = tl.where(in_c1_f >= in_d1, in_d1 - 1, in_c1)
            in_c2 = tl.where(in_c2_f < 0.0, 0, in_c2)
            in_c2 = tl.where(in_c2_f >= in_d2, in_d2 - 1, in_c2)
            in_c3 = tl.where(in_c3_f < 0.0, 0, in_c3)
            in_c3 = tl.where(in_c3_f >= in_d3, in_d3 - 1, in_c3)

            in_idx = in_c0 * in_s0 + in_c1 * in_s1 + in_c2 * in_s2 + in_c3 * in_s3
            data = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
        else:
            in_c0_f = in_c0.to(tl.float32)
            in_c1_f = in_c1.to(tl.float32)
            in_c2_f = in_c2.to(tl.float32)
            in_c3_f = in_c3.to(tl.float32)

            in_c0 = tl.where(in_c0_f < 0.0, in_c0 + in_d0, in_c0)
            in_c0 = tl.where(in_c0_f >= in_d0, in_c0 - in_d0, in_c0)
            in_c1 = tl.where(in_c1_f < 0.0, in_c1 + in_d1, in_c1)
            in_c1 = tl.where(in_c1_f >= in_d1, in_c1 - in_d1, in_c1)
            in_c2 = tl.where(in_c2_f < 0.0, in_c2 + in_d2, in_c2)
            in_c2 = tl.where(in_c2_f >= in_d2, in_c2 - in_d2, in_c2)
            in_c3 = tl.where(in_c3_f < 0.0, in_c3 + in_d3, in_c3)
            in_c3 = tl.where(in_c3_f >= in_d3, in_c3 - in_d3, in_c3)

            in_idx = in_c0 * in_s0 + in_c1 * in_s1 + in_c2 * in_s2 + in_c3 * in_s3
            data = tl.load(in_ptr + in_idx, mask=mask, other=0.0)

        tl.store(out_ptr + offsets, data, mask=mask)


@triton.jit
def pad_kernel_2d(
    in_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr,
    pad_t: tl.constexpr, pad_b: tl.constexpr,
    pad_l: tl.constexpr, pad_r: tl.constexpr,
    in_stride_h: tl.constexpr,
    out_stride_h: tl.constexpr,
    mode: tl.constexpr,
    fill_value: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows_per_core = tl.cdiv(H_out, num_cores)
    row_start = pid * rows_per_core
    row_end = tl.minimum(row_start + rows_per_core, H_out)
    num_rows = row_end - row_start

    for row_idx in range(num_rows):
        row = row_start + row_idx
        in_row = row - pad_t

        num_blocks = tl.cdiv(W_out, BLOCK_SIZE)
        for block_idx in range(num_blocks):
            col_start = block_idx * BLOCK_SIZE
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < W_out
            in_col = cols - pad_l

            if mode == 0:
                c2_f = row.to(tl.float32)
                c3_f = cols.to(tl.float32)
                valid2 = (c2_f >= pad_t) & (c2_f < pad_t + H)
                valid3 = (c3_f >= pad_l) & (c3_f < pad_l + W)
                valid = valid2 & valid3
                safe_row = tl.where(valid2, in_row, 0)
                safe_col = tl.where(valid3, in_col, 0)
                in_idx = safe_row * in_stride_h + safe_col
                data = tl.load(in_ptr + in_idx, mask=valid & mask, other=fill_value)
            elif mode == 1:
                in_row_f = in_row.to(tl.float32)
                in_col_f = in_col.to(tl.float32)
                safe_row = tl.where(in_row_f < 0.0, -in_row, in_row)
                safe_row = tl.where(in_row_f >= H, 2 * (H - 1) - in_row, safe_row)
                safe_col = tl.where(in_col_f < 0.0, -in_col, in_col)
                safe_col = tl.where(in_col_f >= W, 2 * (W - 1) - in_col, safe_col)
                in_idx = safe_row * in_stride_h + safe_col
                data = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
            elif mode == 2:
                in_row_f = in_row.to(tl.float32)
                in_col_f = in_col.to(tl.float32)
                safe_row = tl.where(in_row_f < 0.0, 0, in_row)
                safe_row = tl.where(in_row_f >= H, H - 1, safe_row)
                safe_col = tl.where(in_col_f < 0.0, 0, in_col)
                safe_col = tl.where(in_col_f >= W, W - 1, safe_col)
                in_idx = safe_row * in_stride_h + safe_col
                data = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
            else:
                in_row_f = in_row.to(tl.float32)
                in_col_f = in_col.to(tl.float32)
                safe_row = tl.where(in_row_f < 0.0, in_row + H, in_row)
                safe_row = tl.where(in_row_f >= H, in_row - H, safe_row)
                safe_col = tl.where(in_col_f < 0.0, in_col + W, in_col)
                safe_col = tl.where(in_col_f >= W, in_col - W, safe_col)
                in_idx = safe_row * in_stride_h + safe_col
                data = tl.load(in_ptr + in_idx, mask=mask, other=0.0)

            tl.store(out_ptr + row * out_stride_h + cols, data, mask=mask)


@triton.jit
def copy_kernel_2d(
    in_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr,
    pad_t: tl.constexpr, pad_l: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows_per_core = tl.cdiv(H, num_cores)
    row_start = pid * rows_per_core
    row_end = tl.minimum(row_start + rows_per_core, H)
    num_rows = row_end - row_start

    for row_idx in range(num_rows):
        in_row = row_start + row_idx
        out_row = in_row + pad_t
        base_in = in_row * W
        base_out = out_row * W_out + pad_l

        num_blocks = tl.cdiv(W, BLOCK_SIZE)
        for block_idx in range(num_blocks):
            col_start = block_idx * BLOCK_SIZE
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < W
            data = tl.load(in_ptr + base_in + cols, mask=mask)
            tl.store(out_ptr + base_out + cols, data, mask=mask)


@triton.jit
def copy_kernel_3d(
    in_ptr, out_ptr,
    D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
    D0_out: tl.constexpr, D1_out: tl.constexpr, D2_out: tl.constexpr,
    pad_d0: tl.constexpr, pad_d1: tl.constexpr, pad_d2: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_data_rows = D0 * D1
    rows_per_core = tl.cdiv(total_data_rows, num_cores)
    row_start = pid * rows_per_core
    row_end = tl.minimum(row_start + rows_per_core, total_data_rows)
    num_rows = row_end - row_start

    in_plane = row_start // D1
    in_row = row_start - in_plane * D1

    out_plane = in_plane + pad_d0
    out_row = in_row + pad_d1

    base_in = row_start * D2
    base_out = (out_plane * D1_out + out_row) * D2_out + pad_d2

    for row_idx in range(num_rows):
        num_blocks = tl.cdiv(D2, BLOCK_SIZE)
        for block_idx in range(num_blocks):
            col_start = block_idx * BLOCK_SIZE
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < D2
            data = tl.load(in_ptr + base_in + cols, mask=mask)
            tl.store(out_ptr + base_out + cols, data, mask=mask)

        base_in += D2
        out_row += 1
        if out_row == D1 + pad_d1:
            out_plane += 1
            out_row = pad_d1
            base_out = (out_plane * D1_out + out_row) * D2_out + pad_d2
        else:
            base_out += D2_out


@triton.jit
def pad_kernel_2d_batched(
    in_ptr, out_ptr,
    Batch: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr,
    pad_t: tl.constexpr, pad_b: tl.constexpr,
    pad_l: tl.constexpr, pad_r: tl.constexpr,
    mode: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_out_rows = Batch * H_out
    if pid >= total_out_rows:
        return

    batch_idx = pid // H_out
    row = pid - batch_idx * H_out
    in_row = row - pad_t
    base_out = batch_idx * H_out * W_out + row * W_out

    num_blocks = tl.cdiv(W_out, BLOCK_SIZE)
    for block_idx in range(num_blocks):
        col_start = block_idx * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < W_out
        in_col = cols - pad_l

        if mode == 1:
            in_row_f = in_row.to(tl.float32)
            in_col_f = in_col.to(tl.float32)
            safe_row = tl.where(in_row_f < 0.0, -in_row, in_row)
            safe_row = tl.where(in_row_f >= H, 2 * (H - 1) - in_row, safe_row)
            safe_col = tl.where(in_col_f < 0.0, -in_col, in_col)
            safe_col = tl.where(in_col_f >= W, 2 * (W - 1) - in_col, safe_col)
            in_idx = batch_idx * H * W + safe_row * W + safe_col
            data = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
        elif mode == 2:
            in_row_f = in_row.to(tl.float32)
            in_col_f = in_col.to(tl.float32)
            safe_row = tl.where(in_row_f < 0.0, 0, in_row)
            safe_row = tl.where(in_row_f >= H, H - 1, safe_row)
            safe_col = tl.where(in_col_f < 0.0, 0, in_col)
            safe_col = tl.where(in_col_f >= W, W - 1, safe_col)
            in_idx = batch_idx * H * W + safe_row * W + safe_col
            data = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
        else:
            in_row_f = in_row.to(tl.float32)
            in_col_f = in_col.to(tl.float32)
            safe_row = tl.where(in_row_f < 0.0, in_row + H, in_row)
            safe_row = tl.where(in_row_f >= H, in_row - H, safe_row)
            safe_col = tl.where(in_col_f < 0.0, in_col + W, in_col)
            safe_col = tl.where(in_col_f >= W, in_col - W, safe_col)
            in_idx = batch_idx * H * W + safe_row * W + safe_col
            data = tl.load(in_ptr + in_idx, mask=mask, other=0.0)

        tl.store(out_ptr + base_out + cols, data, mask=mask)


@triton.jit
def pad_kernel_3d_constant_2d(
    in_ptr, out_ptr,
    D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
    D0_out: tl.constexpr, D1_out: tl.constexpr, D2_out: tl.constexpr,
    pad_d0: tl.constexpr, pad_d1: tl.constexpr, pad_d2: tl.constexpr,
    fill_value: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    d0 = tl.program_id(0)
    d1 = tl.program_id(1)

    if d0 >= D0_out or d1 >= D1_out:
        return

    in_d0 = d0 - pad_d0
    in_d1 = d1 - pad_d1
    valid0 = (in_d0 >= 0) & (in_d0 < D0)
    valid1 = (in_d1 >= 0) & (in_d1 < D1)

    safe_d0 = tl.where(valid0, in_d0, 0)
    safe_d1 = tl.where(valid1, in_d1, 0)
    base_in_plane = safe_d0 * D1 * D2 + safe_d1 * D2
    base_out = d0 * D1_out * D2_out + d1 * D2_out

    num_blocks = tl.cdiv(D2_out, BLOCK_SIZE)
    for block_idx in range(num_blocks):
        col_start = block_idx * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < D2_out
        in_d2 = cols - pad_d2
        in_d2_f = in_d2.to(tl.float32)
        valid2 = (in_d2_f >= 0.0) & (in_d2_f < D2)
        safe_d2 = tl.where(valid2, in_d2, 0)
        valid = valid0 & valid1 & valid2
        in_idx = base_in_plane + safe_d2
        data = tl.load(in_ptr + in_idx, mask=valid & mask, other=fill_value)
        tl.store(out_ptr + base_out + cols, data, mask=mask)


@triton.jit
def pad_kernel_3d_nonconstant_2d(
    in_ptr, out_ptr,
    D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
    D0_out: tl.constexpr, D1_out: tl.constexpr, D2_out: tl.constexpr,
    pad_d0: tl.constexpr, pad_d1: tl.constexpr, pad_d2: tl.constexpr,
    mode: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    d0 = tl.program_id(0)
    d1 = tl.program_id(1)

    if d0 >= D0_out or d1 >= D1_out:
        return

    in_d0 = d0 - pad_d0
    in_d1 = d1 - pad_d1
    in_d0_f = in_d0.to(tl.float32)
    in_d1_f = in_d1.to(tl.float32)

    if mode == 1:
        safe_d0 = tl.where(in_d0_f < 0.0, -in_d0, in_d0)
        safe_d0 = tl.where(in_d0_f >= D0, 2 * (D0 - 1) - in_d0, safe_d0)
        safe_d1 = tl.where(in_d1_f < 0.0, -in_d1, in_d1)
        safe_d1 = tl.where(in_d1_f >= D1, 2 * (D1 - 1) - in_d1, safe_d1)
    elif mode == 2:
        safe_d0 = tl.maximum(0, tl.minimum(in_d0, D0 - 1))
        safe_d1 = tl.maximum(0, tl.minimum(in_d1, D1 - 1))
    else:
        safe_d0 = tl.where(in_d0_f < 0.0, in_d0 + D0, in_d0)
        safe_d0 = tl.where(in_d0_f >= D0, in_d0 - D0, safe_d0)
        safe_d1 = tl.where(in_d1_f < 0.0, in_d1 + D1, in_d1)
        safe_d1 = tl.where(in_d1_f >= D1, in_d1 - D1, safe_d1)

    base_in_plane = safe_d0 * D1 * D2 + safe_d1 * D2
    base_out = d0 * D1_out * D2_out + d1 * D2_out

    num_blocks = tl.cdiv(D2_out, BLOCK_SIZE)
    for block_idx in range(num_blocks):
        col_start = block_idx * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < D2_out
        in_d2 = cols - pad_d2

        if mode == 1:
            in_d2_f = in_d2.to(tl.float32)
            safe_d2 = tl.where(in_d2_f < 0.0, -in_d2, in_d2)
            safe_d2 = tl.where(in_d2_f >= D2, 2 * (D2 - 1) - in_d2, safe_d2)
        elif mode == 2:
            safe_d2 = tl.maximum(0, tl.minimum(in_d2, D2 - 1))
        else:
            in_d2_f = in_d2.to(tl.float32)
            safe_d2 = tl.where(in_d2_f < 0.0, in_d2 + D2, in_d2)
            safe_d2 = tl.where(in_d2_f >= D2, in_d2 - D2, safe_d2)

        in_idx = base_in_plane + safe_d2
        data = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
        tl.store(out_ptr + base_out + cols, data, mask=mask)


@triton.jit
def pad_kernel_3d_constant_v2(
    in_ptr, out_ptr,
    D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
    D0_out: tl.constexpr, D1_out: tl.constexpr, D2_out: tl.constexpr,
    pad_d0: tl.constexpr, pad_d1: tl.constexpr, pad_d2: tl.constexpr,
    fill_value: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = D0_out * D1_out
    rows_per_core = tl.cdiv(total_rows, num_cores)
    row_start = pid * rows_per_core
    row_end = tl.minimum(row_start + rows_per_core, total_rows)

    d0 = row_start // D1_out
    d1 = row_start - d0 * D1_out

    num_rows = row_end - row_start
    for _ in range(num_rows):
        base_out = (d0 * D1_out + d1) * D2_out
        in_d0 = d0 - pad_d0
        in_d1 = d1 - pad_d1
        valid01 = (in_d0 >= 0) & (in_d0 < D0) & (in_d1 >= 0) & (in_d1 < D1)
        safe_d0 = tl.where(valid01, in_d0, 0)
        safe_d1 = tl.where(valid01, in_d1, 0)
        base_in = safe_d0 * D1 * D2 + safe_d1 * D2

        num_blocks = tl.cdiv(D2_out, BLOCK_SIZE)
        for block_idx in range(num_blocks):
            col_start = block_idx * BLOCK_SIZE
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < D2_out
            in_d2 = cols - pad_d2
            valid2 = (in_d2 >= 0) & (in_d2 < D2)
            valid = valid01 & valid2
            safe_d2 = tl.where(valid2, in_d2, 0)
            in_idx = base_in + safe_d2
            data = tl.load(in_ptr + in_idx, mask=valid & mask, other=fill_value)
            tl.store(out_ptr + base_out + cols, data, mask=mask)

        d1 += 1
        if d1 == D1_out:
            d1 = 0
            d0 += 1


@triton.jit
def pad_kernel_3d_nonconstant_v2(
    in_ptr, out_ptr,
    D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
    D0_out: tl.constexpr, D1_out: tl.constexpr, D2_out: tl.constexpr,
    pad_d0: tl.constexpr, pad_d1: tl.constexpr, pad_d2: tl.constexpr,
    mode: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = D0 * D1_out
    rows_per_core = tl.cdiv(total_rows, num_cores)
    row_start = pid * rows_per_core
    row_end = tl.minimum(row_start + rows_per_core, total_rows)

    d0 = row_start // D1_out
    d1 = row_start - d0 * D1_out

    num_rows = row_end - row_start
    for _ in range(num_rows):
        base_out = (d0 * D1_out + d1) * D2_out
        base_in_plane = d0 * D1 * D2
        in_d1 = d1 - pad_d1

        if mode == 1:
            in_d1_f = in_d1.to(tl.float32)
            safe_d1 = tl.where(in_d1_f < 0.0, -in_d1, in_d1)
            safe_d1 = tl.where(in_d1_f >= D1, 2 * (D1 - 1) - in_d1, safe_d1)
        elif mode == 2:
            safe_d1 = tl.maximum(0, tl.minimum(in_d1, D1 - 1))
        else:
            in_d1_f = in_d1.to(tl.float32)
            safe_d1 = tl.where(in_d1_f < 0.0, in_d1 + D1, in_d1)
            safe_d1 = tl.where(in_d1_f >= D1, in_d1 - D1, safe_d1)

        num_blocks = tl.cdiv(D2_out, BLOCK_SIZE)
        for block_idx in range(num_blocks):
            col_start = block_idx * BLOCK_SIZE
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < D2_out
            in_d2 = cols - pad_d2

            if mode == 1:
                in_d2_f = in_d2.to(tl.float32)
                safe_d2 = tl.where(in_d2_f < 0.0, -in_d2, in_d2)
                safe_d2 = tl.where(in_d2_f >= D2, 2 * (D2 - 1) - in_d2, safe_d2)
            elif mode == 2:
                safe_d2 = tl.maximum(0, tl.minimum(in_d2, D2 - 1))
            else:
                in_d2_f = in_d2.to(tl.float32)
                safe_d2 = tl.where(in_d2_f < 0.0, in_d2 + D2, in_d2)
                safe_d2 = tl.where(in_d2_f >= D2, in_d2 - D2, safe_d2)

            in_idx = base_in_plane + safe_d1 * D2 + safe_d2
            data = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
            tl.store(out_ptr + base_out + cols, data, mask=mask)

        d1 += 1
        if d1 == D1_out:
            d1 = 0
            d0 += 1


def select_block_size(width):
    if width <= 64:
        return 64
    elif width <= 128:
        return 128
    elif width <= 256:
        return 256
    elif width <= 512:
        return 512
    elif width <= 1024:
        return 1024
    elif width <= 2048:
        return 2048
    else:
        return 4096


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except Exception:
            self.VEC_CORE_NUM = 40

    def forward(self, x, pad, mode='constant', value=None):
        if value is None:
            value = 0.0

        pad_list = list(pad)
        ndim_orig = x.ndim

        dim_deltas = {}
        num_pad_dims = len(pad_list) // 2
        for i in range(num_pad_dims):
            dim_idx = ndim_orig - 1 - i
            dim_deltas[dim_idx] = pad_list[2 * i] + pad_list[2 * i + 1]
        out_shape_orig = [x.shape[i] + dim_deltas.get(i, 0) for i in range(ndim_orig)]

        output = torch.empty(out_shape_orig, device=x.device, dtype=x.dtype)

        if not x.is_contiguous():
            x = x.contiguous()

        squeeze_count = 0
        while squeeze_count < ndim_orig - 1 and x.shape[squeeze_count] == 1:
            dim_idx_from_right = ndim_orig - 1 - squeeze_count
            pad_idx = 2 * dim_idx_from_right
            if pad_idx < len(pad_list):
                if pad_list[pad_idx] == 0 and pad_list[pad_idx + 1] == 0:
                    squeeze_count += 1
                else:
                    break
            else:
                squeeze_count += 1

        x_kernel = x
        output_kernel = output
        pad_list_kernel = pad_list
        ndim = ndim_orig

        if squeeze_count > 0:
            squeeze_dims = tuple(range(squeeze_count))
            x_kernel = x.squeeze(squeeze_dims)
            out_shape_squeezed = out_shape_orig[squeeze_count:]
            output_kernel = output.view(out_shape_squeezed)
            num_pad_pairs = len(pad_list) // 2
            implicit_pad_dims = ndim_orig - num_pad_pairs
            if squeeze_count > implicit_pad_dims:
                entries_to_remove = 2 * (squeeze_count - implicit_pad_dims)
                pad_list_kernel = pad_list[:-entries_to_remove]
            else:
                pad_list_kernel = pad_list
            ndim = x_kernel.ndim

        mode_map = {'constant': 0, 'reflect': 1, 'replicate': 2, 'circular': 3}
        mode_val = mode_map.get(mode, 0)

        if ndim == 2:
            H, W = x_kernel.shape[0], x_kernel.shape[1]
            H_out, W_out = out_shape_squeezed[0] if squeeze_count > 0 else out_shape_orig[0], out_shape_squeezed[1] if squeeze_count > 0 else out_shape_orig[1]
            pad_l = pad_list_kernel[0] if len(pad_list_kernel) > 0 else 0
            pad_r = pad_list_kernel[1] if len(pad_list_kernel) > 1 else 0
            pad_t = pad_list_kernel[2] if len(pad_list_kernel) > 2 else 0
            pad_b = pad_list_kernel[3] if len(pad_list_kernel) > 3 else 0
            block_size = select_block_size(W_out if mode != 'constant' else W)

            if mode == 'constant':
                output_kernel.fill_(float(value))
                copy_kernel_2d[(self.VEC_CORE_NUM,)](
                    x_kernel, output_kernel,
                    H=H, W=W,
                    H_out=H_out, W_out=W_out,
                    pad_t=pad_t, pad_l=pad_l,
                    num_cores=self.VEC_CORE_NUM,
                    BLOCK_SIZE=block_size,
                )
            else:
                pad_kernel_2d[(self.VEC_CORE_NUM,)](
                    x_kernel, output_kernel,
                    H=H, W=W,
                    H_out=H_out, W_out=W_out,
                    pad_t=pad_t, pad_b=pad_b, pad_l=pad_l, pad_r=pad_r,
                    in_stride_h=W,
                    out_stride_h=W_out,
                    mode=mode_val,
                    fill_value=float(value),
                    num_cores=self.VEC_CORE_NUM,
                    BLOCK_SIZE=block_size,
                )
        elif ndim == 3 and mode == 'constant':
            D0, D1, D2 = x_kernel.shape[0], x_kernel.shape[1], x_kernel.shape[2]
            D0_out, D1_out, D2_out = out_shape_squeezed[0] if squeeze_count > 0 else out_shape_orig[0], out_shape_squeezed[1] if squeeze_count > 0 else out_shape_orig[1], out_shape_squeezed[2] if squeeze_count > 0 else out_shape_orig[2]
            pad_l2 = pad_list_kernel[0] if len(pad_list_kernel) > 0 else 0
            pad_l1 = pad_list_kernel[2] if len(pad_list_kernel) > 2 else 0
            pad_l0 = pad_list_kernel[4] if len(pad_list_kernel) > 4 else 0
            block_size = select_block_size(D2)

            output_kernel.fill_(float(value))
            copy_kernel_3d[(self.VEC_CORE_NUM,)](
                x_kernel, output_kernel,
                D0=D0, D1=D1, D2=D2,
                D0_out=D0_out, D1_out=D1_out, D2_out=D2_out,
                pad_d0=pad_l0, pad_d1=pad_l1, pad_d2=pad_l2,
                num_cores=self.VEC_CORE_NUM,
                BLOCK_SIZE=block_size,
            )
        elif ndim == 3 and mode != 'constant':
            D0, D1, D2 = x_kernel.shape[0], x_kernel.shape[1], x_kernel.shape[2]
            D0_out, D1_out, D2_out = out_shape_squeezed[0] if squeeze_count > 0 else out_shape_orig[0], out_shape_squeezed[1] if squeeze_count > 0 else out_shape_orig[1], out_shape_squeezed[2] if squeeze_count > 0 else out_shape_orig[2]
            pad_l2 = pad_list_kernel[0] if len(pad_list_kernel) > 0 else 0
            pad_r2 = pad_list_kernel[1] if len(pad_list_kernel) > 1 else 0
            pad_l1 = pad_list_kernel[2] if len(pad_list_kernel) > 2 else 0
            pad_r1 = pad_list_kernel[3] if len(pad_list_kernel) > 3 else 0
            pad_l0 = pad_list_kernel[4] if len(pad_list_kernel) > 4 else 0
            block_size = select_block_size(D2_out)

            total_grid_rows = D0 * D1_out
            if total_grid_rows > 3000:
                pad_kernel_3d_nonconstant_v2[(self.VEC_CORE_NUM,)](
                    x_kernel, output_kernel,
                    D0=D0, D1=D1, D2=D2,
                    D0_out=D0_out, D1_out=D1_out, D2_out=D2_out,
                    pad_d0=pad_l0, pad_d1=pad_l1, pad_d2=pad_l2,
                    mode=mode_val,
                    num_cores=self.VEC_CORE_NUM,
                    BLOCK_SIZE=block_size,
                )
            else:
                pad_kernel_3d_nonconstant_2d[(D0_out, D1_out)](
                    x_kernel, output_kernel,
                    D0=D0, D1=D1, D2=D2,
                    D0_out=D0_out, D1_out=D1_out, D2_out=D2_out,
                    pad_d0=pad_l0, pad_d1=pad_l1, pad_d2=pad_l2,
                    mode=mode_val,
                    num_cores=self.VEC_CORE_NUM,
                    BLOCK_SIZE=block_size,
                )
        else:
            in_shape_padded = [1] * (4 - ndim) + list(x_kernel.shape)
            out_shape_padded = [1] * (4 - ndim) + list(output_kernel.shape)

            pad_entries = list(zip(pad_list_kernel[::2], pad_list_kernel[1::2]))
            pad_l_dict = {3 - i: e[0] for i, e in enumerate(pad_entries)}
            pad_r_dict = {3 - i: e[1] for i, e in enumerate(pad_entries)}
            pad_l = [pad_l_dict.get(i, 0) for i in range(4)]
            pad_r = [pad_r_dict.get(i, 0) for i in range(4)]

            in_strides = [
                in_shape_padded[1] * in_shape_padded[2] * in_shape_padded[3],
                in_shape_padded[2] * in_shape_padded[3],
                in_shape_padded[3],
                1,
            ]
            out_strides = [
                out_shape_padded[1] * out_shape_padded[2] * out_shape_padded[3],
                out_shape_padded[2] * out_shape_padded[3],
                out_shape_padded[3],
                1,
            ]

            pad_kernel[(self.VEC_CORE_NUM,)](
                x_kernel, output_kernel,
                in_d0=in_shape_padded[0], in_d1=in_shape_padded[1], in_d2=in_shape_padded[2], in_d3=in_shape_padded[3],
                out_d0=out_shape_padded[0], out_d1=out_shape_padded[1], out_d2=out_shape_padded[2], out_d3=out_shape_padded[3],
                pad_l0=pad_l[0], pad_r0=pad_r[0], pad_l1=pad_l[1], pad_r1=pad_r[1],
                pad_l2=pad_l[2], pad_r2=pad_r[2], pad_l3=pad_l[3], pad_r3=pad_r[3],
                in_s0=in_strides[0], in_s1=in_strides[1], in_s2=in_strides[2], in_s3=in_strides[3],
                out_s0=out_strides[0], out_s1=out_strides[1], out_s2=out_strides[2], out_s3=out_strides[3],
                mode=mode_val,
                fill_value=float(value),
                num_cores=self.VEC_CORE_NUM,
            )

        return output
