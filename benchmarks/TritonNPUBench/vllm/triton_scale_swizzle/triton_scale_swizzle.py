# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/vllm_operator_cases/newtest_cases/test_triton_scale_swizzle.py
# Main kernel: triton_scale_swizzle
# PT file: triton_scale_swizzle_v2.pt

import triton
import triton.language as tl
import torch

# === triton_scale_swizzle ===
@triton.jit
def triton_scale_swizzle(
    scale_ptr: torch.Tensor,
    scale_rows: int,
    scale_cols: int,
    output_ptr: torch.Tensor,
    input_row_stride: int,
    output_block_stride: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """
    Rearranges tensor data from row-major to block-scaled swizzle format.

    Args:
        scale_ptr: Pointer to the input scale tensor
        scale_rows: Number of rows in the scale tensor
        scale_cols: Number of columns in the scale tensor
        output_ptr: Pointer to the output tensor
        input_row_stride: Stride between rows in the input tensor
        output_block_stride: Stride between blocks in the output tensor
        BLOCK_ROWS: Number of rows in a tile (compile-time constant)
        BLOCK_COLS: Number of columns in a tile (compile-time constant)
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    rows = tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, BLOCK_COLS)[None, :]

    # Calculate starting row and column for this tile
    start_row = pid_row * BLOCK_ROWS
    start_col = pid_col * BLOCK_COLS
    global_rows = start_row + rows
    global_cols = start_col + cols

    mask = (global_rows < scale_rows) & (global_cols < scale_cols)

    input_scales = tl.load(
        scale_ptr + global_rows * input_row_stride + global_cols,
        mask=mask,
        other=0.0,
    )

    r_div_32 = rows // 32
    r_mod_32 = rows % 32

    # 2) Rearrange to (32, 4, 4) then to final (32, 16) coordinates
    dest_indices = r_mod_32 * 16 + r_div_32 * 4 + cols

    # Flatten
    dest_indices_flat = tl.reshape(dest_indices, (BLOCK_ROWS * BLOCK_COLS))
    scales_flat = tl.reshape(input_scales, (BLOCK_ROWS * BLOCK_COLS))

    # Calculate block offset using provided output block stride
    LOCAL_NUMEL = BLOCK_ROWS * BLOCK_COLS
    block_offset = pid_col * LOCAL_NUMEL + (pid_row * output_block_stride)

    tl.store(
        output_ptr + block_offset + dest_indices_flat,
        scales_flat,
    )

