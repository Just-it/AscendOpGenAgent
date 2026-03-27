import tile.language as tl

@ascend_kernel
def layernorm_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr,
                     rows_per_core, tile_length, n_tiles, norm_size, eps):

    pid = tl.program_id(0)
    row_start_idx = pid * rows_per_core
    row_end_idx   = row_start_idx + rows_per_core

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    row_tile_ub   = tl.alloc_ub(tile_length, dtype=tl.float32)   # tile of input row
    weight_tile_ub = tl.alloc_ub(tile_length, dtype=tl.float32)  # tile of weight
    bias_tile_ub   = tl.alloc_ub(tile_length, dtype=tl.float32)  # tile of bias
    temp_ub       = tl.alloc_ub(tile_length, dtype=tl.float32)   # temporary computation buffer
    shared_ub     = tl.alloc_ub(tile_length, dtype=tl.float32)   # reduction workspace
    out_ub        = tl.alloc_ub(tile_length, dtype=tl.float32)   # output buffer

    # ------------------------------------------------------------
    # Per-row computation
    # ------------------------------------------------------------
    for row_idx in range(row_start_idx, row_end_idx):

        # ========================================================
        # PASS 1: compute mean of the row (tiled)
        # ========================================================
        row_sum = 0.0

        for tile_id in range(n_tiles):
            col_start = tile_id * tile_length
            offsets = row_idx * norm_size + col_start + tl.arange(0, tile_length)

            # ---- Load tile ----
            with tl.copyin():
                tl.load(input_ptr + offsets, row_tile_ub)

            # ---- Compute tile sum ----
            with tl.compute():
                tl.reduce_sum(shared_ub, row_tile_ub, shared_ub)
                tile_sum = tl.extract_scalar(shared_ub, 0)
            row_sum = row_sum + tile_sum

        # Compute mean
        row_mean = row_sum / norm_size

        # ========================================================
        # PASS 2: compute variance (sum of squared differences)
        # ========================================================
        row_var_sum = 0.0

        for tile_id in range(n_tiles):
            col_start = tile_id * tile_length
            offsets = row_idx * norm_size + col_start + tl.arange(0, tile_length)

            # ---- Load tile ----
            with tl.copyin():
                tl.load(input_ptr + offsets, row_tile_ub)

            # ---- Compute (x - mean)^2 for this tile ----
            with tl.compute():
                tl.vsub_scalar(temp_ub, row_tile_ub, row_mean)
                tl.vmul(temp_ub, temp_ub, temp_ub)
                tl.reduce_sum(shared_ub, temp_ub, shared_ub)
                tile_var = tl.extract_scalar(shared_ub, 0)
            row_var_sum = row_var_sum + tile_var

        # Compute variance and std
        row_var = row_var_sum / norm_size
        row_std = tl.sqrt(row_var + eps)

        # ========================================================
        # PASS 3: normalize, scale, and shift each tile
        # ========================================================
        for tile_id in range(n_tiles):
            col_start = tile_id * tile_length
            offsets = row_idx * norm_size + col_start + tl.arange(0, tile_length)
            weight_offsets = col_start + tl.arange(0, tile_length)

            # ---- Load input, weight, and bias tiles ----
            with tl.copyin():
                tl.load(input_ptr + offsets, row_tile_ub)
                tl.load(weight_ptr + weight_offsets, weight_tile_ub)
                tl.load(bias_ptr + weight_offsets, bias_tile_ub)

            # ---- Normalize: (x - mean) / std ----
            with tl.compute():
                tl.vsub_scalar(temp_ub, row_tile_ub, row_mean)
                tl.vdiv_scalar(temp_ub, temp_ub, row_std)

                # Scale and shift: out = normalized * weight + bias
                tl.vmul(out_ub, temp_ub, weight_tile_ub)
                tl.vadd(out_ub, out_ub, bias_tile_ub)

            # ---- Store tile ----
            with tl.copyout():
                tl.store(output_ptr + offsets, out_ub)


def layernorm_host(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                   output: torch.Tensor, eps: float = 1e-5):
    # ------------------------------------------------------------
    # Reshape input to 2D: (batch, norm_size)
    # ------------------------------------------------------------
    batch_size = x.shape[0]
    norm_size = weight.numel()  # features * dim1 * dim2
    rows = batch_size

    # ------------------------------------------------------------
    # Core Partitioning
    # ------------------------------------------------------------
    n_cores = 32
    rows_per_core = rows // n_cores

    # ------------------------------------------------------------
    # Tiling Strategy (column tiling for normalized dimensions)
    # ------------------------------------------------------------
    # Tile the normalized dimensions to fit in UB
    max_tile_len = 4096          # user-defined UB capacity
    tile_length = min(max_tile_len, norm_size)
    n_tiles = (norm_size + tile_length - 1) // tile_length

    layernorm_kernel[n_cores](
        x, weight, bias, output,
        rows_per_core,
        tile_length,
        n_tiles,
        norm_size,
        eps
    )