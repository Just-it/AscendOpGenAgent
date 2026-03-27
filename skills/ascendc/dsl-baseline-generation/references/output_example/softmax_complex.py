import tile.language as tl

@ascend_kernel
def softmax_kernel(input_ptr, output_ptr,
                    rows_per_core, tile_length, n_tiles):

    pid = tl.program_id(0)
    row_start_idx = pid * rows_per_core
    row_end_idx   = row_start_idx + rows_per_core

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    row_tile_ub   = tl.alloc_ub(tile_length, dtype=tl.float32)   # tile of input row
    exp_tile_ub   = tl.alloc_ub(tile_length, dtype=tl.float32)   # tile of exp(x - max)
    shared_ub     = tl.alloc_ub(tile_length, dtype=tl.float32)   # reduction workspace
    out_ub     = tl.alloc_ub(tile_length, dtype=tl.float32)   # reduction workspace


    # ------------------------------------------------------------
    # Per-row computation
    # ------------------------------------------------------------
    for row_idx in range(row_start_idx, row_end_idx):

        # ========================================================
        # PASS 1: compute global max of a long row (tiled)
        # ========================================================
        row_max = -1e30

        for tile_id in range(n_tiles):
            col_start = tile_id * tile_length
            offsets = row_idx * (tile_length * n_tiles) \
                    + col_start + tl.arange(0, tile_length)

            # ---- Load tile ----
            with tl.copyin():
                tl.load(input_ptr + offsets, row_tile_ub)

            # ---- Compute tile max ----
            with tl.compute():
                tl.reduce_max(shared_ub, row_tile_ub, shared_ub)
                tile_max = tl.extract_scalar(shared_ub, 0)
            row_max = tl.max(row_max, tile_max)

        # ========================================================
        # PASS 2: compute global sum of exp(x - row_max)
        # ========================================================
        row_sum = 0.0

        for tile_id in range(n_tiles):
            col_start = tile_id * tile_length
            offsets = row_idx * (tile_length * n_tiles) \
                    + col_start + tl.arange(0, tile_length)

            # ---- Load tile ----
            with tl.copyin():
                tl.load(input_ptr + offsets, row_tile_ub)

            # ---- Compute exp(x - max) for this tile ----
            with tl.compute():
                tl.vsub_scalar(exp_tile_ub, row_tile_ub, row_max)
                tl.vexp(exp_tile_ub, exp_tile_ub)
                tl.reduce_sum(shared_ub, exp_tile_ub, shared_ub)
                tile_sum = tl.extract_scalar(shared_ub, 0)
            row_sum = row_sum + tile_sum

        # ========================================================
        # PASS 3: normalize each tile and store output
        # ========================================================
        for tile_id in range(n_tiles):
            col_start = tile_id * tile_length
            offsets = row_idx * (tile_length * n_tiles) \
                    + col_start + tl.arange(0, tile_length)

            # ---- Load tile ----
            with tl.copyin():
                tl.load(input_ptr + offsets, row_tile_ub)

            # ---- exp(x - max) for this tile ----
            with tl.compute():
                tl.vsub_scalar(exp_tile_ub, row_tile_ub, row_max)
                tl.vexp(exp_tile_ub, exp_tile_ub)

                # normalize output_tile = exp / row_sum
                tl.vdiv_scalar(out_ub, exp_tile_ub, row_sum)

            # ---- store tile ----
            with tl.copyout():
                tl.store(output_ptr + offsets, out_ub)


def softmax_host(x: torch.Tensor, output: torch.Tensor):
    rows = x.shape[0]
    cols = x.shape[1]

    # ------------------------------------------------------------
    # Core Partitioning
    # ------------------------------------------------------------
    n_cores = 32
    rows_per_core = rows // n_cores

    # ------------------------------------------------------------
    # Tiling Strategy (column tiling)
    # ------------------------------------------------------------
    # if columns too long → tile them
    max_tile_len = 8192          # user-defined UB capacity
    tile_length = max_tile_len
    n_tiles = (cols + tile_length - 1) // tile_length

    softmax_kernel[n_cores](
        x, output,
        rows_per_core,
        tile_length,
        n_tiles
    )

