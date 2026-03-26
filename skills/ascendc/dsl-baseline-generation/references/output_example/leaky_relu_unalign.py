import tile.language as tl

@ascend_kernel
def leaky_relu_kernel(input_ptr, output_ptr,
                      total_elems, tile_size, negative_slope, remainder):

    pid = tl.program_id(0)

    # Calculate the number of elements assigned to this core
    elements_per_core = total_elems // 40  # Dividing equally among 40 cores
    if pid < remainder:
        elements_per_core += 1  # Distribute the remainder across the first few cores

    start = pid * elements_per_core

    # Calculate the number of inner loops based on the actual elements assigned to this core
    inner_loops = (elements_per_core + tile_size - 1) // tile_size  # Round up to handle partial tiles

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    x_ub          = tl.alloc_ub(tile_size, dtype=tl.float32)
    pos_ub        = tl.alloc_ub(tile_size, dtype=tl.float32)
    neg_ub        = tl.alloc_ub(tile_size, dtype=tl.float32)
    out_ub        = tl.alloc_ub(tile_size, dtype=tl.float32)

    # ------------------------------------------------------------
    # Tile loop
    # ------------------------------------------------------------
    for i in range(inner_loops):
        tile_start = start + i * tile_size
        offsets = tile_start + tl.arange(0, tile_size)

        # --------------------------------------------------------
        # COPYIN
        # --------------------------------------------------------
        with tl.copyin():
            tl.load(input_ptr + offsets, x_ub)

        # --------------------------------------------------------
        # COMPUTE
        # --------------------------------------------------------
        with tl.compute():
            # pos = max(x, 0)
            tl.vmax(pos_ub, x_ub, 0.0)

            # neg = min(x, 0)
            tl.vmin(neg_ub, x_ub, 0.0)

            # neg_scaled = neg * negative_slope
            tl.vmul_scalar(neg_ub, neg_ub, negative_slope)

            # out = pos + neg_scaled
            tl.vadd(out_ub, pos_ub, neg_ub)

        # --------------------------------------------------------
        # COPYOUT
        # --------------------------------------------------------
        with tl.copyout():
            tl.store(output_ptr + offsets, out_ub)


def leaky_relu_host(x: torch.Tensor, output: torch.Tensor, negative_slope: float):
    total_elems = x.numel()

    # ------------------------------------------------------------
    # Core Partitioning
    # ------------------------------------------------------------
    n_cores = 40
    elements_per_core = total_elems // n_cores
    remainder = total_elems % n_cores

    # ------------------------------------------------------------
    # Tiling Strategy
    # ------------------------------------------------------------
    tile_size = 8192  # elements per buffer, one tile fits one UB buffer size

    # ------------------------------------------------------------
    # Launch kernel with 40 cores
    # ------------------------------------------------------------
    leaky_relu_kernel[n_cores](
        x, output,
        total_elems,
        tile_size,
        negative_slope,
        remainder
    )
