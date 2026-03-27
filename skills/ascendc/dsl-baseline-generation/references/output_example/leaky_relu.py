import tile.language as tl

@ascend_kernel
def leaky_relu_kernel(input_ptr, output_ptr,
                      elements_per_core, tile_size, inner_loops, negative_slope):

    pid = tl.program_id(0)
    start = pid * elements_per_core

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
    n_cores = 16
    elements_per_core = total_elems // n_cores

    # ------------------------------------------------------------
    # Tiling Strategy
    # ------------------------------------------------------------
    tile_size = 2048
    inner_loops = elements_per_core // tile_size

    # ------------------------------------------------------------
    # Launch kernel
    # ------------------------------------------------------------
    leaky_relu_kernel[n_cores](
        x, output,
        elements_per_core,
        tile_size,
        inner_loops,
        negative_slope
    )

