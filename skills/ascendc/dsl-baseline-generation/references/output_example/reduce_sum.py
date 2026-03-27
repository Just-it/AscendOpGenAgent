import tile.language as tl

@ascend_kernel
def reduce_sum_kernel(
    input_ptr,                # [N]
    partial_gm,               # [n_cores]
    output_ptr,               # scalar
    elems_per_core,
    tile_size,
    inner_loops
):
    pid = tl.program_id(0)
    start = pid * elems_per_core

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    x_ub    = tl.alloc_ub(tile_size, dtype=tl.float32)
    accum_ub = tl.alloc_ub(tile_size, dtype=tl.float32)   # reused buffer
    shared_ub = tl.alloc_ub(tile_size, dtype=tl.float32)  # reduction workspace

    # ------------------------------------------------------------
    # Phase 1: Per-core partial reduction
    # ------------------------------------------------------------
    partial_sum = 0.0

    for i in range(inner_loops):
        tile_start = start + i * tile_size
        offsets = tile_start + tl.arange(0, tile_size)

        # -------------------------------
        # COPYIN
        # -------------------------------
        with tl.copyin():
            tl.load(input_ptr + offsets, x_ub)

        # -------------------------------
        # COMPUTE (reduce tile)
        # -------------------------------
        with tl.compute():
            tl.reduce_sum(accum_ub, x_ub, shared_ub)
            tile_sum = extract_scalar(accum_ub, 0)
            partial_sum = partial_sum + tile_sum

    # ------------------------------------------------------------
    # Write per-core partial sum to GM
    # ------------------------------------------------------------
    with tl.copyout():
        tl.set_scalar(partial_gm, pid, partial_sum)

    # ------------------------------------------------------------
    # Phase 2: Core 0 performs final reduction
    # ------------------------------------------------------------
    if pid == 0:

        # Load partial results from GM into UB
        with tl.copyin():
            tl.load(partial_gm + tl.arange(0, tl.num_programs(0)), accum_ub)

        # Final reduction
        with tl.compute():
            tl.reduce_sum(shared_ub, accum_ub, shared_ub)
            final_sum = extract_scalar(shared_ub, 0)

        # Copy out final sum to output scalar
        with tl.copyout():
            tl.set_scalar(output_ptr, 0, final_sum)

def reduce_sum_host(x: torch.Tensor, output: torch.Tensor):
    total_elems = x.numel()

    # ------------------------------------------------------------
    # Core Partitioning
    # ------------------------------------------------------------
    n_cores = 16
    elems_per_core = total_elems // n_cores

    # Allocate GM tensor for per-core partial sums
    partial_gm = torch.empty(n_cores, dtype=torch.float32, device=x.device)

    # ------------------------------------------------------------
    # Tiling Strategy
    # ------------------------------------------------------------
    tile_size = 2048
    inner_loops = elems_per_core // tile_size

    # ------------------------------------------------------------
    # Launch kernel
    # ------------------------------------------------------------
    reduce_sum_kernel[n_cores](
        x,
        partial_gm,
        output,            # scalar output
        elems_per_core,
        tile_size,
        inner_loops
    )
