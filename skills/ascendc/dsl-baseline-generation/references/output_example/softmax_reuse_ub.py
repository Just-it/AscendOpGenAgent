import tile.language as tl

@ascend_kernel
def softmax_kernel(input_ptr, output_ptr, rows_per_core, tile_length):
    pid = tl.program_id(0)

    # Each core processes its assigned rows
    row_start_idx = pid * rows_per_core
    row_end_idx   = row_start_idx + rows_per_core

    # Allocate UB Buffers
    row_ub    = tl.alloc_ub(tile_length, dtype=tl.float32)   # original row — must NOT be polluted
    exp_ub    = tl.alloc_ub(tile_length, dtype=tl.float32)   # exp(row - max)
    shared_ub = tl.alloc_ub(tile_length, dtype=tl.float32)   # reduction workspace

    # Computation Logic
    for row_idx in range(row_start_idx, row_end_idx):

        # Compute offsets for this row
        offsets = row_idx * tile_length + tl.arange(0, tile_length)

        # -------------------------
        # Copy row into UB
        # -------------------------
        with tl.copyin():
            tl.load(input_ptr + offsets, row_ub)

        # -------------------------
        # Compute softmax
        # -------------------------
        with tl.compute():
            # --- Pass 1: max reduction ---
            tl.reduce_max(shared_ub, row_ub, shared_ub)
            row_max = tl.extract_scalar(shared_ub, 0)

            # --- Pass 2: exp(row - max) ---
            tl.vsub_scalar(exp_ub, row_ub, row_max)
            tl.vexp(exp_ub, exp_ub)

            # --- Pass 3: sum reduction ---
            tl.reduce_sum(shared_ub, exp_ub, shared_ub)
            row_sum = tl.extract_scalar(shared_ub, 0)

            # --- Pass 4: normalize ---
            tl.vdiv_scalar(exp_ub, exp_ub, row_sum)

        # -------------------------
        # Store result from UB
        # -------------------------
        with tl.copyout():
            tl.store(output_ptr + offsets, exp_ub)


def softmax_host(x: torch.Tensor, output: torch.Tensor):
    rows = x.shape[0]
    cols = x.shape[1]

    # Core Partitioning
    n_cores = 16
    rows_per_core = rows // n_cores

    # Tiling Strategy (row fits UB)
    tile_length = cols

    softmax_kernel[n_cores](x, output, rows_per_core, tile_length)


