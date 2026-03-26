import tile.language as tl

@ascend_kernel
def mse_loss_kernel(
    pred_ptr,                 # [N]
    target_ptr,               # [N]
    workspace,               # [n_cores]
    output_ptr,               # final scalar
    elems_per_core,
    tile_size,
    inner_loops,
    total_elems               # for computing mean
):
    pid = tl.program_id(0)
    start = pid * elems_per_core

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    pred_ub    = tl.alloc_ub(tile_size, dtype=tl.float32)
    target_ub  = tl.alloc_ub(tile_size, dtype=tl.float32)
    diff_ub    = tl.alloc_ub(tile_size, dtype=tl.float32)
    sq_ub      = tl.alloc_ub(tile_size, dtype=tl.float32)
    shared_ub  = tl.alloc_ub(tile_size, dtype=tl.float32)
    workspace_out_ub = tl.alloc_ub(tile_size, dtype=tl.float32)
    workspace_in_ub   = tl.alloc_ub(tile_size, dtype=tl.float32)
    output_ub   = tl.alloc_ub(tile_size, dtype=tl.float32)

    

    # ------------------------------------------------------------
    # Phase 1: per-core partial reduction of squared errors
    # ------------------------------------------------------------
    partial_sum = 0.0

    for i in range(inner_loops):
        tile_start = start + i * tile_size
        offsets = tile_start + tl.arange(0, tile_size)

        # -------------------------------
        # COPYIN
        # -------------------------------
        with tl.copyin():
            tl.load(pred_ptr   + offsets, pred_ub)
            tl.load(target_ptr + offsets, target_ub)

        # -------------------------------
        # COMPUTE ((pred - target)^2)
        # -------------------------------
        with tl.compute():
            tl.vsub(diff_ub, pred_ub, target_ub)
            tl.vmul(sq_ub, diff_ub, diff_ub)
            tl.reduce_sum(sq_ub, sq_ub, shared_ub)
            tile_sum = extract_scalar(sq_ub, 0)
            partial_sum = partial_sum + tile_sum

    # ------------------------------------------------------------
    # Write per-core partial sum
    # ------------------------------------------------------------
    with tl.copyout():
        tl.set_scalar(workspace_out_ub, 0, partial_sum)
        tl.store(workspace+tl.arange(pid,pid+1), workspace_out_ub)

    # ------------------------------------------------------------
    # Phase 2: Core 0 performs final reduce + mean
    # ------------------------------------------------------------
    if pid == 0:

        # ---------------------------
        # Load all partial sums
        # ---------------------------
        with tl.copyin():
            tl.load(workspace + tl.arange(0, tl.num_programs(0)), workspace_in_ub)

        # ---------------------------
        # Reduce
        # ---------------------------
        with tl.compute():
            tl.reduce_sum(shared_ub, workspace_in_ub, shared_ub)
            sum_sq = extract_scalar(shared_ub, 0)
            mse = sum_sq / float(total_elems)
            tl.set_scalar(output_ub, 0, mse)

        # ---------------------------
        # Copyout final scalar
        # ---------------------------
        with tl.copyout():
           tl.store(output_ptr+tl.arange(0, 1), output_ub)

def mse_loss_host(pred: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    total_elems = pred.numel()

    # ------------------------------------------------------------
    # Core Partitioning
    # ------------------------------------------------------------
    n_cores = 16
    elems_per_core = total_elems // n_cores

    # GM buffer for per-core partial results
    workspace = torch.empty(n_cores, dtype=torch.float32, device=pred.device)

    # ------------------------------------------------------------
    # Tiling Strategy
    # ------------------------------------------------------------
    tile_size = 2048
    inner_loops = elems_per_core // tile_size

    # ------------------------------------------------------------
    # Launch kernel
    # ------------------------------------------------------------
    mse_loss_kernel[n_cores](
        pred,
        target,
        workspace,
        output,
        elems_per_core,
        tile_size,
        inner_loops,
        total_elems
    )

