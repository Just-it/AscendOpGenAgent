import tile.language as tl
import torch


# ============================================================
# 2. KERNEL FUNCTION
# ============================================================
@ascend_kernel
def cumsum_kernel(
    input_ptr, output_ptr,
    scan_len,
    inner_size,
    tile_inner,
    tasks_per_core
):
    """
    Kernel computes inclusive prefix sum along scan axis (axis=0).
    """

    pid = tl.program_id(0)

    # ------------------------------------------------------------
    # Per-core task assignment
    # Each task handles one inner tile
    # ------------------------------------------------------------
    task_start = pid * tasks_per_core
    task_end   = task_start + tasks_per_core

    # ------------------------------------------------------------
    # 2.1 Allocate UB Buffers
    # ------------------------------------------------------------
    x_ub   = tl.alloc_ub(tile_inner, dtype=tl.float32)
    acc_ub = tl.alloc_ub(tile_inner, dtype=tl.float32)
    out_ub = tl.alloc_ub(tile_inner, dtype=tl.float32)

    # ============================================================
    # 2.2 Computation Logic
    # ============================================================
    for task_id in range(task_start, task_end):

        # Decode task_id → inner offset
        inner_base = task_id * tile_inner

        # --------------------------------------------------------
        # acc_ub = 0 (initialize prefix accumulator)
        # --------------------------------------------------------
        with tl.compute():
            tl.duplicate(acc_ub, 0.0)

        # --------------------------------------------------------
        # Sequential scan along scan axis
        # --------------------------------------------------------
        for i in range(scan_len):

            base = i * inner_size + inner_base
            offsets = base + tl.arange(0, tile_inner)

            # Load x[i, inner_slice]
            with tl.copyin():
                tl.load(input_ptr + offsets, x_ub)

            # acc_ub += x_ub
            with tl.compute():
                tl.vadd(acc_ub, acc_ub, x_ub)
                tl.vadd_scalar(out_ub, acc_ub, 0.0)

            with tl.copyout():
                tl.store(output_ptr + offsets, out_ub)


def cumsum_host(x: torch.Tensor, output: torch.Tensor):
    """
    Host Function:
    - Core partitioning
    - Tiling strategy
    - Kernel launch
    """

    # Input already permuted: scan axis is axis 0
    scan_len = x.shape[0]
    inner_size = x.numel() // scan_len

    # ============================================================
    # 1.1 Core Partitioning
    # ============================================================
    n_cores = 16

    # ============================================================
    # 1.2 Tiling Strategy
    # ============================================================
    # UB buffers:
    #   x_ub   : tile_inner
    #   acc_ub : tile_inner
    #   out_ub : tile_inner
    #
    # tile_inner chosen to fit UB capacity
    max_tile_inner = 1024
    tile_inner = min(max_tile_inner, inner_size)

    # ------------------------------------------------------------
    # Task definition: one task = one tile_inner slice
    # ------------------------------------------------------------
    total_tasks = (inner_size + tile_inner - 1) // tile_inner
    assert total_tasks % n_cores == 0, "Require evenly divisible workload"

    tasks_per_core = total_tasks // n_cores

    # ============================================================
    # Kernel Launch
    # ============================================================
    cumsum_kernel[n_cores](
        x, output,
        scan_len,
        inner_size,
        tile_inner,
        tasks_per_core
    )
