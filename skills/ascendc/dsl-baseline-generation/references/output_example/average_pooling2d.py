import tile.language as tl
import torch

@ascend_kernel
def avgpool2d_kernel(
    input_ptr, output_ptr,
    batch_size, height, width, channels,
    out_h, out_w,
    kernel_h, kernel_w,
    tile_c,
    tasks_per_core
):

    pid = tl.program_id(0)

    # ------------------------------------------------------------
    # Per-core task range
    # host guarantees perfect division
    # ------------------------------------------------------------
    task_start = pid * tasks_per_core
    task_end   = task_start + tasks_per_core

    # ------------------------------------------------------------
    # 2.1 Allocate UB Buffers
    # ------------------------------------------------------------
    x_ub   = tl.alloc_ub(tile_c, dtype=tl.float32)     # input slice
    sum_ub = tl.alloc_ub(tile_c, dtype=tl.float32)     # accumulation
    out_ub = tl.alloc_ub(tile_c, dtype=tl.float32)     # final output

    kernel_area = kernel_h * kernel_w

    # ============================================================
    # 2.2 Computation Logic
    # ============================================================
    for task_id in range(task_start, task_end):

        # Decode task_id → (b, oh, ow)
        b  = task_id // (out_h * out_w)
        tmp = task_id %  (out_h * out_w)
        oh = tmp // out_w
        ow = tmp %  out_w

        # pool window start
        h0 = oh * kernel_h
        w0 = ow * kernel_w

        # Process channels in vector tiles
        for c0 in range(0, channels, tile_c):

            # ----------------------------------------------------
            # sum_ub = 0
            # ----------------------------------------------------
            with tl.compute():
                tl.vconst(sum_ub, 0.0)

            # ----------------------------------------------------
            # Accumulate kernel window
            # sum_ub += all x[b, ih, iw, c_slice]
            # ----------------------------------------------------
            for kh in range(kernel_h):
                ih = h0 + kh
                for kw in range(kernel_w):
                    iw = w0 + kw

                    base = (
                        b  * height * width * channels +
                        ih * width * channels +
                        iw * channels +
                        c0
                    )
                    offsets = base + tl.arange(0, tile_c)

                    # Load one tile into x_ub
                    with tl.copyin():
                        tl.load(input_ptr + offsets, x_ub)

                    # Accumulate
                    with tl.compute():
                        tl.vadd(sum_ub, sum_ub, x_ub)

            # ----------------------------------------------------
            # out_ub = sum_ub / kernel_area
            # ----------------------------------------------------
            with tl.compute():
                tl.vdiv_scalar(out_ub, sum_ub, kernel_area)

            # ----------------------------------------------------
            # Store result y[b, oh, ow, c_slice]
            # ----------------------------------------------------
            out_base = (
                b  * out_h * out_w * channels +
                oh * out_w * channels +
                ow * channels +
                c0
            )
            out_offsets = out_base + tl.arange(0, tile_c)

            with tl.copyout():
                tl.store(output_ptr + out_offsets, out_ub)



def avgpool2d_host(x: torch.Tensor, output: torch.Tensor, kernel_size):
    """
    Host Function:
    - Core partitioning
    - Tiling strategy
    - Launch kernel
    """

    # Input shape: NHWC
    batch, height, width, channels = x.shape
    kernel_h = kernel_size
    kernel_w = kernel_size

    out_h = height // kernel_h
    out_w = width // kernel_w

    # ============================================================
    # 1.1 Core Partitioning
    # ============================================================
    n_cores = 16

    total_tasks = batch * out_h * out_w
    assert total_tasks % n_cores == 0, "Require evenly divisible workload"

    tasks_per_core = total_tasks // n_cores

    # ============================================================
    # 1.2 Tiling Strategy
    # ============================================================
    # UB capacity constraint: need 3 buffers (x_ub, sum_ub, out_ub)
    # So tile_c should be reasonably small 
    # Each tile handles tile_c contiguous channels.
    max_tile_c = 1024
    tile_c = min(max_tile_c, channels)

    # Kernel launch
    avgpool2d_kernel[n_cores](
        x, output,
        batch, height, width, channels,
        out_h, out_w,
        kernel_h, kernel_w,
        tile_c,
        tasks_per_core
    )
