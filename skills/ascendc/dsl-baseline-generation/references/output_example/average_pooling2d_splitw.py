import tile.language as tl
import torch

@ascend_kernel
def avgpool2d_kernel(
    input_ptr, output_ptr,
    batch_size, height, width, channels,
    out_h, out_w,
    kernel_h, kernel_w,
    tile_w,
    tasks_per_core
):

    pid = tl.program_id(0)

    task_start = pid * tasks_per_core
    task_end   = task_start + tasks_per_core

    # UB:
    #  x_ub  : tw * C  (一次load)
    #  sum_ub: C
    #  out_ub: C
    x_ub   = tl.alloc_ub(tile_w * channels, dtype=tl.float32)
    sum_ub = tl.alloc_ub(channels,            dtype=tl.float32)
    out_ub = tl.alloc_ub(channels,            dtype=tl.float32)

    kernel_area = kernel_h * kernel_w

    for task_id in range(task_start, task_end):

        # decode (b, oh, ow)
        b  = task_id // (out_h * out_w)
        tmp = task_id %  (out_h * out_w)
        oh = tmp // out_w
        ow = tmp %  out_w

        h0 = oh * kernel_h
        w0 = ow * kernel_w

        # sum_ub = 0
        with tl.compute():
            tl.vconst(sum_ub, 0.0)

        # accumulate kernel window
        for kh in range(kernel_h):
            ih = h0 + kh

            # kw 分块
            for kw0 in range(0, kernel_w, tile_w):

                tw = min(tile_w, kernel_w - kw0)

                # ----------------------------
                # 一次 load tw * channels
                # ----------------------------
                base = (
                    b * height * width * channels +
                    ih * width * channels +
                    (w0 + kw0) * channels
                )
                offsets = base + tl.arange(0, tw * channels)

                with tl.copyin():
                    tl.load(input_ptr + offsets, x_ub)  # count = tw*C 默认

                # ----------------------------
                # 按行（channels）累加 tw 次
                # x_ub 按 [tw][channels] 组织
                # ----------------------------
                with tl.compute():
                    for t in range(tw):
                        tl.vadd(sum_ub,
                                sum_ub,
                                x_ub[t * channels],
                                channels)     # count = channels

        # divide
        with tl.compute():
            tl.vdiv_scalar(out_ub, sum_ub, kernel_area)

        # store
        out_base = (
            b * out_h * out_w * channels +
            oh * out_w * channels +
            ow * channels
        )
        out_offsets = out_base + tl.arange(0, channels)

        with tl.copyout():
            tl.store(output_ptr + out_offsets, out_ub)

def avgpool2d_host(x: torch.Tensor, output: torch.Tensor, kernel_size):

    # Input is NHWC
    batch, height, width, channels = x.shape
    kernel_h = kernel_size
    kernel_w = kernel_size

    # Output size
    out_h = height // kernel_h
    out_w = width // kernel_w

    # ================================================
    # 1. Core Partitioning
    # ================================================
    n_cores = 16
    total_tasks = batch * out_h * out_w

    assert total_tasks % n_cores == 0, "Workload must be divisible"
    tasks_per_core = total_tasks // n_cores

    # ================================================
    # 2. Tiling Strategy
    # ================================================
    # tile_w：一次处理多少个宽度上的元素
    # 不会影响 sum_ub UB 占用（固定 channels）
    # x_ub = tile_w * channels
    tile_w = 3    

    # ================================================
    # 3. Kernel Launch
    # ================================================
    avgpool2d_kernel[n_cores](
        x, output,
        batch, height, width, channels,
        out_h, out_w,
        kernel_h, kernel_w,
        tile_w,
        tasks_per_core
    )
