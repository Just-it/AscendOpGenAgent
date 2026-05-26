import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def repeat_small_kernel(
    x_ptr, out_ptr,
    inner_size,
    r: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    模式 A: Small Grid，适合 total_blocks <= VEC_CORE_NUM。
    grid = (outer_size, num_inner_blocks)
    """
    outer_idx = tl.program_id(0).to(tl.int32)
    local_block = tl.program_id(1).to(tl.int32)

    block_start = local_block * BLOCK
    offs = (block_start + tl.arange(0, BLOCK)).to(tl.int32)
    mask = offs < inner_size

    in_offset = outer_idx * inner_size
    val = tl.load(x_ptr + in_offset + offs, mask=mask)

    # r 为 constexpr，编译期展开
    for repeat_idx in range(r):
        out_offset = outer_idx * inner_size * r + repeat_idx * inner_size
        tl.store(out_ptr + out_offset + offs, val, mask=mask)


@triton.jit
def repeat_large_kernel(
    x_ptr, out_ptr,
    outer_size, inner_size, num_inner_blocks,
    r: tl.constexpr,
    BLOCK: tl.constexpr,
    num_cores: tl.constexpr,
):
    """
    模式 B: Large Grid，适合 total_blocks > VEC_CORE_NUM。
    grid = (min(total_blocks, num_cores),)
    """
    pid = tl.program_id(0).to(tl.int32)
    total_blocks = outer_size * num_inner_blocks

    blocks_per_core = total_blocks // num_cores
    remainder = total_blocks - blocks_per_core * num_cores

    if pid < remainder:
        my_blocks = blocks_per_core + 1
        start_block = pid * (blocks_per_core + 1)
    else:
        my_blocks = blocks_per_core
        start_block = remainder * (blocks_per_core + 1) + (pid - remainder) * blocks_per_core

    for block_idx in range(start_block, start_block + my_blocks):
        outer_idx = block_idx // num_inner_blocks
        local_block = block_idx - outer_idx * num_inner_blocks

        block_start = local_block * BLOCK
        offs = (block_start + tl.arange(0, BLOCK)).to(tl.int32)
        mask = offs < inner_size

        in_offset = outer_idx * inner_size
        val = tl.load(x_ptr + in_offset + offs, mask=mask)

        for repeat_idx in range(r):
            out_offset = outer_idx * inner_size * r + repeat_idx * inner_size
            tl.store(out_ptr + out_offset + offs, val, mask=mask)


def _get_block_size(inner_size: int) -> int:
    """按 inner_size 向上取 2 的幂次，目标使 num_inner_blocks 尽量小。"""
    if inner_size <= 64:
        return 64
    if inner_size <= 128:
        return 128
    if inner_size <= 256:
        return 256
    if inner_size <= 512:
        return 512
    if inner_size <= 1024:
        return 1024
    if inner_size <= 2048:
        return 2048
    if inner_size <= 4096:
        return 4096
    return 8192


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import torch_npu
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except Exception:
            self.VEC_CORE_NUM = 40

    def forward(self, x: torch.Tensor, repeats: tuple) -> torch.Tensor:
        # L1.5: 确保输入 contiguous
        x = x.contiguous()
        shape = list(x.shape)
        ndim = len(shape)

        # 将 repeats 扩展到与 ndim 相同长度（前面补 1）
        repeats = [1] * (ndim - len(repeats)) + list(repeats)

        out = x

        # L1.4: 从最低维到最高维逐维度处理
        for dim_idx in range(ndim - 1, -1, -1):
            r = repeats[dim_idx]
            if r <= 1:
                continue

            outer_size = math.prod(shape[:dim_idx])
            inner_size = out.numel() // outer_size

            BLOCK = _get_block_size(inner_size)
            num_inner_blocks = (inner_size + BLOCK - 1) // BLOCK
            total_blocks = outer_size * num_inner_blocks

            # 构造输出 tensor
            out_shape = list(out.shape)
            out_shape[dim_idx] *= r
            output = torch.empty(out_shape, dtype=out.dtype, device=out.device)

            if total_blocks <= self.VEC_CORE_NUM:
                grid = (outer_size, num_inner_blocks)
                repeat_small_kernel[grid](
                    out, output,
                    inner_size,
                    r=r,
                    BLOCK=BLOCK,
                )
            else:
                grid_cores = total_blocks if total_blocks < self.VEC_CORE_NUM else self.VEC_CORE_NUM
                grid = (grid_cores,)
                repeat_large_kernel[grid](
                    out, output,
                    outer_size, inner_size, num_inner_blocks,
                    r=r,
                    BLOCK=BLOCK,
                    num_cores=grid_cores,
                )

            # L3.1: 更新 out 供下一轮使用
            out = output
            shape[dim_idx] *= r

        return out
