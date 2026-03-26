import tile.language as tl
import math
import torch

@ascend_kernel
def matmul_kernel(A_ptr, B_ptr, C_ptr,
                  M, N, K,
                  block_M, block_N, k_tile, n_blocks):
    """
    Kernel computes one (block_M x block_N) tile of C per program/core.
    Assumes no tail handling (all tiles are full).
    """

    pid = tl.program_id(0)

    # tile coordinates for this program
    pid_m = pid // n_blocks
    pid_n = pid % n_blocks

    # global tile starts
    m_start = pid_m * block_M
    n_start = pid_n * block_N
    k_blocks = K // k_tile

    # -----------------------
    # Allocate L1 and L0 buffers (all declared here)
    # -----------------------
    A_L1 = tl.alloc_l1((block_M, k_tile), dtype=tl.float32)
    B_L1 = tl.alloc_l1((k_tile, block_N), dtype=tl.float32)
    C_L0 = tl.alloc_l0c((block_M, block_N), dtype=tl.float32)

    # Main K loop: load A_L1, B_L1, compute into C_L0
    for kb in range(k_blocks):

        k_start = kb * k_tile

        with tl.copyin():
            tl.load(A_ptr + (m_start * K + k_start), A_L1)
            tl.load(B_ptr + (k_start * N + n_start), B_L1)

        with tl.compute():
            tl.gemm_v0(A_L1, B_L1, C_L0, init=(kb == 0))

    with tl.copyout():
        tl.store(C_ptr + (m_start * N + n_start), C_L0)


# -----------------------
# Host: global planning + launch
# -----------------------
def matmul_host(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
    M, K = A.shape
    K, N = B.shape

    # Core Partitioning
    # No inter-core sync required: each core writes distinct C tile
    block_M = 256
    block_N = 256

    m_blocks = M // block_M
    n_blocks = N // block_N

    n_cores = m_blocks * n_blocks

    # Tiling Strategy (Host)
    # Choose block sizes to fit your hardware L1 with safety headroom.
    k_tile  = 128

    matmul_kernel[n_cores](
        A, B, C,
        M, N, K,
        block_M, block_N, k_tile, n_blocks
    )
