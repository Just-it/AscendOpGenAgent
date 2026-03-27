import torch
import tile.language as tl

@ascend_kernel
def softmax_kernel(input_ptr, output_ptr, rows_per_core, tile_length):
    pid = tl.program_id(0)
    row_start_idx = pid * rows_per_core
    row_end_idx = row_start_idx + rows_per_core

    # Allocate UB Buffers 
    row_ub      = tl.alloc_ub(tile_length, dtype=tl.float32) 
    exp_ub      = tl.alloc_ub(tile_length, dtype=tl.float32) 
    shared_ub   = tl.alloc_ub(tile_length, dtype=tl.float32) 
    out_ub      = tl.alloc_ub(tile_length, dtype=tl.float32) 
    
    # Computation Logic
    for row_idx in range(row_start_idx, row_end_idx):
        offsets = row_idx * tile_length + tl.arange(0, tile_length)

        with tl.copyin():
            tl.load(input_ptr + offsets, row_ub)

        with tl.compute():
            tl.reduce_max(shared_ub, row_ub, shared_ub) 
            row_max = extract_scalar(shared_ub, 0)
            tl.vsub_scalar(exp_ub, row_ub, row_max) 
            tl.vexp(exp_ub, exp_ub) 
            tl.reduce_sum(shared_ub, exp_ub, shared_ub)
            row_sum = extract_scalar(shared_ub, 0)
            tl.vdiv_scalar(out_ub, exp_ub, row_sum)

        with tl.copyout():
            tl.store(output_ptr + offsets, out_ub)

def softmax_host(x: torch.Tensor, output: torch.Tensor):
    rows = x.shape[0]
    cols = x.shape[1]
    
    # Core Partitioning
    n_cores = 32 
    rows_per_core = rows // n_cores 

    # Tiling Strategy
    # Entire row fits into UB.
    tile_length = cols  
    
    softmax_kernel[n_cores](x, output, rows_per_core, tile_length)
