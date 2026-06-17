import torch
import torch.nn as nn


class Model(nn.Module):
    """
    _jagged_flash_attention_bwd_preprocess_basic_kernel - Jagged Flash Attention Backward Preprocess Kernel
    
    This kernel computes the delta term for jagged flash attention backward pass.
    Delta is computed as the sum of element-wise multiplication of O and dO along the head dimension.
    
    The kernel handles variable-length sequences (jagged tensors) using offset pointers.
    """
    def __init__(
        self,
        D: int,
        BLOCK_SIZE_M: int,
        BLOCK_SIZE_D: int,
    ):
        super(Model, self).__init__()
        self.D = D
        self.BLOCK_SIZE_M = BLOCK_SIZE_M
        self.BLOCK_SIZE_D = BLOCK_SIZE_D

    def forward(
        self,
        o_ptr: torch.Tensor,
        o_offset_ptr: torch.Tensor,
        do_ptr: torch.Tensor,
        delta_ptr: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """
        Forward implementation of _jagged_flash_attention_bwd_preprocess_basic_kernel
        
        Args:
            o_ptr: [total_tokens, D] float32 - output tensor from forward pass
            o_offset_ptr: [batch_size + 1] int64 - offset pointers for jagged tensor
            do_ptr: [total_tokens, D] float32 - gradient of output
            delta_ptr: [total_tokens] float32 - output buffer for delta (pre-allocated)
            max_seq_len: maximum sequence length to process
            
        Returns:
            delta_ptr: computed delta values
        """
        D = self.D
        batch_size = o_offset_ptr.shape[0] - 1
        
        # Process each batch
        for pid_batch in range(batch_size):
            begin_o = o_offset_ptr[pid_batch].item()
            end_o = o_offset_ptr[pid_batch + 1].item()
            
            M = min(end_o - begin_o, max_seq_len)
            
            # Process each block of tokens
            num_blocks_m = (M + self.BLOCK_SIZE_M - 1) // self.BLOCK_SIZE_M
            
            for pid_m in range(num_blocks_m):
                start_m = pid_m * self.BLOCK_SIZE_M
                end_m = min(start_m + self.BLOCK_SIZE_M, M)
                
                # Compute delta for this block
                for m in range(start_m, end_m):
                    # delta[m] = sum(o[m, :] * do[m, :])
                    o_row = o_ptr[begin_o + m, :D]
                    do_row = do_ptr[begin_o + m, :D]
                    delta_ptr[begin_o + m] = torch.sum(o_row * do_row)
        
        return delta_ptr


def get_inputs():
    """Return forward() input arguments."""
    total_tokens = 84
    batch_size = 4
    D = 128
    
    # o_ptr: [total_tokens, D] float32
    o_ptr = torch.randn(total_tokens, D, dtype=torch.float32)
    
    # o_offset_ptr: [batch_size + 1] int64 - offset pointers
    # Example: [0, 21, 42, 63, 84] for 4 batches with 21 tokens each
    o_offset_ptr = torch.tensor([0, 21, 42, 63, 84], dtype=torch.int64)
    
    # do_ptr: [total_tokens, D] float32
    do_ptr = torch.randn(total_tokens, D, dtype=torch.float32)
    
    # delta_ptr: [total_tokens] float32 (output buffer)
    delta_ptr = torch.zeros(total_tokens, dtype=torch.float32)
    
    # max_seq_len
    max_seq_len = 64
    
    return [
        o_ptr,
        o_offset_ptr,
        do_ptr,
        delta_ptr,
        max_seq_len,
    ]


def get_init_inputs():
    """Return __init__() arguments."""
    D = 128
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_D = 128
    return [D, BLOCK_SIZE_M, BLOCK_SIZE_D]
