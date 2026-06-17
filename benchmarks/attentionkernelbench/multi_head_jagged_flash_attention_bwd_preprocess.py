import torch
import torch.nn as nn


class Model(nn.Module):
    """Multi-Head Jagged Flash Attention Backward Preprocess Kernel
    
    用于 FBGEMM 的变长序列 Flash Attention 反向传播预处理。
    计算 delta = sum(o * do, axis=-1)，支持 jagged 格式（变长序列）。
    """
    
    def __init__(self, num_heads: int, max_seq_len: int, head_dim: int):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        
    def forward(self, o: torch.Tensor, do: torch.Tensor, 
                o_offset: torch.Tensor) -> torch.Tensor:
        """
        Args:
            o: [num_heads, total_seq_len, head_dim] - 注意力输出
            do: [num_heads, total_seq_len, head_dim] - 输出梯度
            o_offset: [batch_size + 1] - 序列偏移量（jagged 格式）
            
        Returns:
            delta: [num_heads, total_seq_len] - 计算的 delta 值
        """
        num_heads = o.shape[0]
        total_seq_len = o.shape[1]
        head_dim = o.shape[2]
        batch_size = o_offset.shape[0] - 1
        
        # Create output tensor
        delta = torch.zeros(num_heads, total_seq_len, dtype=o.dtype, device=o.device)
        
        # Process each batch
        for b in range(batch_size):
            begin = int(o_offset[b].item())
            end = int(o_offset[b + 1].item())
            seq_len = min(end - begin, self.max_seq_len)
            
            if seq_len == 0:
                continue
            
            # Process each head
            for h in range(num_heads):
                # Get o and do for this batch and head
                o_bh = o[h, begin:begin + seq_len, :]  # [seq_len, head_dim]
                do_bh = do[h, begin:begin + seq_len, :]  # [seq_len, head_dim]
                
                # Compute delta = sum(o * do, axis=-1)
                delta_bh = torch.sum(o_bh * do_bh, dim=-1)  # [seq_len]
                
                # Store result
                delta[h, begin:begin + seq_len] = delta_bh
        
        return delta


def get_inputs():
    """返回 forward() 的输入参数列表"""
    num_heads = 8
    max_seq_len = 64
    head_dim = 32
    batch_size = 4
    
    # Simulate jagged tensor with total_seq_len = 145
    total_seq_len = 145
    
    o = torch.randn(num_heads, total_seq_len, head_dim, dtype=torch.float32)
    do = torch.randn(num_heads, total_seq_len, head_dim, dtype=torch.float32)
    # Offset array: [0, 32, 64, 96, 145] - 4 batches with varying lengths
    o_offset = torch.tensor([0, 32, 64, 96, 145], dtype=torch.int32)
    
    return [o, do, o_offset]


def get_init_inputs():
    """返回 __init__() 的初始化参数列表"""
    num_heads = 8
    max_seq_len = 64
    head_dim = 32
    
    return [num_heads, max_seq_len, head_dim]
