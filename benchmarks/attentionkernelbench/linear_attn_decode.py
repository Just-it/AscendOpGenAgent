import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """Linear Attention Decode Kernel for vLLM
    
    实现线性注意力解码机制，用于推理阶段的自回归生成。
    使用线性注意力近似，将 softmax(QK^T)V 转换为 Q(K^T V) 的形式，
    降低计算复杂度。支持 KV Cache 更新和 decay ratio。
    """
    
    def __init__(self, batch_size: int, num_heads: int, head_dim: int, 
                 max_seq_len: int, block_size: int):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                kv_cache: torch.Tensor, slope_rate: torch.Tensor,
                slot_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: [batch_size, num_heads, 1, head_dim] - Query tensor
            k: [batch_size, num_heads, 1, head_dim] - Key tensor  
            v: [batch_size, num_heads, 1, head_dim] - Value tensor
            kv_cache: [max_seq_len, num_heads, head_dim, head_dim] - KV state cache
            slope_rate: [num_heads] - Slope coefficients for linear attention decay
            slot_idx: [batch_size] - Slot indices for cache access (can be -1 for padding)
            
        Returns:
            output: [batch_size, num_heads, 1, head_dim] - Attention output
        """
        batch_size = q.shape[0]
        num_heads = q.shape[1]
        head_dim = q.shape[3]
        
        # Create output tensor
        output = torch.zeros_like(q)
        
        # Create a copy of kv_cache to update (in-place updates)
        updated_cache = kv_cache.clone()
        
        # Process each batch and head
        for b in range(batch_size):
            slot = int(slot_idx[b].item())
            # Skip if slot_id is -1 (padding)
            if slot == -1:
                continue
            
            for h in range(num_heads):
                # Get current q, k, v for this batch and head
                q_bh = q[b, h, 0, :]  # [head_dim]
                k_bh = k[b, h, 0, :]  # [head_dim]
                v_bh = v[b, h, 0, :]  # [head_dim]
                slope = slope_rate[h].item()
                
                # Get the KV state for this slot and head
                # kv_cache[slot, h] is [head_dim, head_dim] matrix
                kv_state = updated_cache[slot, h, :, :]  # [head_dim, head_dim]
                
                # Update KV state: KV_new = decay * KV_old + k^T * v
                # k_bh: [head_dim], v_bh: [head_dim]
                # Outer product: k_bh.unsqueeze(1) @ v_bh.unsqueeze(0) -> [head_dim, head_dim]
                decay = math.exp(-slope)
                kv_update = torch.outer(k_bh, v_bh)
                kv_state = decay * kv_state + kv_update
                
                # Store updated state back
                updated_cache[slot, h, :, :] = kv_state
                
                # Compute output: o = q @ KV_state
                # q_bh: [head_dim], kv_state: [head_dim, head_dim]
                output_bh = q_bh @ kv_state  # [head_dim]
                output[b, h, 0, :] = output_bh
        
        return output


def get_inputs():
    """返回 forward() 的输入参数列表"""
    batch_size = 4
    num_heads = 8
    head_dim = 64
    max_seq_len = 128
    
    q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float32)
    kv_cache = torch.randn(max_seq_len, num_heads, head_dim, head_dim, dtype=torch.float32)
    slope_rate = torch.randn(num_heads, dtype=torch.float32)
    slot_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    
    return [q, k, v, kv_cache, slope_rate, slot_idx]


def get_init_inputs():
    """返回 __init__() 的初始化参数列表"""
    batch_size = 4
    num_heads = 8
    head_dim = 64
    max_seq_len = 128
    block_size = 32
    
    return [batch_size, num_heads, head_dim, max_seq_len, block_size]
