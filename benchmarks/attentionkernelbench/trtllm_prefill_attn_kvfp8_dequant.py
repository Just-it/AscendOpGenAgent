import torch
import torch.nn as nn


class Model(nn.Module):
    """TRT-LLM Prefill Attention KV FP8 Dequant Kernel for vLLM
    
    用于 vLLM 的 KV Cache FP8 反量化，在 prefill 阶段将 FP8 编码的 KV Cache
    反量化为 float16 格式。
    """
    
    def __init__(self, num_blocks: int, num_kv_heads: int, head_dim: int, 
                 max_seq_len: int, block_size: int):
        super(Model, self).__init__()
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        
    def forward(self, kv_cache: torch.Tensor, block_tables: torch.Tensor,
                k_scale: torch.Tensor, v_scale: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kv_cache: [num_blocks, 2, num_kv_heads, block_size, head_dim] - FP8 编码的 KV Cache
            block_tables: [batch_size, max_blocks_per_seq] - Block 映射表
            k_scale: scalar - K 反量化缩放因子
            v_scale: scalar - V 反量化缩放因子
            
        Returns:
            mock_kv_cache: [batch_size * max_blocks_per_seq + 1, 2, num_kv_heads, block_size, head_dim]
                          - 反量化后的 KV Cache (float16)
        """
        batch_size = block_tables.shape[0]
        max_blocks_per_seq = block_tables.shape[1]
        num_kv_heads = kv_cache.shape[2]
        block_size = kv_cache.shape[3]
        head_dim = kv_cache.shape[4]
        
        # Create output tensor (float16)
        output_size = batch_size * max_blocks_per_seq + 1
        mock_kv_cache = torch.zeros(
            output_size, 2, num_kv_heads, block_size, head_dim,
            dtype=torch.float16, device=kv_cache.device
        )
        
        # K_CACHE_STRIDE = num_kv_heads * block_size * head_dim
        # KV_CACHE_STRIDE = 2 * num_kv_heads * block_size * head_dim
        k_cache_stride = num_kv_heads * block_size * head_dim
        kv_cache_stride = 2 * k_cache_stride
        
        # Process each batch and block
        for batch_idx in range(batch_size):
            for mock_block_idx in range(max_blocks_per_seq):
                # Get original page number from block table
                orig_page_num = int(block_tables[batch_idx, mock_block_idx].item())
                
                # Skip if orig_page_num <= 0
                if orig_page_num <= 0:
                    continue
                
                # Calculate output index (batch_idx * max_blocks_per_seq + mock_block_idx + 1)
                out_idx = batch_idx * max_blocks_per_seq + mock_block_idx + 1
                
                # Dequantize K
                # K is stored at kv_cache[orig_page_num, 0, :, :, :]
                k_fp8 = kv_cache[orig_page_num, 0, :, :, :]  # [num_kv_heads, block_size, head_dim]
                k_dequant = (k_fp8.float() * k_scale).to(torch.float16)
                mock_kv_cache[out_idx, 0, :, :, :] = k_dequant
                
                # Dequantize V
                # V is stored at kv_cache[orig_page_num, 1, :, :, :]
                v_fp8 = kv_cache[orig_page_num, 1, :, :, :]  # [num_kv_heads, block_size, head_dim]
                v_dequant = (v_fp8.float() * v_scale).to(torch.float16)
                mock_kv_cache[out_idx, 1, :, :, :] = v_dequant
        
        return mock_kv_cache


def get_inputs():
    """返回 forward() 的输入参数列表"""
    num_blocks = 1001
    num_kv_heads = 16
    block_size = 32
    head_dim = 32
    batch_size = 128
    max_blocks_per_seq = 16
    
    # KV Cache: [num_blocks, 2(K/V), num_kv_heads, block_size, head_dim]
    kv_cache = torch.randn(num_blocks, 2, num_kv_heads, block_size, head_dim, dtype=torch.float32)
    
    # Block tables: [batch_size, max_blocks_per_seq]
    # Use random valid block indices (1 to num_blocks-1), 0 or negative means invalid
    block_tables = torch.randint(1, num_blocks, (batch_size, max_blocks_per_seq), dtype=torch.int32)
    
    # Scale factors
    k_scale = torch.tensor(0.5, dtype=torch.float32)
    v_scale = torch.tensor(0.5, dtype=torch.float32)
    
    return [kv_cache, block_tables, k_scale, v_scale]


def get_init_inputs():
    """返回 __init__() 的初始化参数列表"""
    num_blocks = 1001
    num_kv_heads = 16
    head_dim = 32
    max_seq_len = 512
    block_size = 32
    
    return [num_blocks, num_kv_heads, head_dim, max_seq_len, block_size]
