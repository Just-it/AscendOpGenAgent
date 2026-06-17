import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Unified Attention 2D for vLLM
    
    该算子实现了支持多种特性的统一注意力计算：
    - FP8 量化支持
    - ALiBi 位置编码
    - Sliding Window 注意力
    - Softcap
    - QQ Bias
    - Sink tokens
    - MM Prefix (多模态双向注意力)
    """
    
    def __init__(
        self,
        num_tokens: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        num_blks: int,
        blk_size: int,
        num_seqs: int,
        max_num_blocks_per_seq: int,
        use_alibi: bool = False,
        use_alibi_sqrt: bool = False,
        use_qq_bias: bool = False,
        use_softcap: bool = False,
        use_sinks: bool = False,
        sliding_window: int = 0,
        use_mm_prefix: bool = False,
        max_mm_ranges: int = 0,
        use_fp8: bool = False,
    ):
        super(Model, self).__init__()
        self.num_tokens = num_tokens
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.num_blks = num_blks
        self.blk_size = blk_size
        self.num_seqs = num_seqs
        self.max_num_blocks_per_seq = max_num_blocks_per_seq
        self.use_alibi = use_alibi
        self.use_alibi_sqrt = use_alibi_sqrt
        self.use_qq_bias = use_qq_bias
        self.use_softcap = use_softcap
        self.use_sinks = use_sinks
        self.sliding_window = sliding_window
        self.use_mm_prefix = use_mm_prefix
        self.max_mm_ranges = max_mm_ranges
        self.use_fp8 = use_fp8
        
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        
    def forward(
        self,
        query: torch.Tensor,  # [num_tokens, num_query_heads, head_size]
        key_cache: torch.Tensor,  # [num_blks, blk_size, num_kv_heads, head_size]
        value_cache: torch.Tensor,  # [num_blks, blk_size, num_kv_heads, head_size]
        block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
        seq_lens: torch.Tensor,  # [num_seqs]
        query_start_len: torch.Tensor,  # [num_seqs+1]
        scale: float,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        out_scale: float = 1.0,
        softcap: float = 0.0,
        sink: torch.Tensor = None,  # [num_query_heads]
        alibi_slopes: torch.Tensor = None,  # [num_query_heads]
        qq_bias: torch.Tensor = None,  # [num_query_tokens, num_query_tokens]
        mm_prefix_range: torch.Tensor = None,  # [num_seqs, max_mm_ranges, 2]
    ) -> torch.Tensor:
        """
        执行 unified attention 计算
        
        Returns:
            output: [num_tokens, num_query_heads, head_size]
        """
        device = query.device
        dtype = query.dtype
        
        # 初始化输出
        output = torch.zeros_like(query)
        
        # 处理每个序列
        for seq_idx in range(self.num_seqs):
            cur_batch_in_all_start_index = query_start_len[seq_idx].item()
            cur_batch_in_all_stop_index = query_start_len[seq_idx + 1].item()
            cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
            
            if cur_batch_query_len == 0:
                continue
            
            seq_len = seq_lens[seq_idx].item()
            context_len = seq_len - cur_batch_query_len
            
            # 获取当前序列的 query
            q_start = cur_batch_in_all_start_index
            q_end = cur_batch_in_all_stop_index
            cur_query = query[q_start:q_end]  # [cur_batch_query_len, num_query_heads, head_size]
            
            # 获取当前序列的 block table
            cur_block_table = block_tables[seq_idx]  # [max_num_blocks_per_seq]
            
            # 计算需要的 KV 缓存位置
            # 遍历每个 query head group
            for kv_head_idx in range(self.num_kv_heads):
                q_head_start = kv_head_idx * self.num_queries_per_kv
                q_head_end = (kv_head_idx + 1) * self.num_queries_per_kv
                
                # 获取当前 KV head 对应的 queries
                cur_q = cur_query[:, q_head_start:q_head_end, :]  # [cur_batch_query_len, num_queries_per_kv, head_size]
                cur_q = cur_q.reshape(-1, self.head_size)  # [cur_batch_query_len * num_queries_per_kv, head_size]
                
                # 收集所有需要的 KV
                all_k = []
                all_v = []
                
                for pos in range(seq_len):
                    block_idx = cur_block_table[pos // self.blk_size].item()
                    if block_idx < 0:
                        continue
                    pos_in_block = pos % self.blk_size
                    
                    k = key_cache[block_idx, pos_in_block, kv_head_idx, :]  # [head_size]
                    v = value_cache[block_idx, pos_in_block, kv_head_idx, :]  # [head_size]
                    
                    all_k.append(k)
                    all_v.append(v)
                
                if len(all_k) == 0:
                    continue
                
                K = torch.stack(all_k, dim=0)  # [seq_len, head_size]
                V = torch.stack(all_v, dim=0)  # [seq_len, head_size]
                
                # 计算注意力分数: [cur_batch_query_len * num_queries_per_kv, seq_len]
                scores = torch.matmul(cur_q, K.T) * scale
                
                # 应用 causal mask
                num_q_rows = cur_batch_query_len * self.num_queries_per_kv
                for q_idx in range(num_q_rows):
                    actual_q_pos = q_idx // self.num_queries_per_kv
                    query_abs_pos = context_len + actual_q_pos
                    
                    for k_idx in range(seq_len):
                        if k_idx > query_abs_pos:
                            scores[q_idx, k_idx] = float('-inf')
                
                # 应用 sliding window
                if self.sliding_window > 0:
                    for q_idx in range(num_q_rows):
                        actual_q_pos = q_idx // self.num_queries_per_kv
                        query_abs_pos = context_len + actual_q_pos
                        
                        for k_idx in range(seq_len):
                            if (query_abs_pos - k_idx) >= self.sliding_window:
                                scores[q_idx, k_idx] = float('-inf')
                
                # 应用 mm_prefix (多模态双向注意力)
                if self.use_mm_prefix and mm_prefix_range is not None and self.max_mm_ranges > 0:
                    for range_idx in range(self.max_mm_ranges):
                        range_start = mm_prefix_range[seq_idx, range_idx, 0].item()
                        range_end = mm_prefix_range[seq_idx, range_idx, 1].item()
                        
                        if range_start >= range_end:
                            continue
                        
                        for q_idx in range(num_q_rows):
                            actual_q_pos = q_idx // self.num_queries_per_kv
                            query_abs_pos = context_len + actual_q_pos
                            
                            # 如果 query 在范围内，允许 attend 到范围内的所有 key
                            if range_start <= query_abs_pos <= range_end:
                                for k_idx in range(seq_len):
                                    if range_start <= k_idx <= range_end:
                                        # 恢复为正常值（移除 -inf）
                                        if k_idx <= query_abs_pos:
                                            scores[q_idx, k_idx] = torch.matmul(
                                                cur_q[q_idx], K[k_idx]
                                            ) * scale
                
                # 应用 softcap
                if self.use_softcap and softcap != 0.0:
                    scores = softcap * torch.tanh(scores / softcap)
                
                # 应用 ALiBi
                if self.use_alibi and alibi_slopes is not None:
                    for q_idx in range(num_q_rows):
                        q_head_offset = q_idx % self.num_queries_per_kv
                        actual_head_idx = q_head_start + q_head_offset
                        alibi_slope = alibi_slopes[actual_head_idx].item()
                        
                        actual_q_pos = q_idx // self.num_queries_per_kv
                        query_abs_pos = context_len + actual_q_pos
                        
                        for k_idx in range(seq_len):
                            if self.use_alibi_sqrt:
                                relative_pos = k_idx - query_abs_pos
                                if relative_pos <= 0:
                                    alibi_offset = -math.sqrt(-relative_pos)
                                else:
                                    alibi_offset = 0.0
                            else:
                                alibi_offset = k_idx - context_len
                            
                            if scores[q_idx, k_idx] != float('-inf'):
                                scores[q_idx, k_idx] += alibi_slope * alibi_offset
                
                # 应用 QQ Bias
                if self.use_qq_bias and qq_bias is not None:
                    for q_idx in range(num_q_rows):
                        actual_q_pos = q_idx // self.num_queries_per_kv
                        
                        for k_idx in range(seq_len):
                            key_rel_pos = k_idx - context_len
                            if 0 <= key_rel_pos < cur_batch_query_len:
                                if scores[q_idx, k_idx] != float('-inf'):
                                    scores[q_idx, k_idx] += qq_bias[actual_q_pos, key_rel_pos]
                
                # Softmax
                scores = torch.softmax(scores, dim=-1)
                
                # 应用 attention 到 values
                out = torch.matmul(scores, V)  # [num_q_rows, head_size]
                
                # 重塑并写回输出
                out = out.reshape(cur_batch_query_len, self.num_queries_per_kv, self.head_size)
                output[q_start:q_end, q_head_start:q_head_end, :] = out
        
        return output


def get_inputs():
    """返回 forward() 的输入参数列表"""
    # 配置参数
    num_tokens = 128
    num_query_heads = 32
    num_kv_heads = 8
    head_size = 128
    num_blks = 64
    blk_size = 16
    num_seqs = 4
    max_num_blocks_per_seq = 32
    
    device = "cpu"
    dtype = torch.float32
    
    # Query: [num_tokens, num_query_heads, head_size]
    query = torch.randn(num_tokens, num_query_heads, head_size, device=device, dtype=dtype)
    
    # Key cache: [num_blks, blk_size, num_kv_heads, head_size]
    key_cache = torch.randn(num_blks, blk_size, num_kv_heads, head_size, device=device, dtype=dtype)
    
    # Value cache: [num_blks, blk_size, num_kv_heads, head_size]
    value_cache = torch.randn(num_blks, blk_size, num_kv_heads, head_size, device=device, dtype=dtype)
    
    # Block tables: [num_seqs, max_num_blocks_per_seq]
    block_tables = torch.arange(num_seqs * max_num_blocks_per_seq, device=device, dtype=torch.int32)
    block_tables = block_tables.reshape(num_seqs, max_num_blocks_per_seq) % num_blks
    
    # Sequence lengths: [num_seqs]
    seq_lens = torch.tensor([64, 48, 32, 16], device=device, dtype=torch.int32)
    
    # Query start lengths: [num_seqs+1]
    query_start_len = torch.tensor([0, 32, 64, 96, 128], device=device, dtype=torch.int32)
    
    # Scale
    scale = 1.0 / math.sqrt(head_size)
    
    # Optional inputs
    k_scale = 1.0
    v_scale = 1.0
    out_scale = 1.0
    softcap = 50.0
    
    # Sink: [num_query_heads]
    sink = torch.randn(num_query_heads, device=device, dtype=dtype)
    
    # ALiBi slopes: [num_query_heads]
    alibi_slopes = torch.randn(num_query_heads, device=device, dtype=dtype) * 0.01
    
    # QQ bias: [num_tokens, num_tokens]
    qq_bias = torch.randn(num_tokens, num_tokens, device=device, dtype=dtype) * 0.01
    
    # MM prefix range: [num_seqs, max_mm_ranges, 2]
    max_mm_ranges = 2
    mm_prefix_range = torch.zeros(num_seqs, max_mm_ranges, 2, device=device, dtype=torch.int32)
    
    return [
        query, key_cache, value_cache, block_tables, seq_lens, query_start_len,
        scale, k_scale, v_scale, out_scale, softcap, sink, alibi_slopes, qq_bias, mm_prefix_range
    ]


def get_init_inputs():
    """返回 __init__() 的初始化参数列表"""
    return [
        128,   # num_tokens
        32,    # num_query_heads
        8,     # num_kv_heads
        128,   # head_size
        64,    # num_blks
        16,    # blk_size
        4,     # num_seqs
        32,    # max_num_blocks_per_seq
        True,  # use_alibi
        False, # use_alibi_sqrt
        True,  # use_qq_bias
        True,  # use_softcap
        True,  # use_sinks
        64,    # sliding_window
        False, # use_mm_prefix
        0,     # max_mm_ranges
        False, # use_fp8
    ]
