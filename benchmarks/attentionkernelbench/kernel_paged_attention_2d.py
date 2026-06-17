import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Paged Attention 2D VLLM 算子
    用于大模型推理中的分页注意力计算
    """
    def __init__(self, num_query_heads: int, num_kv_heads: int, head_size: int, 
                 num_blks: int, blk_size: int, max_num_blocks_per_seq: int,
                 num_seqs: int, scale: float, block_table_stride: int,
                 query_stride_0: int, query_stride_1: int,
                 output_stride_0: int, output_stride_1: int,
                 stride_k_cache_0: int, stride_k_cache_1: int, 
                 stride_k_cache_2: int, stride_k_cache_3: int, stride_k_cache_4: int,
                 stride_v_cache_0: int, stride_v_cache_1: int, 
                 stride_v_cache_2: int, stride_v_cache_3: int,
                 x: int = 8):
        super(Model, self).__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.num_blks = num_blks
        self.blk_size = blk_size
        self.max_num_blocks_per_seq = max_num_blocks_per_seq
        self.num_seqs = num_seqs
        self.scale = scale
        self.block_table_stride = block_table_stride
        self.query_stride_0 = query_stride_0
        self.query_stride_1 = query_stride_1
        self.output_stride_0 = output_stride_0
        self.output_stride_1 = output_stride_1
        self.stride_k_cache_0 = stride_k_cache_0
        self.stride_k_cache_1 = stride_k_cache_1
        self.stride_k_cache_2 = stride_k_cache_2
        self.stride_k_cache_3 = stride_k_cache_3
        self.stride_k_cache_4 = stride_k_cache_4
        self.stride_v_cache_0 = stride_v_cache_0
        self.stride_v_cache_1 = stride_v_cache_1
        self.stride_v_cache_2 = stride_v_cache_2
        self.stride_v_cache_3 = stride_v_cache_3
        self.x = x
        
        self.num_queries_per_kv = num_query_heads // num_kv_heads

    def forward(self, query_ptr: torch.Tensor, key_cache_ptr: torch.Tensor, 
                value_cache_ptr: torch.Tensor, block_tables_ptr: torch.Tensor,
                seq_lens_ptr: torch.Tensor, alibi_slopes_ptr: torch.Tensor,
                query_start_len_ptr: torch.Tensor) -> torch.Tensor:
        """
        执行分页注意力计算
        
        Args:
            query_ptr: [num_tokens, num_query_heads, head_size]
            key_cache_ptr: [num_blks, num_kv_heads, head_size // x, blk_size, x]
            value_cache_ptr: [num_blks, num_kv_heads, head_size, blk_size]
            block_tables_ptr: [num_seqs, max_num_blocks_per_seq]
            seq_lens_ptr: [num_seqs]
            alibi_slopes_ptr: [num_query_heads]
            query_start_len_ptr: [num_seqs + 1]
        
        Returns:
            output_ptr: [num_tokens, num_query_heads, head_size]
        """
        num_tokens = query_ptr.shape[0]
        output_ptr = torch.zeros_like(query_ptr)
        
        # 对每个序列进行处理
        for seq_idx in range(self.num_seqs):
            # 获取当前序列的查询长度和起始位置
            cur_batch_in_all_start_index = query_start_len_ptr[seq_idx].item()
            cur_batch_in_all_stop_index = query_start_len_ptr[seq_idx + 1].item()
            cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
            
            # 如果查询长度大于1，跳过（这是kernel中的filter_by_query_len逻辑）
            if cur_batch_query_len > 1:
                continue
            
            # 获取当前序列长度
            seq_len = seq_lens_ptr[seq_idx].item()
            
            # 对每个KV头进行处理
            for kv_head_idx in range(self.num_kv_heads):
                # 处理该KV头对应的所有查询头
                for q_idx in range(self.num_queries_per_kv):
                    query_head_idx = kv_head_idx * self.num_queries_per_kv + q_idx
                    if query_head_idx >= self.num_query_heads:
                        continue
                    
                    # 获取查询向量
                    query_offset = (cur_batch_in_all_start_index * self.query_stride_0 + 
                                   query_head_idx * self.query_stride_1)
                    Q = query_ptr[cur_batch_in_all_start_index, query_head_idx, :]
                    
                    # 初始化softmax统计量
                    M = float('-inf')
                    L = 0.0
                    acc = torch.zeros(self.head_size, dtype=torch.float32, device=query_ptr.device)
                    
                    # 获取alibi斜率
                    alibi_slope = alibi_slopes_ptr[query_head_idx].item()
                    
                    # 计算需要处理的块数
                    num_blocks = (seq_len + self.blk_size - 1) // self.blk_size
                    
                    # 遍历所有块
                    for block_idx in range(num_blocks):
                        start_n = block_idx * self.blk_size
                        
                        # 获取物理块索引
                        block_table_offset = seq_idx * self.block_table_stride
                        
                        # 对每个token位置进行处理
                        for token_offset in range(self.blk_size):
                            abs_token_idx = start_n + token_offset
                            if abs_token_idx >= seq_len:
                                break
                            
                            # 计算逻辑块索引和物理块索引
                            l_block_idx = abs_token_idx // self.blk_size
                            p_block_idx = block_tables_ptr[seq_idx, l_block_idx].item()
                            internal_offset = abs_token_idx % self.blk_size
                            
                            # 从key cache加载K向量
                            # K shape: [head_size], 从5D tensor中加载
                            K = torch.zeros(self.head_size, dtype=key_cache_ptr.dtype, device=key_cache_ptr.device)
                            for d in range(self.head_size):
                                d0 = d // self.x
                                d1 = d % self.x
                                K[d] = key_cache_ptr[p_block_idx, kv_head_idx, d0, internal_offset, d1]
                            K = K.to(torch.float32)
                            
                            # 从value cache加载V向量
                            # V shape: [head_size], 从4D tensor中加载
                            V = value_cache_ptr[p_block_idx, kv_head_idx, :, internal_offset].to(torch.float32)
                            
                            # 计算QK
                            qk = self.scale * torch.dot(Q.to(torch.float32), K)
                            
                            # 应用alibi bias
                            context_len = seq_len - 1
                            qk = qk + alibi_slope * (abs_token_idx - context_len)
                            
                            # 更新softmax统计量
                            m_j = max(M, qk.item())
                            
                            # 计算概率
                            p = math.exp(qk.item() - m_j)
                            if m_j == float('-inf'):
                                p = 0.0
                            
                            # 更新归一化因子
                            l_j = p
                            
                            # 计算alpha
                            alpha = math.exp(M - m_j) if M != float('-inf') else 0.0
                            
                            # 更新累加器
                            acc = acc * alpha + p * V
                            
                            # 更新统计量
                            L = L * alpha + l_j
                            M = m_j
                    
                    # 归一化输出
                    if L > 0:
                        acc = acc / (L + 1e-10)
                    
                    # 存储输出
                    output_ptr[cur_batch_in_all_start_index, query_head_idx, :] = acc.to(query_ptr.dtype)
        
        return output_ptr


def get_inputs():
    """
    返回forward函数的输入参数列表
    基于实际测试数据: num_seqs=4, num_query_heads=32, num_kv_heads=8, head_size=128
    """
    # 从实际数据加载
    import os
    data_path = '/mnt/w00934874/agent/code/AscendOpGenAgent/benchmarks/attention/kernel_paged_attention_2d_vllm/kernel_paged_attention_2d_v2.pt'
    
    if os.path.exists(data_path):
        data = torch.load(data_path, map_location='cpu')
        input_data = data['input_data']
        
        query_ptr = input_data['query_ptr']
        key_cache_ptr = input_data['key_cache_ptr']
        value_cache_ptr = input_data['value_cache_ptr']
        block_tables_ptr = input_data['block_tables_ptr']
        seq_lens_ptr = input_data['seq_lens_ptr']
        alibi_slopes_ptr = input_data['alibi_slopes_ptr']
        query_start_len_ptr = input_data['query_start_len_ptr']
        
        return [query_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr, 
                seq_lens_ptr, alibi_slopes_ptr, query_start_len_ptr]
    else:
        # 如果数据文件不存在，使用默认配置生成
        num_seqs = 4
        num_tokens = 4
        num_query_heads = 32
        num_kv_heads = 8
        head_size = 128
        num_blks = 23
        blk_size = 32
        max_num_blocks_per_seq = 6
        
        query_ptr = torch.randn(num_tokens, num_query_heads, head_size, dtype=torch.float16)
        key_cache_ptr = torch.randn(num_blks, num_kv_heads, head_size // 8, blk_size, 8, dtype=torch.float16)
        value_cache_ptr = torch.randn(num_blks, num_kv_heads, head_size, blk_size, dtype=torch.float16)
        block_tables_ptr = torch.randint(0, num_blks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)
        seq_lens_ptr = torch.tensor([32, 32, 32, 32], dtype=torch.int32)
        alibi_slopes_ptr = torch.randn(num_query_heads, dtype=torch.float32)
        query_start_len_ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        
        return [query_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr, 
                seq_lens_ptr, alibi_slopes_ptr, query_start_len_ptr]


def get_init_inputs():
    """
    返回__init__函数的初始化参数列表
    """
    num_query_heads = 32
    num_kv_heads = 8
    head_size = 128
    num_blks = 23
    blk_size = 32
    max_num_blocks_per_seq = 6
    num_seqs = 4
    scale = 0.08838834764831843
    block_table_stride = 6
    query_stride_0 = 4096
    query_stride_1 = 128
    output_stride_0 = 4096
    output_stride_1 = 128
    stride_k_cache_0 = 32768
    stride_k_cache_1 = 4096
    stride_k_cache_2 = 256
    stride_k_cache_3 = 8
    stride_k_cache_4 = 1
    stride_v_cache_0 = 32768
    stride_v_cache_1 = 4096
    stride_v_cache_2 = 32
    stride_v_cache_3 = 1
    x = 8
    
    return [num_query_heads, num_kv_heads, head_size, num_blks, blk_size, 
            max_num_blocks_per_seq, num_seqs, scale, block_table_stride,
            query_stride_0, query_stride_1, output_stride_0, output_stride_1,
            stride_k_cache_0, stride_k_cache_1, stride_k_cache_2, 
            stride_k_cache_3, stride_k_cache_4,
            stride_v_cache_0, stride_v_cache_1, stride_v_cache_2, stride_v_cache_3,
            x]
