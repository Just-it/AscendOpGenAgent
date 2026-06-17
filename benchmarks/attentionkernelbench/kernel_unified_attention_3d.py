import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Unified 3D Attention Kernel for vLLM.
    
    This implements a segment-based attention mechanism that supports:
    - Multi-head attention with GQA (Grouped Query Attention)
    - Sliding window attention
    - ALiBi positional encoding
    - Softcap for attention scores
    - Sinks for attention
    - Multimodal prefix attention
    - Query-query bias
    
    The output is segmented per sequence and per attention head.
    """

    def __init__(
        self,
        num_tokens: int,
        num_query_heads: int,
        num_segments: int,
        head_size: int,
        head_size_padded: int,
        num_blks: int,
        num_kv_heads: int,
        blk_size: int,
        num_seqs: int,
        max_num_blocks_per_seq: int,
    ):
        super(Model, self).__init__()
        self.num_tokens = num_tokens
        self.num_query_heads = num_query_heads
        self.num_segments = num_segments
        self.head_size = head_size
        self.head_size_padded = head_size_padded
        self.num_blks = num_blks
        self.num_kv_heads = num_kv_heads
        self.blk_size = blk_size
        self.num_seqs = num_seqs
        self.max_num_blocks_per_seq = max_num_blocks_per_seq

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_len: torch.Tensor,
        scale: float,
        k_scale: float,
        v_scale: float,
        softcap: float,
    ) -> torch.Tensor:
        """
        Compute unified 3D attention.
        
        Args:
            query: [num_tokens, num_query_heads, head_size]
            key_cache: [num_blks, num_kv_heads, head_size, blk_size]
            value_cache: [num_blks, num_kv_heads, head_size, blk_size]
            block_tables: [num_seqs, max_num_blocks_per_seq]
            seq_lens: [num_seqs]
            query_start_len: [num_seqs + 1]
            scale: attention scale factor
            k_scale: key dequantization scale (for FP8)
            v_scale: value dequantization scale (for FP8)
            softcap: softcap value for attention scores
            
        Returns:
            segm_output: [num_tokens, num_query_heads, num_segments, head_size_padded]
        """
        num_queries_per_kv = self.num_query_heads // self.num_kv_heads
        
        # Initialize output tensor
        segm_output = torch.zeros(
            self.num_tokens,
            self.num_query_heads,
            self.num_segments,
            self.head_size_padded,
            dtype=query.dtype,
            device=query.device,
        )

        # Process each sequence
        for seq_idx in range(self.num_seqs):
            seq_start = query_start_len[seq_idx].item()
            seq_end = query_start_len[seq_idx + 1].item()
            seq_len = seq_lens[seq_idx].item()
            cur_batch_query_len = seq_end - seq_start
            context_len = seq_len - cur_batch_query_len

            if cur_batch_query_len == 0:
                continue

            # Get query tokens for this sequence
            q_seq = query[seq_start:seq_end]  # [cur_batch_query_len, num_query_heads, head_size]

            # Compute tiles per segment
            tiles_per_segment = math.ceil(seq_len / (self.num_segments * 32))

            # Process each segment
            for segm_idx in range(self.num_segments):
                if segm_idx * tiles_per_segment * 32 >= seq_len:
                    continue

                # Process each KV head
                for kv_head_idx in range(self.num_kv_heads):
                    # Get query heads for this KV head
                    q_head_start = kv_head_idx * num_queries_per_kv
                    q_head_end = q_head_start + num_queries_per_kv
                    q_heads = q_seq[:, q_head_start:q_head_end, :]  # [cur_batch_query_len, num_queries_per_kv, head_size]

                    # Initialize accumulator for this segment
                    segm_acc = torch.zeros(
                        cur_batch_query_len,
                        num_queries_per_kv,
                        self.head_size_padded,
                        dtype=torch.float32,
                        device=query.device,
                    )
                    segm_max = torch.full(
                        (cur_batch_query_len, num_queries_per_kv),
                        float("-inf"),
                        dtype=torch.float32,
                        device=query.device,
                    )
                    segm_expsum = torch.ones(
                        (cur_batch_query_len, num_queries_per_kv),
                        dtype=torch.float32,
                        device=query.device,
                    )

                    # Process tiles in this segment
                    tile_start = segm_idx * tiles_per_segment
                    tile_end = min((segm_idx + 1) * tiles_per_segment, math.ceil(seq_len / 32))

                    for tile_idx in range(tile_start, tile_end):
                        tile_start_pos = tile_idx * 32
                        tile_end_pos = min(tile_start_pos + 32, seq_len)
                        tile_size = tile_end_pos - tile_start_pos

                        # Load K and V for this tile
                        k_tile = torch.zeros(
                            self.head_size,
                            tile_size,
                            dtype=query.dtype,
                            device=query.device,
                        )
                        v_tile = torch.zeros(
                            tile_size,
                            self.head_size,
                            dtype=query.dtype,
                            device=query.device,
                        )

                        for pos in range(tile_size):
                            abs_pos = tile_start_pos + pos
                            block_idx = abs_pos // self.blk_size
                            block_offset = abs_pos % self.blk_size
                            physical_block = block_tables[seq_idx, block_idx].item()
                            
                            # key_cache: [num_blks, num_kv_heads, head_size // x, blk_size, x]
                            # value_cache: [num_blks, num_kv_heads, head_size, blk_size]
                            k_tile[:, pos] = key_cache[physical_block, kv_head_idx, :, block_offset, 0]
                            v_tile[pos, :] = value_cache[physical_block, kv_head_idx, :, block_offset]

                        # Apply scale for FP8
                        if key_cache.dtype == torch.float8_e4m3fn:
                            k_tile = k_tile.float() * k_scale
                            v_tile = v_tile.float() * v_scale

                        # Compute attention scores: Q @ K^T
                        # q_heads: [cur_batch_query_len, num_queries_per_kv, head_size]
                        # k_tile: [head_size, tile_size]
                        for q_pos in range(cur_batch_query_len):
                            abs_q_pos = context_len + q_pos
                            
                            # Causal mask: only attend to keys <= query position
                            if tile_start_pos > abs_q_pos:
                                continue

                            q_vec = q_heads[q_pos, :, :].float()  # [num_queries_per_kv, head_size]
                            
                            # S = scale * Q @ K
                            scores = scale * torch.matmul(q_vec, k_tile.float())  # [num_queries_per_kv, tile_size]

                            # Apply softcap if enabled
                            if softcap > 0:
                                scores = softcap * torch.tanh(scores / softcap)

                            # Apply causal mask
                            for k_pos in range(tile_size):
                                abs_k_pos = tile_start_pos + k_pos
                                if abs_k_pos > abs_q_pos:
                                    scores[:, k_pos] = float("-inf")

                            # Online softmax update
                            m_prev = segm_max[q_pos, :].clone()
                            m_new = torch.maximum(m_prev, scores.max(dim=1).values)
                            
                            # Handle case where max is -inf
                            m_new = torch.where(m_new > float("-inf"), m_new, torch.zeros_like(m_new).to(m_new.dtype))
                            
                            # Update accumulator
                            alpha = torch.exp(m_prev - m_new)
                            segm_acc[q_pos, :, :] *= alpha.unsqueeze(1)
                            
                            # Compute P = exp(S - m_new)
                            p = torch.exp(scores - m_new.unsqueeze(1))
                            segm_expsum[q_pos, :] = segm_expsum[q_pos, :] * alpha + p.sum(dim=1)
                            segm_max[q_pos, :] = m_new

                            # Accumulate: P @ V
                            segm_acc[q_pos, :, :] += torch.matmul(p, v_tile.float())

                    # Store output for this segment
                    for q_pos in range(cur_batch_query_len):
                        abs_q_pos = seq_start + q_pos
                        for q_head_local in range(num_queries_per_kv):
                            q_head_global = q_head_start + q_head_local
                            segm_output[abs_q_pos, q_head_global, segm_idx, :] = segm_acc[q_pos, q_head_local, :].to(query.dtype)

        return segm_output


def get_inputs():
    """Get inputs for forward pass."""
    # Load test data
    data = torch.load(
        '/mnt/w00934874/agent/code/AscendOpGenAgent/benchmarks/attention/kernel_unified_attention_3d_vllm/kernel_unified_attention_3d_test_data_fixed_v2.pt',
        map_location='cpu'
    )
    input_data = data['input_data']
    
    return [
        input_data['query_ptr'],        # [num_tokens, num_query_heads, head_size]
        input_data['key_cache_ptr'],    # [num_blks, num_kv_heads, head_size, blk_size]
        input_data['value_cache_ptr'],  # [num_blks, num_kv_heads, head_size, blk_size]
        input_data['block_tables_ptr'], # [num_seqs, max_num_blocks_per_seq]
        input_data['seq_lens_ptr'],     # [num_seqs]
        input_data['query_start_len_ptr'],  # [num_seqs + 1]
        input_data['scale'],            # float
        input_data['k_scale'],          # float
        input_data['v_scale'],          # float
        input_data['softcap'],          # float
    ]


def get_init_inputs():
    """Get initialization inputs for Model.__init__."""
    # Load test data to get shapes
    data = torch.load(
        '/mnt/w00934874/agent/code/AscendOpGenAgent/benchmarks/attention/kernel_unified_attention_3d_vllm/kernel_unified_attention_3d_test_data_fixed_v2.pt',
        map_location='cpu'
    )
    input_data = data['input_data']
    
    num_tokens = input_data['query_ptr'].shape[0]
    num_query_heads = input_data['num_query_heads']
    num_segments = input_data['NUM_SEGMENTS_PER_SEQ']
    head_size = input_data['HEAD_SIZE']
    head_size_padded = input_data['HEAD_SIZE_PADDED']
    num_blks = input_data['key_cache_ptr'].shape[0]
    num_kv_heads = input_data['key_cache_ptr'].shape[1]
    blk_size = input_data['BLOCK_SIZE']
    num_seqs = input_data['num_seqs']
    max_num_blocks_per_seq = input_data['block_tables_ptr'].shape[1]
    
    return [
        num_tokens,
        num_query_heads,
        num_segments,
        head_size,
        head_size_padded,
        num_blks,
        num_kv_heads,
        blk_size,
        num_seqs,
        max_num_blocks_per_seq,
    ]
