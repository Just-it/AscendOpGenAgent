import torch
import torch.nn as nn


class Model(nn.Module):
    """
    _fwd_kernel_ep_scatter_1 - MoE expert scatter kernel
    
    This kernel computes the cumulative sum of tokens per expert and
    fills the m_indices array with expert assignments.
    """
    def __init__(self, num_experts: int, BLOCK_E: int, BLOCK_EXPERT_NUM: int):
        super(Model, self).__init__()
        self.num_experts = num_experts
        self.BLOCK_E = BLOCK_E
        self.BLOCK_EXPERT_NUM = BLOCK_EXPERT_NUM

    def forward(
        self,
        num_recv_tokens_per_expert: torch.Tensor,
        expert_start_loc: torch.Tensor,
        m_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward implementation of _fwd_kernel_ep_scatter_1
        
        Args:
            num_recv_tokens_per_expert: [num_experts] int32 - tokens per expert
            expert_start_loc: [num_experts] int32 - output, expert start locations
            m_indices: [total_tokens] int32 - output, token to expert mapping
            
        Returns:
            m_indices: filled with expert assignments
        """
        num_experts = self.num_experts
        BLOCK_E = self.BLOCK_E
        BLOCK_EXPERT_NUM = self.BLOCK_EXPERT_NUM
        
        # Compute cumulative sum for expert_start_loc
        # cumsum[i] = sum(tokens_per_expert[0:i])
        cumsum = torch.cumsum(num_recv_tokens_per_expert, dim=0) - num_recv_tokens_per_expert
        expert_start_loc.copy_(cumsum)
        
        # Fill m_indices for each expert
        for cur_expert in range(num_experts):
            cur_expert_start = expert_start_loc[cur_expert].item()
            cur_expert_token_num = num_recv_tokens_per_expert[cur_expert].item()
            
            # Fill m_indices with cur_expert for tokens belonging to this expert
            end_idx = min(cur_expert_start + cur_expert_token_num, m_indices.shape[0])
            if cur_expert_start < m_indices.shape[0]:
                m_indices[cur_expert_start:end_idx] = cur_expert
        
        return m_indices


def get_inputs():
    """Return forward() input arguments."""
    num_experts = 8
    total_tokens = 1664
    
    # num_recv_tokens_per_expert: [num_experts] int32
    num_recv_tokens_per_expert = torch.tensor([208, 208, 208, 208, 208, 208, 208, 208], dtype=torch.int32)
    
    # expert_start_loc: [num_experts] int32 (output buffer)
    expert_start_loc = torch.zeros(num_experts, dtype=torch.int32)
    
    # m_indices: [total_tokens] int32 (output buffer)
    m_indices = torch.zeros(total_tokens, dtype=torch.int32)
    
    return [
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
    ]


def get_init_inputs():
    """Return __init__() arguments."""
    num_experts = 8
    BLOCK_E = 128
    BLOCK_EXPERT_NUM = 8
    return [num_experts, BLOCK_E, BLOCK_EXPERT_NUM]
