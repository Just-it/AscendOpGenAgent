import torch
import torch.nn as nn


class Model(nn.Module):
    """
    _fwd_kernel_ep_scatter_2 - MoE expert scatter kernel (phase 2)
    
    This kernel scatters tokens to their assigned experts based on topk indices.
    It copies token data (recv_x) and scale data (recv_x_scale) to output buffers
    according to expert assignments.
    """
    def __init__(
        self,
        topk_num: int,
        HIDDEN_SIZE: int,
        HIDDEN_SIZE_PAD: int,
        SCALE_HIDDEN_SIZE: int,
        SCALE_HIDDEN_SIZE_PAD: int,
    ):
        super(Model, self).__init__()
        self.topk_num = topk_num
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.HIDDEN_SIZE_PAD = HIDDEN_SIZE_PAD
        self.SCALE_HIDDEN_SIZE = SCALE_HIDDEN_SIZE
        self.SCALE_HIDDEN_SIZE_PAD = SCALE_HIDDEN_SIZE_PAD

    def forward(
        self,
        total_token_num: int,
        expert_start_loc: torch.Tensor,
        recv_x: torch.Tensor,
        recv_x_scale: torch.Tensor,
        recv_topk: torch.Tensor,
        output_tensor: torch.Tensor,
        output_tensor_scale: torch.Tensor,
        output_index: torch.Tensor,
    ) -> tuple:
        """
        Forward implementation of _fwd_kernel_ep_scatter_2
        
        Args:
            total_token_num: total number of tokens
            expert_start_loc: [num_experts] int32 - expert start locations (updated in-place)
            recv_x: [total_token_num, HIDDEN_SIZE] float32 - input token data
            recv_x_scale: [total_token_num, SCALE_HIDDEN_SIZE] float32 - input scale data
            recv_topk: [total_token_num, topk_num] int32 - expert assignments for each token
            output_tensor: [output_size, HIDDEN_SIZE] float32 - output buffer for token data
            output_tensor_scale: [output_size, SCALE_HIDDEN_SIZE] float32 - output buffer for scale data
            output_index: [total_token_num, topk_num] int32 - output destination indices
            
        Returns:
            (output_tensor, output_tensor_scale, output_index): updated output buffers
        """
        topk_num = self.topk_num
        HIDDEN_SIZE = self.HIDDEN_SIZE
        SCALE_HIDDEN_SIZE = self.SCALE_HIDDEN_SIZE
        
        # Create a copy of expert_start_loc to track current positions
        expert_current_loc = expert_start_loc.clone()
        
        # Process each token
        for token_id in range(total_token_num):
            # Load token data
            to_copy = recv_x[token_id, :HIDDEN_SIZE].clone()
            to_copy_s = recv_x_scale[token_id, :SCALE_HIDDEN_SIZE].clone()
            
            # Process each topk assignment
            for topk_idx in range(topk_num):
                expert_id = recv_topk[token_id, topk_idx].item()
                
                if expert_id >= 0:
                    # Get destination index and increment counter
                    dest_token_index = expert_current_loc[expert_id].item()
                    expert_current_loc[expert_id] += 1
                    
                    # Store output index
                    output_index[token_id, topk_idx] = dest_token_index
                    
                    # Copy data to output buffers
                    output_tensor[dest_token_index, :HIDDEN_SIZE] = to_copy
                    output_tensor_scale[dest_token_index, :SCALE_HIDDEN_SIZE] = to_copy_s
        
        return output_tensor, output_tensor_scale, output_index


def get_inputs():
    """Return forward() input arguments."""
    total_token_num = 128
    num_experts = 16
    topk_num = 16
    HIDDEN_SIZE = 128
    SCALE_HIDDEN_SIZE = 1
    output_size = 2048
    
    # expert_start_loc: [num_experts] int32 - starting positions for each expert
    expert_start_loc = torch.zeros(num_experts, dtype=torch.int32)
    
    # recv_x: [total_token_num, HIDDEN_SIZE] float32
    recv_x = torch.randn(total_token_num, HIDDEN_SIZE, dtype=torch.float32)
    
    # recv_x_scale: [total_token_num, SCALE_HIDDEN_SIZE] float32
    recv_x_scale = torch.randn(total_token_num, SCALE_HIDDEN_SIZE, dtype=torch.float32)
    
    # recv_topk: [total_token_num, topk_num] int32 - expert assignments
    # Create alternating expert assignments
    recv_topk = torch.arange(topk_num, dtype=torch.int32).unsqueeze(0).repeat(total_token_num, 1)
    recv_topk = recv_topk % num_experts
    
    # output_tensor: [output_size, HIDDEN_SIZE] float32
    output_tensor = torch.zeros(output_size, HIDDEN_SIZE, dtype=torch.float32)
    
    # output_tensor_scale: [output_size, SCALE_HIDDEN_SIZE] float32
    output_tensor_scale = torch.zeros(output_size, SCALE_HIDDEN_SIZE, dtype=torch.float32)
    
    # output_index: [total_token_num, topk_num] int32
    output_index = torch.zeros(total_token_num, topk_num, dtype=torch.int32)
    
    return [
        total_token_num,
        expert_start_loc,
        recv_x,
        recv_x_scale,
        recv_topk,
        output_tensor,
        output_tensor_scale,
        output_index,
    ]


def get_init_inputs():
    """Return __init__() arguments."""
    topk_num = 16
    HIDDEN_SIZE = 128
    HIDDEN_SIZE_PAD = 128
    SCALE_HIDDEN_SIZE = 1
    SCALE_HIDDEN_SIZE_PAD = 1
    return [topk_num, HIDDEN_SIZE, HIDDEN_SIZE_PAD, SCALE_HIDDEN_SIZE, SCALE_HIDDEN_SIZE_PAD]
