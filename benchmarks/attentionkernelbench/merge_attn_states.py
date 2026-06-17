import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Merge attention states kernel - merges prefix and suffix attention outputs.
    
    This is a reference PyTorch implementation of the merge_attn_states_kernel
    from vLLM, which combines two attention state tensors (prefix and suffix)
    using log-sum-exp (LSE) values for numerical stability.
    """
    
    def __init__(self, num_tokens: int, num_heads: int, head_size: int, padded_head_size: int):
        super(Model, self).__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_size = head_size
        self.padded_head_size = padded_head_size
        
    def forward(
        self,
        prefix_output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        prefix_lse: torch.Tensor,     # [NUM_HEADS, NUM_TOKENS]
        suffix_output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        suffix_lse: torch.Tensor,     # [NUM_HEADS, NUM_TOKENS]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Merge prefix and suffix attention states.
        
        Args:
            prefix_output: Prefix attention output [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
            prefix_lse: Prefix log-sum-exp [NUM_HEADS, NUM_TOKENS]
            suffix_output: Suffix attention output [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
            suffix_lse: Suffix log-sum-exp [NUM_HEADS, NUM_TOKENS]
        
        Returns:
            output: Merged attention output [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
            output_lse: Merged log-sum-exp [NUM_HEADS, NUM_TOKENS]
        """
        num_tokens = prefix_output.shape[0]
        num_heads = prefix_output.shape[1]
        head_size = prefix_output.shape[2]
        
        # Transpose LSE to [NUM_TOKENS, NUM_HEADS] for easier indexing
        p_lse = prefix_lse.t()  # [NUM_TOKENS, NUM_HEADS]
        s_lse = suffix_lse.t()  # [NUM_TOKENS, NUM_HEADS]
        
        # Handle inf values (FA2 behavior) - convert inf to -inf for consistency
        p_lse = torch.where(torch.isinf(p_lse) & (p_lse > 0), 
                           torch.tensor(float('-inf'), dtype=p_lse.dtype, device=p_lse.device), 
                           p_lse)
        s_lse = torch.where(torch.isinf(s_lse) & (s_lse > 0), 
                           torch.tensor(float('-inf'), dtype=s_lse.dtype, device=s_lse.device), 
                           s_lse)
        
        # Compute max LSE for numerical stability
        max_lse = torch.maximum(p_lse, s_lse)  # [NUM_TOKENS, NUM_HEADS]
        
        # Subtract max for numerical stability
        p_lse_stable = p_lse - max_lse
        s_lse_stable = s_lse - max_lse
        
        # Compute exp of stabilized LSE values
        p_se = torch.exp(p_lse_stable)  # [NUM_TOKENS, NUM_HEADS]
        s_se = torch.exp(s_lse_stable)  # [NUM_TOKENS, NUM_HEADS]
        
        # Sum of exp values
        out_se = p_se + s_se  # [NUM_TOKENS, NUM_HEADS]
        
        # Compute output LSE
        out_lse = torch.log(out_se) + max_lse  # [NUM_TOKENS, NUM_HEADS]
        
        # Transpose back to [NUM_HEADS, NUM_TOKENS]
        output_lse = out_lse.t()
        
        # Compute scale factors
        # Expand to match output dimensions [NUM_TOKENS, NUM_HEADS, 1]
        p_scale = (p_se / out_se).unsqueeze(-1)  # [NUM_TOKENS, NUM_HEADS, 1]
        s_scale = (s_se / out_se).unsqueeze(-1)  # [NUM_TOKENS, NUM_HEADS, 1]
        
        # Weighted combination of outputs
        output = prefix_output * p_scale + suffix_output * s_scale
        
        return output, output_lse


def get_inputs():
    """
    Returns input tensors for the forward method.
    Based on the test data: NUM_TOKENS=128, NUM_HEADS=16, HEAD_SIZE=128
    """
    num_tokens = 128
    num_heads = 16
    head_size = 128
    
    # Load test data from the provided file
    data = torch.load(
        '/mnt/w00934874/agent/code/AscendOpGenAgent/benchmarks/attention/merge_attn_states_kernel_vllm/merge_attn_states_test_v2.pt',
        weights_only=False,
        map_location='cpu'
    )
    input_data = data['input_data']
    
    prefix_output = input_data['prefix_output']
    prefix_lse = input_data['prefix_lse']
    suffix_output = input_data['suffix_output']
    suffix_lse = input_data['suffix_lse']
    
    return [prefix_output, prefix_lse, suffix_output, suffix_lse]


def get_init_inputs():
    """
    Returns initialization parameters for the Model class.
    """
    num_tokens = 128
    num_heads = 16
    head_size = 128
    padded_head_size = 128
    
    return [num_tokens, num_heads, head_size, padded_head_size]
