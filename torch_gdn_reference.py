"""
Standalone PyTorch reference implementations for GatedDeltaNet kernels.
Extracted from qwen3_gdn.py for testing purposes.
"""

from typing import Optional
import torch
import torch.nn.functional as F


def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """
    Causal Conv1D Update Function (Pure PyTorch Fallback).
    
    Args:
        hidden_states: Input of shape (batch_size, hidden_size, seq_len)
        conv_state: Sliding window state of shape (batch_size, hidden_size, state_len)
        weight: Conv1D weight of shape (hidden_size, 1, state_len)
        bias: Optional bias tensor of shape (hidden_size,)
        activation: Optional activation string ('silu' or similar)
    
    Returns:
        Output tensor of shape (batch_size, hidden_size, seq_len)
    """
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight, bias, padding=0, groups=hidden_size)
    
    if activation == 'silu':
        out = F.silu(out[:, :, -seq_len:])
    else:
        out = out[:, :, -seq_len:]
    
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 Normalization along specified dimension."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Chunked Gated DeltaNet Forward Pass.
    
    Args:
        query: Query tensor of shape (batch_size, seq_len, linear_num_value_heads, linear_key_head_dim)
        key: Key tensor of shape (batch_size, seq_len, linear_num_value_heads, linear_key_head_dim)
        value: Value tensor of shape (batch_size, seq_len, linear_num_value_heads, linear_value_head_dim)
        g: Gate/decay tensor of shape (batch_size, seq_len, linear_num_value_heads)
        beta: Scalar/beta tensor of shape (batch_size, seq_len, linear_num_value_heads)
        chunk_size: Size of chunks to process (default 64)
        initial_state: Optional initial recurrent state
        output_final_state: Whether to return the final recurrent state
        use_qk_l2norm_in_kernel: Whether to L2-normalize query/key
    
    Returns:
        Tuple of (output, final_state)
        - output: Shape (batch_size, seq_len, linear_num_value_heads, linear_value_head_dim)
        - final_state: Shape (batch_size, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim) or None
    """
    initial_dtype = query.dtype
    
    # Optional L2 normalization
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    
    # Transpose to (batch, heads, seq, dim) and convert to float32
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) 
        for x in (query, key, value, beta, g)
    ]
    
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    
    # Padding to chunk boundaries
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    
    # Scale query
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    
    # Compute v_beta and k_beta
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    
    # Reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) 
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    
    # Causal mask
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    
    # Chunk decay and attention
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    
    # Associative scan
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    
    # Initialize recurrent state
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    
    # Process each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    
    if not output_final_state:
        last_recurrent_state = None
    
    # Reshape and remove padding
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    
    # Transpose back to (batch, seq, heads, dim)
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    
    return core_attn_out, last_recurrent_state
