import torch
import math

from torch import nn
from torch.nn import functional as Func

class MultiHeadAttention(nn.Module):
    """
    Multi head Attention layer.

    Args:
     - heads (int): Number of attention heads.
     - dim (int): dimension of the embedding.
     - ibias (bool): Whether to use bias in the input projection.
     - obias (bool): Whether to use bias in the output projection.

    Note: 
     - The input is projected into three different spaces (Q, K, V) via three matrices (Wq, Wk, Wv) using a single linear layer.
    
    Refernce:
     - https://github.com/hkproj/pytorch-stable-diffusion
    """
    def __init__(self, heads: int, dim: int, ibias: bool = True, obias: bool = True):
        super().__init__()
        self.Wi = nn.Linear(dim, 3 * dim, bias=ibias)
        self.Wo = nn.Linear(dim, dim, bias=obias)
        self.heads = heads
        self.head_dim = dim // heads

    def forward(self, x, causal_mask=False):
        """
        Args:
         - x (Tensor): Input tensor of shape (batch_Size, sequence_len, dim).
         - causal_mask (bool): Whether to apply a causal mask to the attention weights.

        Note:
         - sequence_len is the number of tokens in the input sequence.
        """
        input_shape = x.shape 
        batch_Size, sequence_len, d = input_shape 

        interim_shape = (batch_Size, sequence_len, self.heads, self.head_dim) 

        # Get query (q), key (k) and value (v) as three tensors of shape (batch_Size, sequence_len, dim) as Wi output is of shape (batch_Size, sequence_len, 3 * dim)
        q, k, v = self.Wi(x).chunk(3, dim=-1)
        
        # (batch_Size, sequence_len, dim) - view -> (batch_Size, sequence_len, head_dim, dim / head_dim) - transpose -> (batch_Size, head_dim, sequence_len, dim / head_dim)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Compute normalized attention scores
        # (batch_Size, head_dim, sequence_len, dim / head_dim) @ (batch_Size, head_dim, dim / head_dim, sequence_len) -> (batch_Size, head_dim, sequence_len, sequence_len)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            mask = torch.triu(torch.ones_like(weight, dtype=torch.bool), diagonal=1)
            weight.masked_fill_(mask, -torch.inf) 
        
        weight /= math.sqrt(self.head_dim) 
        weight = Func.softmax(weight, dim=-1) 

        # (batch_Size, head_dim, sequence_len, sequence_len) @ (batch_Size, head_dim, sequence_len, dim / head_dim) -> (batch_Size, head_dim, sequence_len, dim / head_dim)
        output = weight @ v

        # (batch_Size, head_dim, sequence_len, dim / head_dim) -> (batch_Size, sequence_len, head_dim, dim / head_dim)
        output = output.transpose(1, 2) 

        # (batch_Size, sequence_len, head_dim, dim / head_dim) -> (batch_Size, sequence_len, dim)
        output = output.reshape(input_shape) 

        # (batch_Size, sequence_len, dim) -> (batch_Size, sequence_len, dim)
        return self.Wo(output) 
        

# class CrossAttention(nn.Module):
    """
    
    Refernce:
     - https://github.com/hkproj/pytorch-stable-diffusion
    """
    def __init__(self, heads, f, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(f, f, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, f, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, f, bias=in_proj_bias)
        self.out_proj = nn.Linear(f, f, bias=out_proj_bias)
        self.heads = heads
        self.d_head = f // heads
    
    def forward(self, x, y):
        # x (latent): # (batch_Size, sequence_len_Q, dim_Q)
        # y (context): # (batch_Size, sequence_len_KV, dim_KV) = (batch_Size, 77, 768)

        input_shape = x.shape
        batch_Size, sequence_length, f = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * heads = dim_Q
        interim_shape = (batch_Size, -1, self.heads, self.d_head)
        
        # (batch_Size, sequence_len_Q, dim_Q) -> (batch_Size, sequence_len_Q, dim_Q)
        q = self.q_proj(x)
        # (batch_Size, sequence_len_KV, dim_KV) -> (batch_Size, sequence_len_KV, dim_Q)
        k = self.k_proj(y)
        # (batch_Size, sequence_len_KV, dim_KV) -> (batch_Size, sequence_len_KV, dim_Q)
        v = self.v_proj(y)

        # (batch_Size, sequence_len_Q, dim_Q) -> (batch_Size, sequence_len_Q, H, dim_Q / H) -> (batch_Size, H, sequence_len_Q, dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2) 
        # (batch_Size, sequence_len_KV, dim_Q) -> (batch_Size, sequence_len_KV, H, dim_Q / H) -> (batch_Size, H, sequence_len_KV, dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2) 
        # (batch_Size, sequence_len_KV, dim_Q) -> (batch_Size, sequence_len_KV, H, dim_Q / H) -> (batch_Size, H, sequence_len_KV, dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2) 
        
        # (batch_Size, H, sequence_len_Q, dim_Q / H) @ (batch_Size, H, dim_Q / H, sequence_len_KV) -> (batch_Size, H, sequence_len_Q, sequence_len_KV)
        weight = q @ k.transpose(-1, -2)
        
        # (batch_Size, H, sequence_len_Q, sequence_len_KV)
        weight /= math.sqrt(self.d_head)
        
        # (batch_Size, H, sequence_len_Q, sequence_len_KV)
        weight = F.softmax(weight, dim=-1)
        
        # (batch_Size, H, sequence_len_Q, sequence_len_KV) @ (batch_Size, H, sequence_len_KV, dim_Q / H) -> (batch_Size, H, sequence_len_Q, dim_Q / H)
        output = weight @ v
        
        # (batch_Size, H, sequence_len_Q, dim_Q / H) -> (batch_Size, sequence_len_Q, H, dim_Q / H)
        output = output.transpose(1, 2).contiguous()
        
        # (batch_Size, sequence_len_Q, H, dim_Q / H) -> (batch_Size, sequence_len_Q, dim_Q)
        output = output.view(input_shape)
        
        # (batch_Size, sequence_len_Q, dim_Q) -> (batch_Size, sequence_len_Q, dim_Q)
        output = self.out_proj(output)

        # (batch_Size, sequence_len_Q, dim_Q)
        return output