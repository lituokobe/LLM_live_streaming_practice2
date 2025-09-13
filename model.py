# import libraries
import torch
from torch import nn
import torch.nn.functional as F

# Define RMSNorm class
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x *torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

# Pre-calculate cos and sin for each position
def precompute_freqs_cis(dim: int, end: int = int(32*1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # [: (dim // 2)] means dim must be even
    t = torch.arange(end, device=freqs.device)
    #m*theta
    freqs = torch.outer(t, freqs).float()
    #cos(m*theta)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    #sin(m*theta)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin
# apply positional embedding
def apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin, position_ids=None, unsqueeze_dim = 1):
    def rotate_half(x):
        # arr[..., 2:4] = arr[:, :, :, 2:4]
        # categorize in the input to groups of 2, but not (x1, x2) (x3, x40... but (x1, x_d/2) (x2, x_d/2+1)....
        # this is RoPE done by HuggingFace and Qwen
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_emb = q * freqs_cos.unsqueeze(unsqueeze_dim) + (rotate_half(q) * freqs_sin.unsqueeze(unsqueeze_dim))
    k_emb = k * freqs_cos.unsqueeze(unsqueeze_dim) + (rotate_half(k) * freqs_sin.unsqueeze(unsqueeze_dim))
    return q_emb, k_emb