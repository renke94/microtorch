import numpy as np

import microtorch.nn as nn
from microtorch.tensor import Tensor, concat


class AttentionHead(nn.Module):
    """Attention head."""

    def __init__(self, embedding_dim: int, head_size: int) -> None:
        super().__init__()
        self.scale = head_size ** -0.5
        self.q = nn.Linear(embedding_dim, head_size, bias=False)  # 32, 8
        self.k = nn.Linear(embedding_dim, head_size, bias=False)  # 32, 8
        self.v = nn.Linear(embedding_dim, head_size, bias=False)  # 32, 8

    def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        _, N, _ = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        w = q @ k.transpose(-2, -1) * self.scale
        if is_causal:
            w = w.masked_fill(Tensor.tril(Tensor.ones(N, N)) == 0, float('-inf'))
        w = w.softmax(dim=-1)
        out = w @ v
        return out.leaky_relu(negative_slope=0.01)


class MultiheadAttentionSlow(nn.Module):
    """Multi-attention head."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(embedding_dim, embedding_dim // num_heads)  # 32, 8
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        return self.proj(concat([h(x, is_causal) for h in self.heads], dim=-1))  # [32, 8] * nheads -> [32, 32]



class MultiheadAttentionFast(nn.Module):
    """Multi-head attention - fast implementation."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = embed_dim
        self.nhead = num_heads
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.head_size = embed_dim // num_heads
        self.scale = self.head_size ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)  # 32 96
        scale = np.sqrt(2.0 / (embed_dim + self.head_size))
        self.qkv.weight = Tensor(np.random.randn(embed_dim, embed_dim * 3) * scale, requires_grad=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, input: Tensor, is_causal: bool = True) -> Tensor:
        B, N, E = input.shape
        q, k, v = self.qkv(input).chunk(3, dim=-1)  # B N E -> B 32 32
        q = q.reshape(B, N, self.nhead, self.head_size).transpose(1, 2)  # B H N E -> B N 4 8 -> B 4 N 8
        k = k.reshape(B, N, self.nhead, self.head_size).transpose(1, 2)  # B H N E
        v = v.reshape(B, N, self.nhead, self.head_size).transpose(1, 2)  # B H N E
        weights = q @ k.transpose(-2, -1) * self.scale                   #            B 4 N 8 @ B 4 8 N -> B 4 N N
        if is_causal:
            weights = weights.masked_fill(Tensor.tril(Tensor.ones(N, N)) == 0, float('-inf'))
        weights = weights.softmax(dim=-1)
        weights = self.dropout(weights)
        out = weights @ v                                                # B H N E: B 4 N N @ B 4 N 8 -> B 4 N 8
        out = out.transpose(1, 2).reshape(B, N, E)                       # B 4 N 8 -> B N 4 8 -> B N 32
        out = self.proj(out)
        return out
