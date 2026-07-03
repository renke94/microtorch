import numpy as np

from microtorch.tensor import Tensor


def dropout(x: Tensor, p: float = 0.5) -> Tensor:
    """Dropout layer."""
    if p > 0.0:
        return x.masked_fill(np.random.binomial(1, p, size=x.shape), 0)
    return x

def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0
) -> Tensor:
    """Scaled dot-product attention."""
    scale = q.shape[-1] ** -0.5
    w = q @ k.transpose(-2, -1) * scale
    if is_causal:
        w = w.masked_fill(Tensor.tril(Tensor.ones(q.shape[-2], k.shape[-2])) == 0, float('-inf'))
    if attn_mask is not None:
        w = w.masked_fill(attn_mask, float('-inf'))
    w = w.softmax(dim=-1)
    w = dropout(w, dropout_p)
    return w @ v
