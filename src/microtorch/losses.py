from typing import Any

import numpy as np

from microtorch.tensor import Tensor


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss."""
    return (pred - target).abs().mean()


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error loss."""
    return ((pred - target) ** 2).mean()


def bce_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Binary cross entropy loss."""
    p = pred.data
    y = target.data
    o = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / len(y)

    out = Tensor(o, children=[pred], requires_grad=True, op='bce')

    def bce_backward() -> None:
        pred.grad += (p - y) / (p * (1 - p)) / len(y) * out.grad

    out._backward = bce_backward
    return out


def bce_with_logits_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Binary cross entropy loss with logits."""
    z = logits.data
    y = target.data
    o = np.sum(np.maximum(0, z) - z * y + np.log(1 + np.exp(-np.abs(z)))) / len(y) # type: ignore

    out = Tensor(o, children=[logits], requires_grad=True, op='bce_with_logits')

    def bce_with_logits_backward() -> None:
        # (1/N) * sigmoid(z) - y * out.grad
        logits.grad += (1 / (1 + np.exp(-z)) - y) / len(y) * out.grad

    out._backward = bce_with_logits_backward
    return out


def cross_entropy_with_logits_loss(logits: Tensor, target: Tensor | np.ndarray[Any, Any]) -> Tensor:
    """Cross entropy loss.

    https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    """
    # calculate softmax
    shape = logits.shape
    assert shape[:-1] == target.shape, f'target shape {target.shape} must have shape of logits.shape[:-1] {shape[:-1]}'

    data = logits.data.reshape(-1, shape[-1])
    m = data.max()
    e = np.exp(data - m)
    s = e / e.sum(axis=-1, keepdims=True)
    n = s.shape[0]

    if isinstance(target, Tensor):
        t = target.data
    else:
        t = target
    t = np.asarray(t, dtype=np.int64)
    t = t.reshape(-1)

    rows = np.arange(n, dtype=np.int64)

    ce = - np.log(s[rows, t]).sum() / n
    out = Tensor(ce, children=[logits], requires_grad=True, op='cross_entropy')

    def cross_entropy_backward() -> None:
        onehot = np.zeros_like(s)
        onehot[rows, t] = 1.0
        grad = (s - onehot) / n * out.grad
        logits.grad += grad.reshape(shape)

    out._backward = cross_entropy_backward
    return out


def cross_entropy_loss(probs: Tensor, target: Tensor | np.ndarray[Any, Any]) -> Tensor:
    """Cross entropy loss."""
    p = probs.data
    n = p.shape[0]

    if isinstance(target, Tensor):
        t = target.data
    else:
        t = target
    t = np.asarray(t, dtype=np.int64)

    rows = np.arange(n, dtype=np.int64)

    eps = 1e-12
    clipped = np.clip(p, eps, 1.0)

    ce = - np.log(clipped[rows, t]).sum() / n
    out = Tensor(ce, children=[probs], requires_grad=True, op='cross_entropy')

    def cross_entropy_backward() -> None:
        onehot = np.zeros_like(clipped)
        onehot[rows, t] = 1.0
        probs.grad += -(onehot / clipped) / n * out.grad

    out._backward = cross_entropy_backward
    return out
