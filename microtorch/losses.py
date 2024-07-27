import numpy as np

from .nn import Module
from .tensor import Tensor


class MSELoss(Module):
    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        o = np.sum((input.data - target.data) ** 2) / len(target)
        out = Tensor(o, _children=(input,))

        def mse_backward():
            input.grad += (input.data - target.data) * 2 / len(target) * out.grad

        out._backward = mse_backward
        return out


class BCELoss(Module):
    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        p = probs.data
        y = target.data
        o = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / len(y)

        out = Tensor(o, _children=(probs,))

        def bce_backward():
            probs.grad += (p - y) / (p * (1 - p)) / len(y) * out.grad

        out._backward = bce_backward
        return out


class BCEWithLogitsLoss(Module):
    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        z = logits.data
        y = target.data
        o = np.sum(np.maximum(0, z) - z * y + np.log(1 + np.exp(-np.abs(z)))) / len(y)

        out = Tensor(o, _children=(logits,))

        def bce_with_logits_backward():
            # (1/N) * sigmoid(z) - y * out.grad
            logits.grad += (1 / (1 + np.exp(-z)) - y) / len(y) * out.grad

        out._backward = bce_with_logits_backward
        return out


# https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
class CrossEntropyLoss(Module):
    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        # calculate softmax
        m = logits.data.max()
        e = np.exp(logits.data - m)
        s = e / e.sum(axis=-1, keepdims=True)
        n = s.shape[0]

        if isinstance(target, Tensor):
            target = target.data

        rows = np.arange(n)
        ce = - np.log(s[rows, target]).sum() / n
        out = Tensor(ce, _children=(logits,))

        def cross_entropy_backward():
            onehot = np.zeros_like(s)
            onehot[rows, target] = 1.0
            logits.grad += (s - onehot) / n * out.grad

        out._backward = cross_entropy_backward
        return out



if __name__ == '__main__':
    bce = BCEWithLogitsLoss()
    p = Tensor([[0.3367], [0.1288], [0.2345], [0.2303]])
    y = Tensor([[0.0], [1.0], [0.0], [0.0]])

    loss = bce(p, y)
    loss.backward()

    print(loss)
    print(p.grad)
