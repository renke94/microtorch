import numpy as np

from .nn import Module
from .tensor import Tensor


class MSELoss(Module):
    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return Tensor.l2(input, target)


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


if __name__ == '__main__':
    bce = BCELoss()
    p = Tensor([[0.3367], [0.1288], [0.2345], [0.2303]])
    y = Tensor([[0.0], [1.0], [0.0], [0.0]])

    loss = bce(p, y)
    loss.backward()

    print(loss)
    print(p.grad)