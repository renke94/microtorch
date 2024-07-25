from typing import Iterable

from .tensor import Tensor


class Optimizer:
    def step(self):
        pass

    def zero_grad(self):
        pass


class SGD(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 0.001):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()