from typing import Iterable

import numpy as np
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


class Adam(Optimizer):
    def __init__(
            self,
            params: Iterable[Tensor],
            lr: float = 0.001,
            betas: tuple[float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
    ):
        self.params = list(params)
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.timestep = 1

    def step(self):
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            m[:] = self.betas[0] * m + (1 - self.betas[0]) * g
            v[:] = self.betas[1] * v + (1 - self.betas[1]) * g**2

            m_hat = m / (1 - self.betas[0]**self.timestep)
            v_hat = v / (1 - self.betas[1]**self.timestep)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        self.timestep += 1

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

