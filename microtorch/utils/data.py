import numpy as np

from ..tensor import Tensor


class DataLoader:
    def __init__(self, X, y, batchsize: int = 32, shuffle=False):
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.batchsize = batchsize

    def __len__(self):
        return len(self.X) // self.batchsize

    def __iter__(self):
        n = len(self.X)
        b = self.batchsize
        idx = np.arange(n)
        if self.shuffle:
            np.random.shuffle(idx)
        for i in range(0, n, b):
            i = idx[i:i+b]
            yield Tensor(self.X[i].reshape(-1, 28**2)), Tensor(self.y[i])