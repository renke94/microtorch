import numpy as np

from microtorch import Tensor
from microtorch import nn
from microtorch.optim import SGD

x = Tensor([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

y = Tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

np.random.seed(42)
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.LeakyReLU(0.01),
    nn.Linear(16, 16),
    nn.LeakyReLU(0.01),
    nn.Linear(16, 1),
    nn.Sigmoid()
)


if __name__ == '__main__':
    optimizer = SGD(model.params(), lr=0.1)
    for i in range(1000):
        loss = Tensor.l2(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    pred = model(x)
    print(pred)
