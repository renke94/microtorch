import numpy as np

from microtorch import Tensor
from microtorch import nn
from microtorch.optim import Adam
from microtorch.losses import MSELoss, BCEWithLogitsLoss

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
    nn.Sigmoid(),
    nn.Linear(16, 16),
    nn.Sigmoid(),
    nn.Linear(16, 1),
)

mse = MSELoss()
bce = BCEWithLogitsLoss()

optimizer = Adam(model.params(), lr=0.01)

if __name__ == '__main__':
    for i in range(200):
        loss = bce(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    pred = model(x)
    print(pred.sigmoid())




