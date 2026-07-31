# MicroTorch

A minimal PyTorch clone for educational purposes using plain NumPy.


## Usage

```python
from microtorch.tensor import Tensor
from microtorch import nn
from microtorch import optim
from microtorch.losses import bce_with_logits_loss

import numpy as np
np.random.seed(42)

x = Tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

y = Tensor([[0.0, 1.0, 1.0, 0.0]]).T

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

optimizer = optim.Adam(model.params(), lr=1e-2)

def train_step() -> float:
    logits = model(x)
    loss = bce_with_logits_loss(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

for _ in range(100):
    loss = train_step()

model(x).sigmoid()

>>> Tensor([[0.00268848],
            [0.99754794],
            [0.99744306],
            [0.00264953]])
```

## More Examples

- [MNIST Classification](notebooks/mnist.ipynb)
- [GPT on Tiny Shakespeare](notebooks/MicroTorch%20GPT.ipynb)

## Resources
- [Deriving categorical cross entropy and softmax](https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/)
- [Calculate Maximum Likelihood Estimator with Newton-Raphson Method using R](https://towardsdatascience.com/calculate-maximum-likelihood-estimator-with-newton-raphson-method-using-r-7d3f69fbf8fe/)
- [Derivative of the Softmax Function and the Categorical Cross-Entropy Loss](https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)
