# microtorch
A minimal PyTorch clone using plain NumPy. 

Inspiration for this implementation comes from:
- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- [PyTorch](https://pytorch.org/) by Meta

## Examples

### XOR
~~~Python
import numpy as np
from microtorch import Tensor
from microtorch import nn
from microtorch.optim import Adam
from microtorch.losses import BCEWithLogitsLoss

x = Tensor([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

y = Tensor([[0.0, 1.0, 1.0, 0.0]]).T

np.random.seed(42)

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.Sigmoid(),
    nn.Linear(16, 16),
    nn.Sigmoid(),
    nn.Linear(16, 1),
)

optimizer = Adam(model.params(), lr=0.01)

bce = BCEWithLogitsLoss()

for i in range(200):
    loss = bce(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

pred = model(x).sigmoid()
# Tensor([[0.00218708],
#         [0.99718049],
#         [0.99636125],
#         [0.00461519]])
~~~

## Todo's
- [x] Implementing a generic backward function for array broadcasting
- [x] Optimizer implementations
- [x] Weight initialization (Xavier and He)
- [x] BCELoss and BCEWithLogitsLoss
- [x] Softmax and Categorical Cross Entropy
- [x] MNIST image classification
- [x] Tensor operations for reshape (and more)
- [ ] Indexing with backward
- [ ] Batch matrix multiplication

## Contributions
This framework is for educational purposes.
Ideas, suggestions, and issues are welcome! 
Feel free to open an issue or submit a pull request.
Thank you for your support!
