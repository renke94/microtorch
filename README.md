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
    nn.Sigmoid()
)

optimizer = Adam(model.params(), lr=0.1)

for i in range(1000):
    loss = Tensor.l2(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

pred = model(x)
# Tensor([[0.00644011],
#         [0.99703074],
#         [0.99702374],
#         [0.0016798 ]])
~~~

## Todo's
- [x] Implementing a generic backward function for array broadcasting
- [ ] Stable Sigmoid backward implementation
- [ ] Optimizer implementations
- [ ] Weight initialization (Xavier and He)

## Contributions
This framework is for educational purposes.
Ideas, suggestions, and issues are welcome! 
Feel free to open an issue or submit a pull request.
Thank you for your support!
