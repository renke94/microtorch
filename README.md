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
    nn.LeakyReLU(0.01),
    nn.Linear(16, 16),
    nn.LeakyReLU(0.01),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

def optimize(params, lr=0.001):
    for p in params:
        p.data -= lr * p.grad
        
def zero_grad(*params):
    for p in params:
        p.zero_grad()

for i in range(1000):
    loss = Tensor.l2(model(x), y)
    zero_grad(*model.params(), x)
    loss.backward()
    optimize(model.params(), lr=0.1)

pred = model(x)
#Tensor([[0.02210465],
#        [0.98219411],
#        [0.9799512 ],
#        [0.0193837 ]])
~~~

## Todo's
- [x] Implementing a generic backward function for array broadcasting
- [ ] Stable Sigmoid backward implementation

## Contributions
This framework is for educational purposes.
Ideas, suggestions, and issues are welcome! 
Feel free to open an issue or submit a pull request.
Thank you for your support!
