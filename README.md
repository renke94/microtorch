# microtorch
A minimal PyTorch clone using plain NumPy. 
This framework is for educational use only!

Inspiration for this implementation comes from:
- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- [PyTorch](https://pytorch.org/) by Meta

## Examples

### XOR
~~~Python
from microtorch import Tensor
from microtorch import nn

x = Tensor([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

y = Tensor([[0.0, 1.0, 1.0, 0.0]]).T

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
#Tensor([[0.01606589],
#        [0.99038489],
#        [0.99465307],
#        [0.00223828]])
~~~

# Todo's
- [ ] Implementing a generic backward function for array broadcasting