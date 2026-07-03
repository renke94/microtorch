# MicroTorch

A minimal PyTorch clone for educational purposes using plain NumPy


## Installation

```bash
pip install microtorch
```

## Usage

```python
from microtorch.tensor import Tensor

x = Tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

y = Tensor([0, 1, 1, 0]).T
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync

# Run the project
uv run main

# Run linting
uv run --group lint ruff check
uv run --group lint mypy src

# Run tests
uv run pytest
```

## Resources
- [Deriving categorical cross entropy and softmax](https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/)
- [Calculate Maximum Likelihood Estimator with Newton-Raphson Method using R](https://towardsdatascience.com/calculate-maximum-likelihood-estimator-with-newton-raphson-method-using-r-7d3f69fbf8fe/)
- [Derivative of the Softmax Function and the Categorical Cross-Entropy Loss](https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)
- []()