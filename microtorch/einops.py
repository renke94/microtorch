import numpy as np

from .tensor import Tensor


def einsum(a: Tensor, b: Tensor, pattern: str):
    ops, result = list(map(str.strip, pattern.split('->')))
    op1, op2 = list(map(str.strip, ops.split(',')))

    out = np.einsum(pattern, a.data, b.data)
    out = Tensor(out, _children=(a, b), _op=pattern)

    def einsum_backward():
        a.grad += np.einsum(f'{op2},{result}->{op1}', b.data, out.grad)
        b.grad += np.einsum(f'{op1},{result}->{op2}', a.data, out.grad)

    out._backward = einsum_backward
    return out