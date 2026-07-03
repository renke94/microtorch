import numpy as np

from microtorch.tensor import Tensor


def einsum(a: Tensor, b: Tensor, pattern: str) -> Tensor:
    """Perform an einsum operation."""
    ops, result = list(map(str.strip, pattern.split('->')))
    op1, op2 = list(map(str.strip, ops.split(',')))

    out_data = np.einsum(pattern, a.data, b.data) # type: ignore
    out = Tensor(out_data, [a, b], a.requires_grad or b.requires_grad, op=pattern)

    def einsum_backward() -> None:
        if a.requires_grad:
            a.grad += np.einsum(f'{op2},{result}->{op1}', b.data, out.grad) # type: ignore
        if b.requires_grad:
            b.grad += np.einsum(f'{op1},{result}->{op2}', a.data, out.grad) # type: ignore

    out._backward = einsum_backward
    return out


def rearrange(x: Tensor, pattern: str) -> Tensor:
    """Rearrange the tensor according to the pattern."""
    old, new = list(map(str.strip, pattern.split('->')))

    out_data = np.einsum(pattern, x.data) # type: ignore
    out = Tensor(out_data, [x], x.requires_grad, pattern)

    def rearrange_backward() -> None:
        if x.requires_grad:
            x.grad += np.einsum(f'{new}->{old}', out.grad) # type: ignore

    out._backward = rearrange_backward
    return out
