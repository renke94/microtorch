from collections.abc import Callable, Collection, Sequence
from types import EllipsisType
from typing import Any, Union

import numpy as np

Operand = Union['Tensor', float, int, np.ndarray[Any, Any]]
ArrayType = Union['Tensor', np.ndarray[Any, Any]]
IndexParam = Union[int, tuple[int, ...], slice, 'Tensor', np.ndarray[Any, Any]]

def _make_tensor(data: Operand) -> 'Tensor':
    """Make a tensor from the data."""
    if isinstance(data, Tensor):
        return data
    elif isinstance(data, float | int | np.ndarray):
        return Tensor(data=data, requires_grad=False)


def _sum_to_shape(grad: np.ndarray[Any, Any], target_shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
    """Sum gradient over broadcasted dimensions to match target shape."""
    if grad.shape == target_shape:
        return grad

    # Handle leading dimensions that don't exist in target: e.g. shapes [2] to [3, 2]
    # Afterwards, grad has the same number of dimensions as the target_shape -> len(grad.shape) == len(target_shape)
    for _ in range(len(grad.shape) - len(target_shape)):
        grad = np.sum(grad, axis=0)

    # Collect the dimensions where target has size 1 but grad doesn't
    sum_dims: list[int] = []
    for dim, (target_size, grad_size) in enumerate(zip(target_shape, grad.shape)):
        if target_size == 1 and grad_size != 1:
            sum_dims.append(dim)

    # Sum over the collected dimensions
    if sum_dims:
        grad = np.sum(grad, axis=tuple(sum_dims), keepdims=True)

    return grad


class Index:
    """Index factory class to create slices at a given dimension."""

    def __init__(self, dim: int, ndims: int) -> None:
        """Initialize the index factory."""
        if dim < 0:
            dim = ndims + dim
        self.slices: list[slice] = [slice(None) for _ in range(dim)]

    def __getitem__(self, item: int | slice) -> tuple[slice | int, ...]:
        """Get the index at the given dimension."""
        return tuple(self.slices + [item])


class Tensor:
    """Tensor class."""

    def __init__(self,
                 data: Any,
                 children: Collection['Tensor'] = [],
                 requires_grad: bool = False,
                 op: str = ''
                ) -> None:
        """Initialize the tensor."""
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        self.data: np.ndarray[Any, Any] = data
        self.grad: np.ndarray[Any, Any] = np.zeros_like(self.data)
        self.requires_grad: bool = requires_grad
        self.op: str = op
        self.children: set[Tensor] = set(children)
        self._backward: Callable[[], None] = lambda: None


    def backward(self) -> None:
        """Backpropagate the gradient through the tensor."""
        topology = list[Tensor]()
        visited = set[Tensor]()

        def build_topology(node: Tensor) -> None:
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    build_topology(child)
                topology.append(node)

        build_topology(self)

        self.grad += 1.0

        for node in reversed(topology):
            node._backward()


    def __str__(self) -> str:
        if self.data.ndim == 0:
            return str(self.data)
        return self.__repr__()


    def __repr__(self) -> str:
        lines = repr(self.data).split('\n')
        lines = [' ' + l for l in lines]
        return self.__class__.__name__ + '\n'.join(lines)[6:]


    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the tensor."""
        return self.data.shape


    @property
    def dtype(self) -> np.dtype[Any]:
        """Get the dtype of the tensor."""
        return self.data.dtype


    @property
    def dim(self) -> int:
        """Get the number of dimensions of the tensor."""
        return self.data.ndim


    # ================================
    # Initializers
    # ================================

    @classmethod
    def ones(cls, *shape: int, dtype: np.dtype[Any] = np.dtype(np.float32), requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of ones."""
        return cls(data=np.ones(shape, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def ones_like(cls, input: 'Tensor', requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of ones with the same shape as the input tensor."""
        return cls(data=np.ones_like(input.data), requires_grad=requires_grad)

    @classmethod
    def zeros(cls, *shape: int, dtype: np.dtype[Any] = np.dtype(np.float32), requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of zeros."""
        return cls(data=np.zeros(shape, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def zeros_like(cls, input: 'Tensor', requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of zeros with the same shape as the input tensor."""
        return cls(data=np.zeros_like(input.data), requires_grad=requires_grad)

    @classmethod
    def eye(cls, n: int, dtype: np.dtype[Any] = np.dtype(np.float32), requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of ones with the same shape as the input tensor."""
        return cls(data=np.eye(n, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def randn(cls, *shape: int, requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of random numbers from a normal distribution."""
        return cls(data=np.random.randn(*shape), requires_grad=requires_grad)

    @classmethod
    def uniform(cls, low: float, high: float, shape: tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of random numbers from a uniform distribution."""
        return cls(data=np.random.uniform(low, high, size=shape), requires_grad=requires_grad)

    @classmethod
    def arange(cls, start: int, end: int, step: int = 1) -> 'Tensor':
        """Create a tensor of evenly spaced values."""
        return cls(data=np.arange(start, end, step), requires_grad=False)

    @classmethod
    def tril(cls, input: 'Tensor', diagonal: int = 0) -> 'Tensor':
        """Return the lower triangular part of the matrix."""
        return cls(data=np.tril(input.data, k=diagonal), requires_grad=False)

    def item(self) -> Any:
        """Get the item of the tensor."""
        return self.data.item()

    def to_int(self) -> 'Tensor':
        """Convert the tensor to an integer tensor."""
        return Tensor(self.data.astype(int), [], False)

    def to_float(self) -> 'Tensor':
        """Convert the tensor to a float tensor."""
        return Tensor(self.data.astype(float), [], False)

    def to_bool(self) -> 'Tensor':
        """Convert the tensor to a boolean tensor."""
        return Tensor(self.data.astype(bool), [], False)

    def zero_grad(self) -> None:
        """Zero the gradient of the tensor."""
        self.grad.fill(0)

    def __len__(self) -> int:
        """Get the length of the tensor."""
        return len(self.data)

    def __getitem__(self, index: IndexParam) -> 'Tensor':
        """Get the item of the tensor."""
        if isinstance(index, Tensor):
            index = index.data

        out = Tensor(
            data=self.data[index],
            children=[self],
            requires_grad=self.requires_grad,
            op='indexing',
        )

        def indexing_backward() -> None:
            if self.requires_grad:
                self.grad[index] += out.grad

        out._backward = indexing_backward
        return out

    # ================================
    # View operations
    # ================================

    @property
    def T(self) -> 'Tensor':
        """Get the transpose of the tensor."""
        out = Tensor(
            data=self.data.T,
            children=[self],
            requires_grad=self.requires_grad,
            op='transpose',
        )
        def transpose_backward() -> None:
            if self.requires_grad:
                self.grad = self.grad.T

        out._backward = transpose_backward
        return out


    def permute(self, *dims: int) -> 'Tensor':
        """Permute the dimensions of the tensor."""
        out = Tensor(self.data.transpose(*dims), [self], self.requires_grad, 'permute')

        def permute_backward() -> None:
            if self.requires_grad:
                position_mapping = { old: new for new, old in enumerate(dims) }
                reordered_dims = [position_mapping[i] for i in range(len(position_mapping))]
                self.grad += out.grad.transpose(*reordered_dims)

        out._backward = permute_backward
        return out

    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        """Transpose the tensor."""
        out = Tensor(self.data.swapaxes(dim0, dim1), [self], self.requires_grad, 'transpose')

        def transpose_backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.swapaxes(dim0, dim1)

        out._backward = transpose_backward
        return out

    def reshape(self, *shape: int) -> 'Tensor':
        """Reshape the tensor."""
        out = Tensor(self.data.reshape(*shape), [self], self.requires_grad, 'reshape')

        def reshape_backward() -> None:
            if self.requires_grad:
                self.grad = self.grad.reshape(self.shape)

        out._backward = reshape_backward
        return out


    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> 'Tensor':
        shape = list[int](self.shape)
        shape[end_dim] = -1
        del shape[start_dim:end_dim]
        return self.reshape(*shape)


    def squeeze(self, dim: int) -> 'Tensor':
        """Squeeze the tensor.

        Removes the dimension at the specified index. The dimension must be of size 1.
        """
        assert self.shape[dim] == 1, 'squeeze dimension must be 1'
        shape = list(self.shape)
        del shape[dim]
        return self.reshape(*shape)


    def unsqueeze(self, dim: int) -> 'Tensor':
        """Unsqueeze the tensor.

        Inserts a new dimension at the specified index. The dimension will be of size 1.
        """
        assert dim >= 0 and dim < self.dim, 'unsqueeze dimension out of range'
        shape = list(self.shape)
        shape.insert(dim, 1)
        return self.reshape(*shape)


    # ================================
    # Arithmetic operations
    # ================================

    def __add__(self, other: Operand) -> 'Tensor':
        """Add the tensor to another tensor."""
        other = _make_tensor(other)
        out = Tensor(self.data + other.data, [self, other], self.requires_grad or other.requires_grad, 'add')

        def add_backward() -> None:
            # Handle broadcasting: sum over broadcasted dimensions
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad, self.shape)

            if other.requires_grad:
                other.grad += _sum_to_shape(out.grad, other.shape)

        out._backward = add_backward
        return out

    def __radd__(self, other: Operand) -> 'Tensor':
        """Add the tensor to another tensor."""
        return self + other

    def __mul__(self, other: Operand) -> 'Tensor':
        """Multiply the tensor by another tensor."""
        other = _make_tensor(other)
        out = Tensor(self.data * other.data, [self, other], self.requires_grad or other.requires_grad, 'mul')

        def mul_backward() -> None:
            if self.requires_grad:
                self.grad += _sum_to_shape(other.data * out.grad, self.shape)

            if other.requires_grad:
                other.grad += _sum_to_shape(self.data * out.grad, other.shape)

        out._backward = mul_backward
        return out

    def __rmul__(self, other: Operand) -> 'Tensor':
        """Multiply the tensor by another tensor."""
        return self * other

    def __neg__(self) -> 'Tensor':
        """Negate the tensor."""
        return self * -1

    def __sub__(self, other: Operand) -> 'Tensor':
        """Subtract the tensor from another tensor."""
        return self + (-other)

    def __rsub__(self, other: Operand) -> 'Tensor':
        """Subtract the tensor from another tensor."""
        return -self + other

    def __pow__(self, other: int | float) -> 'Tensor':
        """Raise the tensor to a power."""
        assert isinstance(other, int | float), 'only supporting integer and float powers for now'
        out = Tensor(self.data ** other, [self], self.requires_grad, 'pow')

        def pow_backward() -> None:
            if self.requires_grad:
                self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = pow_backward
        return out

    def __truediv__(self, other: Operand) -> 'Tensor':
        """Divide the tensor by another tensor."""
        return self * (other ** -1)

    def __matmul__(self, other: ArrayType) -> 'Tensor':
        """Matrix multiply the tensor by another tensor."""
        other = _make_tensor(other)

        out = Tensor(self.data @ other.data, [self, other], self.requires_grad or other.requires_grad, 'matmul')

        def matmul_backward() -> None:
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad @ other.data.swapaxes(-2, -1), self.shape)

            if other.requires_grad:
                other.grad += _sum_to_shape(self.data.swapaxes(-2, -1) @ out.grad, other.shape)

        out._backward = matmul_backward
        return out

    def __rmatmul__(self, other: ArrayType) -> 'Tensor':
        """Matrix multiply the tensor by another tensor."""
        other = _make_tensor(other)
        return other @ self

    # ================================
    # Comparison operations
    # ================================

    def __eq__(self, other: object) -> 'Tensor':  # type: ignore[override]
        """Compare the tensor to another tensor."""
        if isinstance(other, Tensor):
            return Tensor(self.data == other.data, [], False, 'eq')
        elif isinstance(other, float | int | np.ndarray):
            return Tensor(self.data == other, [], False, 'eq')
        else:
            raise ValueError(f'Unsupported operand type for ==: {type(other)}')


    def __hash__(self) -> int:
        return super().__hash__()

    # ================================
    # Activation functions
    # ================================

    def sigmoid(self) -> 'Tensor':
        """Apply the sigmoid function to the tensor."""
        out_data = 1 / (1 + np.exp(-self.data))
        out = Tensor(out_data, [self], self.requires_grad, 'sigmoid')

        def sigmoid_backward() -> None:
            if self.requires_grad:
                self.grad += out_data * (1 - out_data) * out.grad

        out._backward = sigmoid_backward
        return out

    def tanh(self) -> 'Tensor':
        """Apply the tanh function to the tensor."""
        out_data = np.tanh(self.data)
        out = Tensor(out_data, [self], self.requires_grad, 'tanh')

        def tanh_backward() -> None:
            if self.requires_grad:
                self.grad += (1 - out_data ** 2) * out.grad

        out._backward = tanh_backward
        return out

    def relu(self) -> 'Tensor':
        """Apply the ReLU function to the tensor."""
        out = Tensor(np.maximum(0, self.data), [self], self.requires_grad, 'relu')

        def relu_backward() -> None:
            if self.requires_grad:
                self.grad += np.where(self.data > 0, 1, 0) * out.grad

        out._backward = relu_backward
        return out

    def leaky_relu(self, negative_slope: float = 0.01) -> 'Tensor':
        """Apply the Leaky ReLU function to the tensor."""
        out = Tensor(np.maximum(negative_slope * self.data, self.data), [self], self.requires_grad, 'leaky_relu')

        def leaky_relu_backward() -> None:
            if self.requires_grad:
                self.grad += np.where(self.data > 0, 1, negative_slope) * out.grad

        out._backward = leaky_relu_backward
        return out

    def softmax_with_jacobian(self, dim: int = -1) -> 'Tensor':
        """Apply the softmax function to the tensor.

        This implementation uses the Jacobian matrix to compute the gradient.
        https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        It doesn't produce the same gradients as the PyTorch implementation.
        """
        m = np.max(self.data, axis=dim, keepdims=True)
        e = np.exp(self.data - m)
        s = e / np.sum(e, axis=dim, keepdims=True)
        out = Tensor(s, [self], self.requires_grad, 'softmax')

        def softmax_backward() -> None:
            ss = s.swapaxes(dim, -1)
            s_shape = ss.shape
            sj = ss.reshape(-1, s_shape[-1])
            diags = np.stack([np.diag(v) for v in sj])
            sj = sj[..., None]
            j = sj@sj.swapaxes(-2, -1)
            j = diags - j
            j = j.reshape(*s_shape, s_shape[-1])

            g = out.grad.swapaxes(dim, -1)
            g = g[..., None].swapaxes(-2, -1)
            d = g@j
            d = d.swapaxes(-2, -1)[..., 0]
            d = d.swapaxes(dim, -1)
            self.grad += d

        out._backward = softmax_backward

        return out

    def softmax(self, dim: int = -1) -> 'Tensor':
        """Apply the softmax function to the tensor.

        This implementation uses the efficient Jacobian-vector product to compute the gradient.
        It doesn't produce the same gradients as the PyTorch implementation.
        """
        max_data = np.max(self.data, axis=dim, keepdims=True)
        exp = np.exp(self.data - max_data)
        out_data = exp / np.sum(exp, axis=dim, keepdims=True)
        out = Tensor(out_data, [self], self.requires_grad, 'softmax')

        def softmax_backward() -> None:
            if self.requires_grad:
                # Efficient Jacobian-vector product: (JVP implementation)
                # dx = y * (g - sum(g * y, axis=axis, keepdims=True))
                dot = np.sum(out.grad * out.data, axis=dim, keepdims=True)
                self.grad += out.data * (out.grad - dot)

        out._backward = softmax_backward
        return out

    # ================================
    # Utility functions
    # ================================

    def sum(self, dim: tuple[int, ...] | int | None = None, keepdims: bool = False) -> 'Tensor':
        """Sum the tensor over the specified dimensions."""
        out = Tensor(self.data.sum(axis=dim, keepdims=keepdims), [self], self.requires_grad, 'sum')

        axes = dim
        def sum_backward() -> None:
            if not self.requires_grad:
                return

            if axes is None:
                self.grad += np.ones_like(self.data) * out.grad
            else:
                dim = (axes,) if isinstance(axes, int) else axes
                dim = tuple(a if a >= 0 else a + self.dim for a in dim)

                if keepdims:
                    g_reshaped = out.grad
                else:
                    shape = list(self.shape)
                    for ax in dim:
                        shape[ax] = 1
                    g_reshaped = out.grad.reshape(shape)
                self.grad += np.ones_like(self.data) * g_reshaped

        out._backward = sum_backward
        return out


    def mean(self, dim: tuple[int, ...] | int | None = None, keepdims: bool = False) -> 'Tensor':
        """Mean the tensor over the specified dimensions."""
        if dim is not None:
            if isinstance(dim, int):
                dim = (dim, )
            numel = np.prod([self.shape[d] for d in dim])
        else:
            numel = np.prod(self.shape)

        out = Tensor(self.data.mean(axis=dim, keepdims=keepdims), [self], self.requires_grad, 'mean')

        def mean_backward() -> None:
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad / numel

        out._backward = mean_backward
        return out

    def var(self, dim: int | None = None, keepdims: bool = False) -> 'Tensor':
        """Variance the tensor over the specified dimensions."""
        if dim is None:
            numel = np.prod(self.shape)
        else:
            assert isinstance(dim, int), 'dim must be an integer or None'
            numel = self.shape[dim]

        out = Tensor(self.data.var(axis=dim, keepdims=keepdims), [self], self.requires_grad, 'var')

        def var_backward() -> None:
            if self.requires_grad:
                mean = self.data.mean(axis=dim, keepdims=True)
                self.grad += 2 * (self.data - mean) * out.grad / numel

        out._backward = var_backward
        return out


    def exp(self) -> 'Tensor':
        """Apply the exponential function to the tensor."""
        out = Tensor(np.exp(self.data), [self], self.requires_grad, 'exp')

        def exp_backward() -> None:
            if self.requires_grad:
                self.grad += out.data * out.grad

        out._backward = exp_backward
        return out

    def log(self) -> 'Tensor':
        """Apply the logarithmic function to the tensor."""
        out = Tensor(np.log(self.data), [self], self.requires_grad, 'log')

        def log_backward() -> None:
            if self.requires_grad:
                self.grad += 1 / self.data * out.grad

        out._backward = log_backward
        return out

    def abs(self) -> 'Tensor':
        """Apply the absolute value function to the tensor."""
        out = Tensor(np.abs(self.data), [self], self.requires_grad, 'abs')

        def abs_backward() -> None:
            if self.requires_grad:
                self.grad += np.sign(self.data) * out.grad

        out._backward = abs_backward
        return out


    def chunk(self, chunks: int, dim: int = 0) -> list['Tensor']:
        """Chunk the tensor into a list of tensors."""
        out = np.split(self.data, chunks, axis=dim)
        out = [Tensor(t, [self], self.requires_grad, 'chunk') for t in out]

        size = self.shape[dim] // chunks
        index = Index(dim, self.dim)

        def chunk_backward_factory(tensor: 'Tensor', idx: tuple[slice | int, ...]) -> Callable[[], None]:
            def chunk_backward() -> None:
                if self.requires_grad:
                    self.grad[idx] += tensor.grad
            return chunk_backward

        for i, chunk in enumerate(out):
            start = i * size
            end = start + size
            chunk._backward = chunk_backward_factory(chunk, index[start:end])

        return out

    def masked_fill(self, mask: ArrayType, value: float | int) -> 'Tensor':
        """Fill the tensor with a value where the mask is True."""
        if isinstance(mask, Tensor):
            mask = mask.data

        out = Tensor(np.where(mask, value, self.data), [self], self.requires_grad, 'masked_fill')

        def masked_fill_backward() -> None:
            if self.requires_grad:
                self.grad += np.where(mask, 0, 1) * out.grad

        out._backward = masked_fill_backward
        return out

    def tolist(self) -> list[Any]:
        """Convert the tensor to a list."""
        return self.data.tolist()

    def numpy(self) -> np.ndarray[Any, Any]:
        """Convert the tensor to a numpy array."""
        return self.data

    def detach(self) -> 'Tensor':
        """Detach the tensor from the graph."""
        return Tensor(self.data.copy(), [], False)


def split_numpy(
    arr: np.ndarray[Any, Any],
    split_size_or_sections: int | Sequence[int | slice | EllipsisType],
    axis: int = 0,
    append_remainder: bool = False
) -> list[np.ndarray[Any, Any]]:
    """Split the array into a list of arrays.

    Usage:
    >>> arr = np.array([1, 2, 3, 4])
    >>> split_numpy(arr, 2)
    [array([1, 2]), array([3, 4])]


    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> split_numpy(arr, [2, 3])
    [array([1, 2]), array([3, 4, 5])]

    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> split_numpy(arr, [..., 2])
    [array([1, 2, 3]), array([4, 5])]
    """
    if isinstance(split_size_or_sections, int):
        s = arr.shape[axis] / split_size_or_sections
        if s.is_integer():
            s = int(s)
        else:
            raise ValueError(f'arr shape[{axis}] ({arr.shape[axis]}) is not divisable by {split_size_or_sections}')
        return np.split(arr, s, axis=axis)
    else:
        r = [s == Ellipsis for s in split_size_or_sections]
        c = np.count_nonzero(r)
        if c > 1:
            raise ValueError('Ellipsis ... is used more than once in sections')
        elif c == 1:
            s = np.array(split_size_or_sections)
            s[r] = 0
            s[r] = arr.shape[axis] - s.sum()
        else:
            s = split_size_or_sections

        s = np.cumsum(s)  # type: ignore
        splits = np.split(arr, s, axis=axis)  # type: ignore
        del splits[-1]
        return splits


def stack(tensors: Sequence['Tensor'], dim: int = 0) -> 'Tensor':
    """Stack the tensors along the specified dimension."""
    out_data = np.stack([t.data for t in tensors], axis=dim)
    out = Tensor(out_data, tensors, any(t.requires_grad for t in tensors), 'stack')

    def stack_backward() -> None:
        grads = split_numpy(out.grad, 1, axis=dim)
        for t, g in zip(tensors, grads):
            t.grad += np.squeeze(g, axis=dim)

    out._backward = stack_backward
    return out


def concat(tensors: Sequence[Tensor], dim: int = 0) -> 'Tensor':
    """Concatenate the tensors along the specified dimension."""
    out_data = np.concatenate([t.data for t in tensors], axis=dim)
    out = Tensor(out_data, tensors, any(t.requires_grad for t in tensors), 'concat')

    def concat_backward() -> None:
        grads = split_numpy(out.grad, [t.shape[dim] for t in tensors], axis=dim)
        for t, g in zip(tensors, grads):
            t.grad += g

    out._backward = concat_backward
    return out


def multinomial(input: Tensor, num_samples: int) -> Tensor:
    """Sample from the multinomial distribution."""
    return Tensor(np.stack([
        [np.random.multinomial(n=1, pvals=p, size=None).argmax(axis=-1) for p in input.data]
        for _ in range(num_samples)
    ], axis=-1))



