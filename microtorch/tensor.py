from typing import Sequence

import numpy as np


class Index:
    def __init__(self, dim: int, ndims: int):
        if dim < 0:
            dim = ndims + dim
        self.slices: list[slice] = [slice(None) for _ in range(dim)]

    def __getitem__(self, item):
        return tuple(self.slices + [item])


class Tensor:
    def __init__(self, data, _children=(), requires_grad=False, _op=''):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._op = _op
        self._prev = set(_children)
        self.requires_grad = requires_grad

    def zero_grad(self):
        self.grad.fill(0.0)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        out = Tensor(self.data.T, _children=(self,), _op='transpose')

        def transpose_backward():
            self.grad += out.grad.T

        out._backward = transpose_backward
        return out

    def __repr__(self):
        # to appropriately aligns columns
        lines = self.data.__repr__().split('\n')
        lines = [" " + l for l in lines]
        return "Tensor" + "\n".join(lines)[6:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], _children=(self,), _op='indexing')

        def indexing_backward():
            self.grad[idx] += out.grad

        out._backward = indexing_backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.shape != other.shape:
            a, b = broadcast(self, other)
        else:
            a, b = self, other
        out = Tensor(a.data + b.data, _children=(a, b), _op='+')

        def _backward():
            a.grad += out.grad
            b.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.shape != other.shape:
            a, b = broadcast(self, other)
        else:
            a, b = self, other
        out = Tensor(a.data * b.data, _children=(a, b), _op='*')

        def _backward():
            a.grad += b.data * out.grad
            b.grad += a.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, _children=(self,), _op='**')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __matmul__(self, other):
        return self.mv_mult(other)

    def mv_mult(self, other):
        out = Tensor(self.data @ other.data, _children=(self, other), _op='@')

        def _backward():
            # df/dw
            #             print(f"{self.label}.grad = {out.label}.grad @ {other.label}.T")
            self.grad += out.grad @ other.data.T
            # df/dx
            #             print(f"{other.label}.grad = {self.label}.T @ {out.label}.grad")
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), _children=(self,), _op='relu')

        def _backward():
            self.grad += np.where(self.data > 0, 1, 0) * out.grad

        out._backward = _backward

        return out

    def leaky_relu(self, negative_slope=0.01):
        o = np.maximum(self.data, 0) + negative_slope * np.minimum(self.data, 0)

        out = Tensor(o, _children=(self,), _op='leaky_relu')

        def _backward():
            self.grad += np.where(self.data > 0, 1, negative_slope) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        o = 1 / (1 + np.exp(-self.data))

        out = Tensor(o, _children=(self,), _op='sigmoid')

        def _backward():
            self.grad += o * (1 - o) * out.grad

        out._backward = _backward

        return out

    # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    def softmax(self, dim: int = -1):
        m = self.data.max()
        e = np.exp(self.data - m)
        s = e / e.sum(axis=dim, keepdims=True)
        out = Tensor(s, _children=(self,), _op='softmax')

        def softmax_backward():
            _s = np.moveaxis(s, dim, -1)
            shape = _s.shape
            _s = _s.reshape(-1, _s.shape[-1])
            grad = np.zeros_like(_s)
            for i in range(len(_s)):
                si = _s[i].reshape(1, -1)
                j = np.diagflat(si) - si.T @ si  # jacobian matrix
                grad[i] += j.sum(-1)
            grad = grad.reshape(shape)
            grad = np.moveaxis(grad, -1, dim)
            self.grad += grad

        out._backward = softmax_backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), _children=(self,), _op='reshape')

        def reshape_backward():
            self.grad += out.grad.reshape(self.shape)

        out._backward = reshape_backward
        return out

    def flatten(self, start_dim=0, end_dim=-1):
        shape = list(self.shape)
        shape[end_dim] = -1
        del shape[start_dim:end_dim]
        return self.reshape(*shape)

    def permute(self, *dims):
        out = Tensor(self.data.transpose(*dims), _children=(self,), _op='permute')

        def permute_backward():
            r = {b: a for a, b in enumerate(dims)}
            r = [r[i] for i in range(len(r))]
            self.grad += out.grad.transpose(*r)

        out._backward = permute_backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), _children=(self,), _op='exp')

        def _backward():
            self.grad = out.data * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _children=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out

    def l2(self, target):
        o = np.sum((self.data - target.data) ** 2) / len(target)

        out = Tensor(o, _children=(self,))

        def _backward():
            self.grad += (self.data - target.data) * out.grad * 2 / len(target)

        out._backward = _backward

        return out

    def sum(self):
        # TODO: Handle dims
        out = Tensor(np.sum(self.data), _children=(self,), _op='sum')

        def _backward():
            self.grad += out.grad

        out._backward = _backward

        return out

    def chunk(self, chunks: int, dim: int = 0):
        out = np.split(self.data, chunks, axis=dim)
        out = tuple([Tensor(t, _children=(self,), _op='chunk') for t in out])

        size = self.shape[dim] // chunks
        index = Index(dim, self.dim)

        def backward_factory(t: Tensor, idx):
            def chunk_backward():
                self.grad[idx] += t.grad

            t._backward = chunk_backward

        for i, t in enumerate(out):
            backward_factory(t, index[i * size:i * size + size])

        return out

    def split(self, split_size_or_sections, dim: int = 0):
        out = split_numpy(self.data, split_size_or_sections, dim)
        out = tuple([Tensor(t, _children=(self,), _op='split') for t in out])

        index = Index(dim, self.dim)

        def backward_factory(t: Tensor, idx):
            def split_backward():
                self.grad[idx] += t.grad

            t._backward = split_backward

        sections = [0] + [t.shape[dim] for t in out]
        sections = np.cumsum(sections)
        for t, start, end in zip(out, sections[:-1], sections[1:]):
            backward_factory(t, index[start:end])

        return out

    def squeeze(self, dim: int):
        assert self.shape[dim] == 1, "squeeze dimension must be 1"
        shape = list(self.shape)
        del shape[dim]
        return self.reshape(*shape)

    def unsqueeze(self, dim: int):
        shape = list(self.shape)
        shape.insert(dim, 1)
        return self.reshape(*shape)

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        #         print([n.label for n in topo])

        self.grad += 1.0

        for node in reversed(topo):
            node._backward()

    @staticmethod
    def ones(*shape):
        return Tensor(np.ones(shape))

    @staticmethod
    def zeros(*shape):
        return Tensor(np.zeros(shape))

    @staticmethod
    def randn(*shape):
        return Tensor(np.random.randn(*shape))


def check_shape(a, b):
    return np.argwhere(np.array(a.shape) != np.array(b.shape)).flatten()


def fold_broadcast(a, dims):
    for d in dims:
        a = np.sum(a, axis=d, keepdims=True)
    return a


def broadcast(a: Tensor, b: Tensor):
    _a, _b = np.broadcast_arrays(a.data, b.data)

    _a = Tensor(_a, _children=(a,), _op='broadcast')
    a_dims = check_shape(_a.data, a.data)

    def a_backward():
        a.grad += fold_broadcast(_a.grad, a_dims)

    _a._backward = a_backward

    _b = Tensor(_b, _children=(b,), _op='broadcast')
    b_dims = check_shape(_b.data, b.data)

    def b_backward():
        b.grad += fold_broadcast(_b.grad, b_dims)

    _b._backward = b_backward
    return _a, _b


def split_numpy(arr: np.ndarray, split_size_or_sections, axis: int = 0, append_remainder=False):
    if isinstance(split_size_or_sections, int):
        s = arr.shape[axis] / split_size_or_sections
        if s.is_integer():
            s = int(s)
        else:
            raise ValueError(f"arr shape[{axis}] ({arr.shape[axis]}) is not divisable by {split_size_or_sections}")
        return np.split(arr, s, axis=axis)
    else:
        r = [s is ... for s in split_size_or_sections]
        c = np.count_nonzero(r)
        if c > 1:
            raise ValueError(f"Ellipsis ... is used more than once in sections")
        elif c == 1:
            s = np.array(split_size_or_sections)
            s[r] = 0
            s[r] = arr.shape[axis] - s.sum()
        else:
            s = split_size_or_sections

        s = np.cumsum(s)
        splits = np.split(arr, s, axis=axis)
        del splits[-1]
        return splits


def stack(tensors: Sequence[Tensor], dim=0):
    out = np.stack([t.data for t in tensors], axis=dim)
    out = Tensor(out, _children=tensors, _op='stack')

    def stack_backward():
        grads = split_numpy(out.grad, 1, axis=dim)
        for t, g in zip(tensors, grads):
            t.grad += np.squeeze(g, axis=dim)

    out._backward = stack_backward
    return out


def concat(tensors: Sequence[Tensor], dim=0):
    out = np.concatenate([t.data for t in tensors], axis=dim)
    out = Tensor(out, _children=tensors, _op='concat')

    def concat_backward():
        grads = split_numpy(out.grad, [t.shape[dim] for t in tensors], axis=dim)
        for t, g in zip(tensors, grads):
            t.grad += g

    out._backward = concat_backward
    return out


def test():
    x = Tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])

    np.random.seed(42)
    w = Tensor.randn(2, 4);
    w.label = 'w'
    b = Tensor.randn(1, 4);
    b.label = 'b'

    xw = x @ w
    out = xw + b
    out.backward()

    print(w.grad)
    print(b.grad)


if __name__ == '__main__':
    test()
