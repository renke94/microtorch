import numpy as np


class Tensor:
    def __init__(self, data, _children=(), requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
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
        return Tensor(self.data.T)

    def __repr__(self):
        # to appropriately aligns columns
        lines = self.data.__repr__().split('\n')
        lines = [" " + l for l in lines]
        return "Tensor" + "\n".join(lines)[6:]

    def __len__(self):
        return len(self.data)

    def broadcast(self, other):
        shape = np.broadcast(self.data, other.data).shape

        if other.shape != shape:
            if other.shape[0] == 1:
                return Tensor.ones((shape[0], 1)) @ other
            else:
                return other @ Tensor.ones((1, shape[0]))
        else:
            return other

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        other = self.broadcast(other)
        out = Tensor(self.data + other.data, _children=(self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        other = self.broadcast(other)
        out = Tensor(self.data * other.data, _children=(self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

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
        out = Tensor(self.data ** other, _children=(self,))

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __matmul__(self, other):
        return self.mv_mult(other)

    def mv_mult(self, other):
        out = Tensor(self.data @ other.data, _children=(self, other))

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
        out = Tensor(np.maximum(self.data, 0), _children=(self,))

        def _backward():
            self.grad += np.where(self.data > 0, 1, 0) * out.grad

        out._backward = _backward

        return out

    def leaky_relu(self, negative_slope=0.01):
        o = np.maximum(self.data, 0) + negative_slope * np.minimum(self.data, 0)

        out = Tensor(o, _children=(self,))

        def _backward():
            self.grad += np.where(self.data > 0, 1, negative_slope) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        o = 1 / (1 + np.exp(-self.data))

        out = Tensor(o, _children=(self,))

        def _backward():
            self.grad += o * (1 - o) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Tensor(np.exp(self.data), _children=(self,))

        def _backward():
            self.grad = out.data * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _children=(self,))

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
        out = Tensor(np.sum(self.data), _children=(self,))

        def _backward():
            self.grad += out.grad

        out._backward = _backward

        return out

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
    def ones(shape):
        return Tensor(np.ones(shape))

    @staticmethod
    def randn(*shape):
        return Tensor(np.random.randn(*shape))