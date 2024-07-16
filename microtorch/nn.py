from .tensor import Tensor


def param(t: Tensor) -> Tensor:
    t.requires_grad = True
    return t


class Module:
    def params(self):
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                yield v
            if isinstance(v, Module):
                for p in v.params():
                    yield p

        return filter(lambda v: v.requires_grad, (v for k, v in self.__dict__.items()))

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Sequential(Module):
    def __init__(self, *modules):
        self.modules = list(modules)

    def __repr__(self):
        modules_repr = "\n".join(f"  ({i}) {m.__repr__()}" for i, m in enumerate(self.modules))
        return f"Sequential(\n{modules_repr}\n)"

    def params(self):
        for m in self.modules:
            for p in m.params():
                yield p

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x


class Linear(Module):
    def __init__(self, fan_in, fan_out):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.w = param(Tensor.randn(fan_in, fan_out))
        self.b = param(Tensor.randn(1, fan_out))

    def __repr__(self):
        return f"Linear({self.fan_in}, {self.fan_out})"

    def __call__(self, x):
        return x @ self.w + self.b


class Sigmoid(Module):
    def __call__(self, x):
        return Tensor.sigmoid(x)


class ReLU(Module):
    def __call__(self, x):
        return Tensor.relu(x)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope

    def __call__(self, x):
        return Tensor.leaky_relu(x, negative_slope=self.negative_slope)

