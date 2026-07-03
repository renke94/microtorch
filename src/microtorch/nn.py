import math
from os import PathLike
import pickle
from collections import OrderedDict
from collections.abc import Generator, Iterator
from typing import Any

import numpy as np

from microtorch.initializers import kaiming_he_initialization  #, xavier_uniform_initialization
from microtorch.tensor import Tensor, concat, multinomial


def param(t: Tensor) -> Tensor:
    """Make a tensor a parameter."""
    t.requires_grad = True
    return t


class Module:
    """Base class for all modules."""

    def __init__(self) -> None:
        """Initialize the module."""
        self.training = True

    def _set_training(self, training: bool) -> None:
        """Set the training mode of the module."""
        self.training = training
        for v in self.__dict__.values():
            if isinstance(v, Module):
                v._set_training(training)

    def train(self) -> 'Module':
        """Set the module to training mode."""
        self._set_training(True)
        return self

    def eval(self) -> 'Module':
        """Set the module to evaluation mode."""
        self._set_training(False)
        return self

    def params(self) -> Generator[Tensor, None, None]:
        """Get the parameters of the module."""
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                yield v
            if isinstance(v, Module):
                yield from v.params()

    def num_params(self) -> int:
        """Get the number of parameters of the module."""
        return sum(np.prod(p.shape) for p in self.params())  # type: ignore

    def state_dict(self) -> OrderedDict[str, Any]:
        """Get the state dictionary of the module."""
        modules: OrderedDict[str, Any] = OrderedDict()
        params: OrderedDict[str, Tensor] = OrderedDict()

        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                for name, value in v.state_dict().items():
                    modules[f'{k}.{name}'] = value
            elif isinstance(v, Tensor) and v.requires_grad:
                params[k] = v

        return modules | params


    def _set_param(self, key: list[str], value: Tensor) -> None:
        """Set the parameter of the module."""
        head, *tail = key
        param = getattr(self, head)
        if param is None:
            raise ValueError(f'Module {self.__class__.__name__} has no attribute {head}')
        if len(tail) == 0:
            assert isinstance(param, Tensor)
            param.data = value.data
        else:
            assert isinstance(param, Module)
            param._set_param(tail, value)


    def load_state_dict(self, state_dict: OrderedDict[str, Any]) -> None:
        """Load the state dictionary of the module."""
        for k, v in state_dict.items():
            self._set_param(k.split('.'), v)
        print('All keys matched successfully.')


    def save(self, path: PathLike[str] | str) -> None:
        state_dict = {k: v.data for k, v in self.state_dict().items()}
        with open(path, 'wb') as file:
            pickle.dump(state_dict, file)


    def load(self, path: PathLike[str] | str) -> None:
        with open(path, 'rb') as file:
            data = pickle.load(file)
        state_dict = OrderedDict({k: Tensor(v) for k, v in data.items()})
        self.load_state_dict(state_dict)


    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class ModuleList(Module):
    """Module list."""
    def __init__(self, modules: list[Module]) -> None:
        """Initialize the module list."""
        super().__init__()
        self.modules = modules

    def params(self) -> Generator[Tensor, None, None]:
        """Get the parameters of the module list."""
        for module in self.modules:
            yield from module.params()

    def state_dict(self) -> OrderedDict[str, Any]:
        """Get the state dictionary of the module list."""
        modules: OrderedDict[str, Any] = OrderedDict()
        for k, module in enumerate(self.modules):
            for name, value in module.state_dict().items():
                modules[f'{k}.{name}'] = value
        return modules

    def _set_param(self, key: list[str], value: Tensor) -> None:
        """Set the parameter of the module list."""
        head, *tail = key
        if not head.isdigit():
            raise ValueError(f'Invalid key: {key}')
        index = int(head)
        if index < 0 or index >= len(self.modules):
            raise ValueError(f'Module {self.__class__.__name__} has no module at index {index}')
        module = self.modules[index]
        module._set_param(tail, value)

    def __len__(self) -> int:
        """Get the length of the module list."""
        return len(self.modules)

    def __getitem__(self, index: int) -> Module:
        """Get the module at the given index."""
        return self.modules[index]

    def __setitem__(self, index: int, module: Module) -> None:
        """Set the module at the given index."""
        self.modules[index] = module

    def __iter__(self) -> Iterator[Module]:
        """Iterate over the module list."""
        return iter(self.modules)

    def __repr__(self) -> str:
        return f'ModuleList({self.modules})'


class Sequential(Module):
    """Sequential container."""

    def __init__(self, *modules: Module) -> None:
        """Initialize the sequential container."""
        super().__init__()
        self.modules = list(modules)

    def __repr__(self) -> str:
        modules_repr = '\n'.join(f'  ({i}) {m.__repr__()}' for i, m in enumerate(self.modules))
        return f'Sequential(\n{modules_repr}\n)'

    def params(self) -> Generator[Tensor, None, None]:
        """Get the parameters of the sequential container."""
        for m in self.modules:
            yield from m.params()

    def state_dict(self) -> OrderedDict[str, Any]:
        """Get the state dictionary of the sequential container."""
        modules: OrderedDict[str, Any] = OrderedDict()
        for k, m in enumerate(self.modules):
            for name, value in m.state_dict().items():
                modules[f'{k}.{name}'] = value
        return modules

    def _set_param(self, key: list[str], value: Tensor) -> None:
        """Set the parameter of the module list."""
        head, *tail = key
        if not head.isdigit():
            raise ValueError(f'Invalid key: {key}')
        index = int(head)
        if index < 0 or index >= len(self.modules):
            raise ValueError(f'Module {self.__class__.__name__} has no module at index {index}')
        module = self.modules[index]
        module._set_param(tail, value)

    def __getitem__(self, index: int) -> Module:
        """Get the module at the given index."""
        return self.modules[index]

    def __setitem__(self, index: int, module: Module) -> None:
        """Set the module at the given index."""
        self.modules[index] = module

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass through the sequential container."""
        for module in self.modules:
            x = module(x)
        return x


class Identity(Module):
    """Identity layer."""
    def __call__(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    """Linear layer."""
    def __init__(self, fan_in: int, fan_out: int, bias: bool = True) -> None:
        """Initialize the linear layer."""
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.weight = param(kaiming_he_initialization(fan_in, fan_out))
        self.bias = None
        if bias:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = param(Tensor.uniform(-bound, bound, shape=(1, fan_out)))

    def __repr__(self) -> str:
        return f'Linear({self.fan_in}, {self.fan_out})'

    def __call__(self, x: Tensor) -> Tensor:
        if self.bias is not None:
            return x @ self.weight + self.bias
        else:
            return x @ self.weight


class Sigmoid(Module):
    """Sigmoid activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        return Tensor.sigmoid(x)


class ReLU(Module):
    """ReLU activation function."""

    def __call__(self, x: Tensor) -> Tensor:
        return Tensor.relu(x)


class LeakyReLU(Module):
    """Leaky ReLU activation function."""

    def __init__(self, negative_slope: float = 0.01) -> None:
        """Initialize the leaky ReLU activation function."""
        super().__init__()
        self.negative_slope = negative_slope

    def __call__(self, x: Tensor) -> Tensor:
        return Tensor.leaky_relu(x, negative_slope=self.negative_slope)


class Softmax(Module):
    """Softmax activation function."""

    def __init__(self, dim: int = -1) -> None:
        """Initialize the softmax activation function."""
        super().__init__()
        self.dim = dim

    def __call__(self, x: Tensor) -> Tensor:
        return x.softmax(dim=self.dim)


class Flatten(Module):
    """Flatten layer."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        """Initialize the flatten layer."""
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)


class BatchNorm1d(Module):
    """Batch normalization layer."""

    def __init__(self, dim: int, momentum: float = 0.1) -> None:
        """Initialize the batch normalization layer."""
        super().__init__()
        self.dim = dim
        self.gamma = Tensor.ones(dim)
        self.beta = Tensor.zeros(dim)
        self.running_mean = Tensor.zeros(dim)
        self.running_var = Tensor.ones(dim)
        self.momentum = momentum
        self.eps = 1e-5

    def __call__(self, x: Tensor) -> Tensor:
        if self.training:
            xmean = x.mean(dim=0, keepdims=True)
            xmean.requires_grad = False
            xvar = x.var(dim=0, keepdims=True)
            xvar.requires_grad = False
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / (xvar + self.eps) ** 0.5
        self.out = self.gamma * xhat + self.beta

        if self.training:
            # self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * xmean
            # self.running_var = self.momentum * self.running_var + (1 - self.momentum) * xvar
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out


class LayerNorm1d(Module):
    """Layer normalization layer."""

    def __init__(self, dim: int) -> None:
        """Initialize the batch normalization layer."""
        super().__init__()
        self.eps = 1e-5
        self.gamma = Tensor.ones(dim, requires_grad=True)
        self.beta = Tensor.zeros(dim, requires_grad=True)


    def __call__(self, x: Tensor) -> Tensor:
        xmean = x.mean(dim=-1, keepdims=True).detach()
        xvar = x.var(dim=-1, keepdims=True).detach()
        xhat = (x - xmean) / (xvar + self.eps) ** 0.5
        out = self.gamma * xhat + self.beta
        return out


class Dropout(Module):
    """Dropout layer."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize the dropout layer."""
        super().__init__()
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        return x.masked_fill(np.random.binomial(1, self.p, size=x.shape), 0)


class Embedding(Module):
    """Embedding layer."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_table = Tensor.randn(vocab_size, embedding_dim, requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        return self.embedding_table[input]


class MultiheadAttention(Module):
    """Multi-head attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = embed_dim
        self.nhead = num_heads
        self.dropout = Dropout(dropout) if dropout > 0.0 else Identity()
        self.head_size = embed_dim // num_heads
        self.scale = self.head_size ** -0.5
        self.qkv = Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = Linear(embed_dim, embed_dim)

    def forward(self, input: Tensor, is_causal: bool = True) -> Tensor:
        B, N, E = input.shape
        q, k, v = self.qkv(input).chunk(3, dim=-1)  # B N E
        q = q.reshape(B, N, self.nhead, self.head_size).transpose(1, 2)  # B H N E
        k = k.reshape(B, N, self.nhead, self.head_size).transpose(1, 2)  # B H N E
        v = v.reshape(B, N, self.nhead, self.head_size).transpose(1, 2)  # B H N E
        weights = q @ k.transpose(-2, -1) * self.scale
        if is_causal:
            weights = weights.masked_fill(Tensor.tril(Tensor.ones(N, N)) == 0, float('-inf'))
        weights = weights.softmax(dim=-1)
        weights = self.dropout(weights)
        out = weights @ v  # B H N E
        out = out.transpose(1, 2).reshape(B, N, E)
        out = self.proj(out)
        return out


class TransformerEncoderLayer(Module):
    """Transformer encoder layer."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = LayerNorm1d(d_model)
        self.norm2 = LayerNorm1d(d_model)
        self.attention = MultiheadAttention(d_model, nhead, dropout)
        self.feed_forward = Sequential(
            Linear(d_model, dim_feedforward),
            LeakyReLU(),
            Linear(dim_feedforward, d_model),
            Dropout(dropout) if dropout > 0.0 else Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class TransformerEncoder(Module):
    """Transformer encoder."""

    def __init__(self, d_model: int, nhead: int, num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = Sequential(
            *[
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ],
            LayerNorm1d(d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TransformerDecoderLayer(Module):
    """Transformer decoder layer."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = LayerNorm1d(d_model)
        self.norm2 = LayerNorm1d(d_model)
        self.attention = MultiheadAttention(d_model, nhead, dropout)
        self.feed_forward = Sequential(
            Linear(d_model, dim_feedforward),
            LeakyReLU(),
            Linear(dim_feedforward, d_model),
            Dropout(dropout) if dropout > 0.0 else Identity(),
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        is_causal: bool = True
    ) -> Tensor:
        raise NotImplementedError


class GPTBlock(Module):
    """GPT block."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = LayerNorm1d(d_model)
        self.norm2 = LayerNorm1d(d_model)
        self.attention = MultiheadAttention(d_model, nhead, dropout)
        self.feed_forward = Sequential(
            Linear(d_model, dim_feedforward),
            LeakyReLU(),
            Linear(dim_feedforward, d_model),
            Dropout(dropout) if dropout > 0.0 else Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x), is_causal=True)
        x = x + self.feed_forward(self.norm2(x))
        return x


class GPT(Module):
    """GPT."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        context_length: int = 64,
    ) -> None:
        super().__init__()
        self.token_embedding_table = Tensor.randn(vocab_size, d_model, requires_grad=True)
        self.position_embedding_table = Tensor.randn(context_length, d_model, requires_grad=True)
        self.context_length = context_length
        self.blocks = Sequential(
            *[GPTBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)],
            LayerNorm1d(d_model),
        )
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, input: Tensor) -> Tensor:
        _, T = input.shape
        tok_emb = self.token_embedding_table[input]
        pos_emb = self.position_embedding_table[Tensor.arange(0, T)]
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.lm_head(x)
        return x

    def generate(
        self,
        input: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0
    ) -> Tensor:
        for _ in range(max_new_tokens):
            logits = self(input[:, -self.context_length:])  # type: ignore
            logits = logits[:, -1, :] / temperature
            probs = logits.softmax(dim=-1)
            idx_next = multinomial(probs, num_samples=1)
            input = concat([input, idx_next], dim=-1)
        return input

    def generate_stream(
        self,
        input: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0
    ) -> Generator[Tensor, None, None]:
        for _ in range(max_new_tokens):
            logits = self(input[:, -self.context_length:])  # type: ignore
            logits = logits[:, -1, :] / temperature
            probs = logits.softmax(dim=-1)
            idx_next = multinomial(probs, num_samples=1)
            input = concat([input, idx_next], dim=-1)
            yield idx_next
