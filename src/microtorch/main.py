import numpy as np
from tqdm import tqdm

import microtorch.nn as nn
from microtorch.losses import bce_with_logits_loss, cross_entropy_with_logits_loss
from microtorch.optim import Adam
from microtorch.tensor import Tensor

x = Tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])

y = Tensor([[0, 1, 1, 0]]).T


def xor_bce_with_logits() -> None:
    """XOR problem with BCE with logits loss."""
    np.random.seed(42)

    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.Sigmoid(),
        nn.Linear(16, 16),
        nn.Sigmoid(),
        nn.Linear(16, 1),
    )

    optimizer = Adam(model.params(), lr=0.01)

    with tqdm(range(200)) as pbar:
        for _ in pbar:
            loss = bce_with_logits_loss(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    pred = model(x).sigmoid()

    print(pred)

def xor_cross_entropy_loss() -> None:
    """XOR problem with cross entropy loss."""
    pred = Tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    # np.random.seed(42)

    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.Sigmoid(),
        nn.Linear(16, 16),
        nn.Sigmoid(),
        nn.Linear(16, 2),
        # nn.Softmax(dim=-1),
    )

    optimizer = Adam(model.params(), lr=0.001)

    with tqdm(range(1000)) as pbar:
        for _ in pbar:
            loss = cross_entropy_with_logits_loss(model(x), y)
            # print(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix(loss=loss.item()) # type: ignore

    pred = model(x).softmax(dim=-1)

    print(pred)


def main() -> None:
    """Main function to run the application."""
    xor_cross_entropy_loss()
    # xor_bce_with_logits()
