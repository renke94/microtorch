import sys
from collections.abc import Iterable
from typing import Any

from tqdm import tqdm

from microtorch.data.data import DataLoader
from microtorch.nn import Module
from microtorch.optim import Optimizer
from microtorch.tensor import Tensor
from microtorch.utils.loss_dict import LossDict


def progressbar(iterable: Iterable[Any], desc: str) -> tqdm: # type: ignore
    """Progress bar."""
    return tqdm(iterable, desc=desc, file=sys.stdout, ncols=100)

class Trainer:
    """Trainer class."""
    def __init__(
        self,
        model: Module,
        criterion: Module,
        optimizer: Optimizer,
        metrics: dict[str, Module]
    ) -> None:
        """Initialize the trainer."""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0
        self.metrics = metrics

    def compute_metrics(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Compute the metrics."""
        return {k: m(*args, **kwargs).data for k, m in self.metrics.items()}

    def train_step(self, batch: tuple[Tensor, Tensor]) -> LossDict:
        """Train step."""
        raise NotImplementedError

    def train_epoch(self, dataloader: DataLoader) -> None:
        """Train epoch."""
        loss = LossDict()
        with progressbar(dataloader, f'Epoch {self.epoch + 1}') as pbar: # type: ignore
            for step, batch in enumerate(pbar): # type: ignore
                loss += self.train_step(batch) # type: ignore
                total_loss = loss / (step + 1) # type: ignore
                total_loss = {k:f'{v:.4f}' for k, v in total_loss.items()} # type: ignore
                pbar.set_postfix(**total_loss) # type: ignore
        self.epoch += 1

    def fit(self, epochs: int, dataloader: DataLoader) -> None:
        """Fit the trainer."""
        for _ in range(epochs):
            self.train_epoch(dataloader)