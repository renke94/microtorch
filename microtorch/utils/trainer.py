import sys

from tqdm import tqdm

from .data import DataLoader
from .loss_dict import LossDict
from ..nn import Module
from ..optim import Optimizer


def progressbar(iterable, desc):
    return tqdm(iterable, desc=desc, file=sys.stdout)

class Trainer:
    def __init__(self,
                 model: Module,
                 criterion: Module,
                 optimizer: Optimizer,
                 metrics: dict[str, Module]):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0
        self.metrics = metrics

    def compute_metrics(self, *args, **kwargs) -> dict[str, float]:
        return {k: m(*args, **kwargs).data for k, m in self.metrics.items()}

    def train_step(self, batch) -> LossDict:
        pass

    def train_epoch(self, dataloader: DataLoader):
        loss = LossDict()
        with progressbar(dataloader, f"Epoch {self.epoch + 1}") as pbar:
            for step, batch in enumerate(pbar):
                loss += self.train_step(batch)
                total_loss = loss / (step + 1)
                total_loss = {k:f"{v:.4f}" for k, v in total_loss.items()}
                pbar.set_postfix(**total_loss)
        self.epoch += 1

    def fit(self, epochs: int, dataloader: DataLoader):
        for _ in range(epochs):
            self.train_epoch(dataloader)