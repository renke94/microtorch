from collections.abc import Callable, Iterator

import numpy as np

from microtorch.tensor import Tensor, stack


class Dataset:
    """Dataset base class."""
    def __len__(self) -> int:
        """Get the length of the dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get the item at the given index."""
        raise NotImplementedError


Batch = list[tuple[Tensor, Tensor]]

def default_collate_fn(batch: Batch) -> tuple[Tensor, Tensor]:
    """Default collate function."""
    x, y = zip(*batch)
    return stack(x), stack(y)


class DataLoader:
    """DataLoader class."""
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        collate_fn: Callable[[Batch], tuple[Tensor, Tensor]] | None = None
    ) -> None:
        """Initialize the DataLoader."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn or default_collate_fn

    def __len__(self) -> int:
        """Get the length of the DataLoader."""
        return len(self.dataset) // self.batch_size

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Iterate over the DataLoader."""
        n = len(self.dataset)
        b = self.batch_size
        idx = np.arange(n)
        if self.shuffle:
            np.random.shuffle(idx)
        for s in range(0, n, b):
            idx_batch = idx[s:s + b]
            batch = [self.dataset[i] for i in idx_batch]
            yield self.collate_fn(batch)
