import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from microtorch import nn
from microtorch.attention import MultiheadAttentionFast, MultiheadAttentionSlow
from microtorch.losses import cross_entropy_with_logits_loss
from microtorch.optim import Adam
from microtorch.tensor import Tensor, concat, multinomial, stack

MHA_TYPE = type[MultiheadAttentionSlow | MultiheadAttentionFast]

with open('datasets/tiny-shakespeare.txt', 'r') as file:
    text = file.read()

chars = sorted(set(text))
vocab_size = len(chars)

stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(l: list[int]) -> str:
    return ''.join([itos[i] for i in l])

data = Tensor(encode(text))
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

np.random.seed(1337)
batch_size = 16
block_size = 64

def get_batch(split: str) -> tuple[Tensor, Tensor]:
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))  # type: ignore
    x = stack([data[i:i+block_size] for i in ix])
    y = stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


class FeedForward(nn.Module):
    """Feed-forward network."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class Block(nn.Module):
    """Block."""

    def __init__(self, embedding_dim: int, _mha_class: MHA_TYPE) -> None:
        super().__init__()
        self.sa = _mha_class(embedding_dim, num_heads=4)
        self.ff = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm1d(embedding_dim)
        self.ln2 = nn.LayerNorm1d(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.sa(self.ln1(x), is_causal=True)
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        num_layers: int = 6,
        num_heads: int = 4,
        _mha_class: MHA_TYPE = MultiheadAttentionSlow
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(context_length, embedding_dim)
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, _mha_class) for _ in range(num_layers)],
            nn.LayerNorm1d(embedding_dim),
        )
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

        self.optimizer = Adam(self.params(), lr=1e-4, weight_decay=0.0)

    def forward(self, idx: Tensor) -> Tensor:
        _, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(Tensor.arange(0, T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.context_length:])  # type: ignore
            logits = logits[:, -1, :]
            probs = logits.softmax(dim=-1)
            idx_next = multinomial(probs, num_samples=1)
            idx = concat([idx, idx_next], dim=-1)
        return idx

    def generate_stream(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.context_length:])  # type: ignore
            logits = logits[:, -1, :] / temperature
            probs = logits.softmax(dim=-1)
            idx_next = multinomial(probs, num_samples=1)
            idx = concat([idx, idx_next], dim=-1)
            yield idx_next[0].tolist()

    def train_step(self, idx: Tensor, targets: Tensor) -> float:
        logits = self(idx)
        loss = cross_entropy_with_logits_loss(logits, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, epoch: int):
        self.train()
        with tqdm(range(5000), desc=f'Epoch {epoch}', file=sys.stdout) as pbar:
            total_loss: list[float] = []
            for _ in pbar:
                loss = self.train_step(*get_batch('train'))
                total_loss.append(loss)
                pbar.set_postfix(loss=np.mean(total_loss))  # type: ignore

        return total_loss


def main() -> None:
    """Main entry point for the GPT training script."""
    NUM_EPOCHS = 10
    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=128,
        num_layers=6,
        num_heads=8,
        context_length=block_size,
        _mha_class=MultiheadAttentionSlow
    )

    model_checkpoints = list(Path('.').glob('gpt_epoch_*.mt'))
    model_checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    if model_checkpoints:
        print('continue learning from checkpoint:', model_checkpoints[-1])
        model.load(model_checkpoints[-1])
        epoch = int(model_checkpoints[-1].stem.split('_')[-1])
    else:
        print('start training from scratch')
        epoch = 1

    print('Number of parameters:', model.num_params())
    for _ in range(NUM_EPOCHS):
        epoch += 1
        model.train_epoch(epoch=epoch)
        if epoch % 1 == 0:
            print('Generating text...')
            idx = Tensor([[0]])
            for token in model.generate_stream(idx, max_new_tokens=300, temperature=0.6):
                print(decode(token), end='', flush=True)
            print()

        model.save(f'gpt_epoch_{epoch}.mt')
