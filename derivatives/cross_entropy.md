Perfect — you’re now talking about the cross-entropy loss with pre-computed probabilities (already softmaxed), i.e. you want:

$$L = -\frac{1}{N} \sum_i \log(p_{i, t_i})$$
where $p_{i}$ are probabilities (not logits).

Here’s a clean, numerically safe implementation compatible with your autograd Tensor class (similar style to your other functions):

⸻

✅ Implementation (using softmax probabilities)

```python
import numpy as np

def cross_entropy_from_probs(probs: Tensor, target) -> Tensor:
    """
    Cross-entropy loss given *probabilities* (already softmaxed).

    Args:
        probs: Tensor of shape (N, C), each row sums to 1.
        target: class indices (Tensor or array-like of shape (N,))
    Returns:
        Scalar loss Tensor.
    """
    p = probs.data

    # Target processing
    if isinstance(target, Tensor):
        t = target.data
    else:
        t = target
    t = np.asarray(t, dtype=np.int64)

    n = p.shape[0]
    rows = np.arange(n, dtype=np.int64)

    # Avoid log(0)
    eps = 1e-12
    clipped = np.clip(p, eps, 1.0)

    # Cross-entropy
    ce = -np.log(clipped[rows, t]).mean()

    out = Tensor(ce, children=[probs], requires_grad=probs.requires_grad, op="cross_entropy_from_probs")

    def backward_cross_entropy():
        if out.grad is None or not probs.requires_grad:
            return

        # one-hot targets
        onehot = np.zeros_like(p)
        onehot[rows, t] = 1.0

        # derivative: dL/dp = -(1/p) * onehot / N  →  simplify to (-onehot / (p * N))
        grad_p = -(onehot / np.clip(p, eps, 1.0)) / n * out.grad

        # If you later chain this with a softmax layer,
        # this gradient can propagate through softmax.backward()
        if probs.grad is None:
            probs.grad = np.zeros_like(p)
        probs.grad += grad_p

    out._backward = backward_cross_entropy
    return out
```

⸻

🔍 Mathematical correctness check

Given
$$L = -\frac{1}{N}\sum_i \log(p_{i, t_i})$$
you have

$$\frac{\partial L}{\partial p_{i,j}} =
\begin{cases}
-\frac{1}{N p_{i,j}}, & j = t_i \\
0, & \text{otherwise}
\end{cases}$$

That’s exactly what the grad_p line computes.

If you later multiply this with the gradient of the softmax $(\frac{\partial p}{\partial z})$ in backprop, you’ll recover the usual simplification $\frac{1}{N}(p - \text{onehot})$ you saw in your cross_entropy_with_logits_loss.

⸻

✅ Usage example

```python
# Suppose we already have softmax probabilities
probs = Tensor(np.array([[0.1, 0.7, 0.2],
                         [0.8, 0.1, 0.1]]), requires_grad=True)
target = np.array([1, 0])

loss = cross_entropy_from_probs(probs, target)

# Seed gradient for backprop
loss.grad = np.array(1.0)
loss._backward()

print("Loss:", loss.data)
print("Grad wrt probs:\n", probs.grad)
```

Expected:
- The loss ≈ average of -log(prob_true)
- Gradient: only the target column has -1/(p*n) entries.

⸻

Would you like me to show the combined version that takes logits or probabilities automatically (detecting which one it is)?