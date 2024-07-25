# Binary Cross Entropy

$$ BCE = -\frac{1}{N} \left[ y_i \log(p_i) + (1 - p_i)\log(1-p_i) \right] $$

- $N$ is the number of samples
- $y_i$ is the true label for sample $i$ (0 or 1)
- $p$ is the predicted probability for sample $i$ (output of the sigmoid function ranging between 0 and 1).

## Derivative

1. For $y\log(p)$:
$$\frac{\partial}{\partial p}y\log(p) = \frac{y}{p}$$

2. For $(1-y)\log(1-p)$:
$$\frac{\partial}{\partial p}(1-y)\log(1-p) = \frac{-(1-y)}{1-p}$$

Combining these results:
$$\frac{\partial}{\partial p}y\log(p)+(1-y)\log(1-p) = \frac{y}{p} - \frac{1-y}{1-p} = \frac{y-p}{p(1-p)}$$

Derive the whole term:
$$\frac{\partial BCE}{\partial p} = -\frac{1}{N}\frac{y-p}{p(1-p)}$$

