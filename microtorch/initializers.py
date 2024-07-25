import numpy as np
from .tensor import Tensor


def xavier_uniform_initialization(fan_in: int, fan_out: int) -> Tensor:
    """
    Xavier Initialization or Glorot Initialization:
    - Biases are initialized be 0
    - Weights are initialized as follows

    :param fan_in: int specifying the number of input neurons
    :param fan_out: int specifying the number of output neurons
    :return: Tensor initialized according to Xavier initialization
    """
    scale = np.sqrt(6. / (fan_in + fan_out))
    weight = np.random.uniform(-scale, scale, size=(fan_in, fan_out))
    return Tensor(weight)

def kaiming_he_initialization(fan_in: int, fan_out: int) -> Tensor:
    """
    Kaiming Initialization or He Initialization:
    - Biases are initialized be 0
    - Weights are zero centered Gaussian with standard deviation (2/n)**0.5

    :param fan_in: int specifying the number of input neurons
    :param fan_out: int specifying the number of output neurons
    :return: Tensor initialized according to Kaiming He initialization
    """
    weight = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))
    return Tensor(weight)