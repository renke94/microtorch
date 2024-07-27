import pandas as pd


class LossDict(pd.Series):
    """
    Loss dictionary class extending the pandas Series. This is a helper class to accumulate and average loss values
    during training without the need to store the values in lists.

    Example:
        >>> loss_dict = LossDict(l1_loss=0.1, l2_loss=0.2)
        >>> loss_dict += LossDict(l1_loss=0.3, l2_loss=0.4)
        >>> print(loss_dict)
        l1_loss    0.4
        l2_loss    0.6
        dtype: float64
    """
    def __init__(self, **kwargs):
        super(LossDict, self).__init__(kwargs)

    def item(self) -> float:
        n = len(self)
        assert n == 1, f"A LossDict with {n} elements cannot be converted to Scalar"
        return self.iloc[0]

    def __add__(self, other):
        if isinstance(other, dict):
            other = LossDict(**other)
        out = pd.Series.add(self, other, fill_value=0.0)
        out.__class__ = LossDict
        return out

    def __iadd__(self, other):
        return self + other

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = super(LossDict, self).__neg__()
        out.__class__ = LossDict
        return out

    def __sub__(self, other):
        if isinstance(other, dict):
            other = LossDict(**other)
        out = pd.Series.sub(self, other, fill_value=0.0)
        out.__class__ = LossDict
        return out

    def __isub__(self, other):
        return self - other

    def __rsub__(self, other):
        return -self + other

    def __truediv__(self, other):
        out = super(LossDict, self).__truediv__(other)
        out.__class__ = LossDict
        return out

    def __mul__(self, other):
        if isinstance(other, dict):
            other = LossDict(**other)
        out = super(LossDict, self).__mul__(other)
        out.__class__ = LossDict
        return out

    def __imul__(self, other):
        return self * other

    def __rmul__(self, other):
        return self * other

    @staticmethod
    def from_locals(local_vars: dict) -> 'LossDict':
        f = lambda i: 'loss' in i[0] and i[0] != 'loss'
        local_vars = {k: v.item() for k, v in filter(f, local_vars.items())}
        return LossDict(**local_vars)


def make_loss_dict(local_vars: dict) -> LossDict:
    f = lambda i: 'loss' in i[0] and i[0] != 'loss'
    local_vars = {k: v for k, v in filter(f, local_vars.items())}
    return LossDict(**local_vars)