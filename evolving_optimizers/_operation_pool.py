import warnings
import random
from collections.abc import Sequence, Callable

from ._bases import BaseOperation, BasePool

def _get_weight(op: type[BaseOperation], cur: int, reg: float, weight_fn):
    if cur == 0 and (not op.ALLOW_FIRST): return 0
    return weight_fn(op) * op.WEIGHT * (1-reg) ** (cur * op.N_OPERANDS)

class RandomPool(BasePool):
    def __init__(self, pool: Sequence[type[BaseOperation]], reg: float = 0.1):
        self.pool = list(pool)
        self.reg = reg

    def select(self, cur: int, weight_fn: Callable[[type[BaseOperation]], float] = lambda x: 1):
        weights = [_get_weight(op, cur, self.reg, weight_fn) for op in self.pool]

        if sum(weights) == 0:
            warnings.warn("weights sum to 0, they will be set to None")
            weights = None

        return random.choices(self.pool, weights, k=1)[0]

