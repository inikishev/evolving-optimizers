import warnings
import random
from collections.abc import Sequence, Callable

from ._bases import BaseOperation, BasePool

def _get_weight(op: type[BaseOperation], cur: int, reg: float, weight_fn):
    if cur == 0 and (not op.ALLOW_FIRST): return 0
    return weight_fn(op) * op.WEIGHT * (1-reg) ** (cur * op.N_OPERANDS)

class RandomPool(BasePool):
    def __init__(self, pool: Sequence[type[BaseOperation]], reg: float = 0.1, must_contain: Sequence[type[BaseOperation]] = ()):
        super().__init__(must_contain=must_contain)
        self.pool = list(pool)
        self.reg = reg

    def select(self, cur: int, weight_fn: Callable[[type[BaseOperation]], float] = lambda x: 1):
        weights = [_get_weight(op, cur, self.reg, weight_fn) for op in self.pool]

        if sum(weights) == 0:
            warnings.warn("weights sum to 0, they will be set to None")
            weights = None

        return random.choices(self.pool, weights, k=1)[0]

class UnbiasedRandomPool(BasePool):
    """First it picks number of operands, and then it picks an operand with this many operands"""
    def __init__(self, pool: Sequence[type[BaseOperation]], reg: float = 0.1, must_contain: Sequence[type[BaseOperation]] = ()):
        super().__init__(must_contain=must_contain)
        self.pool = list(pool)

        self.reg = reg

    def select(self, cur: int, weight_fn: Callable[[type[BaseOperation]], float] = lambda x: 1):
        # we need to make sure weight_fn isn't 0 for all operands with selected N_OPERANDS
        pool = [op for op in self.pool if weight_fn(op) > 0]
        operand_nums = sorted(set(op.N_OPERANDS for op in pool))

        num_weights = [(1-self.reg) ** (cur * n) for n in operand_nums]
        n_operands = random.choices(operand_nums, num_weights, k=1)[0]

        pool = [op for op in self.pool if op.N_OPERANDS == n_operands]
        weights = [weight_fn(op) for op in pool]

        if sum(weights) == 0:
            warnings.warn("weights sum to 0, they will be set to None")
            weights = None

        return random.choices(pool, weights, k=1)[0]

