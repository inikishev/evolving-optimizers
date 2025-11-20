import torch
from .._bases import BaseOperation, _format_hyperparam
from .._utils import TList
from .._hyperparameter import Continuous, Log, Choice, Integer
from .pools import decorator_common


@decorator_common
class Previous(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(n=Integer.partial(min=0, sigma=10))
    DEFAULTS = dict(n=1)

    def initialize(self, ref):
        self.history = TList()

    def forward(self, state):
        operand = self.get_operand(0, state)
        n = self.get_hyperparam("n")

        assert len(self.history) <= n

        if len(self.history) == n:
            if n == 0: return operand
            value = self.history.pop(0)
            self.history.append(operand)
            return value

        self.history.append(operand)
        return self.history[0]

@decorator_common
class LastDifference(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(n=Integer.partial(min=1, sigma=10))
    DEFAULTS = dict(n=1)

    def initialize(self, ref):
        self.history = TList()

    def forward(self, state):
        operand = self.get_operand(0, state)
        n = self.get_hyperparam("n")

        assert len(self.history) <= n

        if len(self.history) == n:
            value = self.history.pop(0)
            self.history.append(operand)
            return operand - value

        self.history.append(operand)

        if len(self.history) == 1:
            return operand

        return operand - self.history[0]


@decorator_common
class LastProduct(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(n=Integer.partial(min=1, sigma=10))
    DEFAULTS = dict(n=1)

    def initialize(self, ref):
        self.history = TList()

    def forward(self, state):
        operand = self.get_operand(0, state)
        n = self.get_hyperparam("n")

        assert len(self.history) <= n

        if len(self.history) == n:
            value = self.history.pop(0)
            self.history.append(operand)
            return operand * value

        self.history.append(operand)

        if len(self.history) == 1:
            return operand.square()

        return operand * self.history[0]
