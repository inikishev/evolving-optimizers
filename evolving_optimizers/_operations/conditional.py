import torch
from torch.nn import functional as F

from .._bases import BaseOperation, _format_hyperparam
from .._hyperparameter import Choice, Continuous, Log
from .pools import decorator_common


@decorator_common
class CompareScalar(BaseOperation):
    N_OPERANDS = 3
    HYPERPARAMS = dict(threshold=Continuous.partial(sigma=10))
    DEFAULTS = dict(threshold=0)

    def forward(self, state):
        mask = self.get_operand(0, state) > self.get_hyperparam("threshold")
        true = self.get_operand(1, state)
        false = self.get_operand(2, state)
        return torch.where(mask, true, false)

@decorator_common
class Threshold(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(
        threshold=Continuous.partial(sigma=10),
        below=Continuous.partial(sigma=3),
        above=Continuous.partial(sigma=3)
    )
    DEFAULTS = dict(threshold=0, below=-1, above=1)

    def forward(self, state):
        operand = self.get_operand(0, state)
        threshold = self.get_hyperparam("threshold")
        above = self.get_hyperparam("above")
        below = self.get_hyperparam("below")
        return torch.where(operand > threshold, above, below)

@decorator_common
class Compare(BaseOperation):
    N_OPERANDS = 4

    def forward(self, state):
        mask = self.get_operand(0, state) > self.get_operand(1, state)
        true = self.get_operand(2, state)
        false = self.get_operand(3, state)
        return torch.where(mask, true, false)


@decorator_common
class GreaterThan(BaseOperation):
    N_OPERANDS = 2
    ALLOW_FIRST = False

    def forward(self, state):
        operand1 = self.get_operand(0, state)
        operand2 = self.get_operand(1, state)
        return (operand1 > operand2).to(operand1)

    def __repr__(self):
        return f'({self.operands[0]} > {self.operands[1]})'

@decorator_common
class SignConsistency(BaseOperation):
    N_OPERANDS = 2
    ALLOW_FIRST = False

    def forward(self, state):
        operand1 = self.get_operand(0, state)
        operand2 = self.get_operand(1, state)
        return operand1.sign() * operand2.sign()



