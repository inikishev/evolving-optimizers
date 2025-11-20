import torch

from .._bases import BaseOperation, _format_hyperparam
from .._hyperparameter import Choice, Continuous, Log
from .pools import decorator_common

@decorator_common
class EMA(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(beta=Continuous.partial(sigma=2), debias=Choice.partial(True, False))
    DEFAULTS = dict(beta=0.9, debias=True)

    def initialize(self, ref):
        self.exp_avg = torch.zeros_like(ref)
        self.current_step = 1

    def forward(self, state):
        operand = self.get_operand(0, state)
        beta = self.get_hyperparam("beta")

        # update
        self.exp_avg = self.exp_avg.lerp(operand, (1-beta))

        # debias
        exp_avg = self.exp_avg
        if self.get_hyperparam("debias") and beta > torch.finfo(self.exp_avg.dtype).eps:
            exp_avg = exp_avg / (1 - beta ** self.current_step)

        self.current_step += 1
        return exp_avg

@decorator_common
class Momentum(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(momentum=Continuous.partial(sigma=2), nesterov=Choice.partial(True, False))
    DEFAULTS = dict(momentum=0.9, nesterov=False)

    def initialize(self, ref):
        self.velocity = torch.zeros_like(ref)

    def forward(self, state):
        operand = self.get_operand(0, state)
        momentum = self.get_hyperparam("momentum")
        nesterov = self.get_hyperparam("nesterov")

        self.velocity = self.velocity.mul(momentum).add(operand)
        if nesterov: return self.velocity + operand
        return self.velocity


@decorator_common
class AccumulateMaximum(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(decay=Continuous)
    DEFAULTS = dict(decay=0)

    def initialize(self, ref):
        self.accumulator = torch.zeros_like(ref)

    def forward(self, state):
        operand = self.get_operand(0, state)
        decay = self.get_hyperparam("decay")

        self.accumulator = self.accumulator.mul((1-decay)).maximum(operand)
        return self.accumulator

@decorator_common
class AccumulateMinimum(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(decay=Continuous)
    DEFAULTS = dict(decay=0)

    def initialize(self, ref):
        self.accumulator = torch.zeros_like(ref)

    def forward(self, state):
        operand = self.get_operand(0, state)
        decay = self.get_hyperparam("decay")

        self.accumulator = self.accumulator.mul((1-decay)).minimum(operand)
        return self.accumulator


@decorator_common
class AccumulateSum(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(decay=Continuous)
    DEFAULTS = dict(decay=0)

    def initialize(self, ref):
        self.accumulator = torch.zeros_like(ref)

    def forward(self, state):
        operand = self.get_operand(0, state)
        decay = self.get_hyperparam("decay")

        self.accumulator = self.accumulator.mul((1-decay)).add(operand)
        return self.accumulator

@decorator_common
class AccumulateProduct(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(decay=Continuous)
    DEFAULTS = dict(decay=0)

    def initialize(self, ref):
        self.accumulator = torch.ones_like(ref)

    def forward(self, state):
        operand = self.get_operand(0, state)
        decay = self.get_hyperparam("decay")

        self.accumulator = self.accumulator.mul((1-decay)).mul(operand)
        return self.accumulator

@decorator_common
class NormalizedMomentum(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(momentum=Continuous.partial(sigma=2), p=Continuous.partial(sigma=4), lerp=Choice.partial(True, False))
    DEFAULTS = dict(momentum=0.9, p=2, lerp=True)

    def initialize(self, ref):
        self.velocity: torch.Tensor = torch.zeros_like(ref)

    def forward(self, state):
        operand = self.get_operand(0, state)
        momentum = self.get_hyperparam("momentum")
        lerp = self.get_hyperparam("lerp")
        p = self.get_hyperparam("p")

        eps = torch.finfo(operand.dtype).tiny * 2
        vel_norm = torch.linalg.vector_norm(self.velocity, ord=p, dim=-1, keepdim=True) # pylint:disable=not-callable
        self.velocity = self.velocity / vel_norm.clip(min=eps)

        op_norm = torch.linalg.vector_norm(operand, ord=p, dim=-1, keepdim=True) # pylint:disable=not-callable
        operand = operand / op_norm.clip(min=eps)

        if lerp:
            self.velocity = self.velocity.lerp(operand, weight=1-momentum)
        else:
            self.velocity = self.velocity.add(operand).mul(momentum)

        return self.velocity