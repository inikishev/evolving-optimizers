import torch
from .._bases import BaseOperation, _format_hyperparam
from .._hyperparameter import Continuous, Log, Choice
from .pools import decorator_common

@decorator_common
class AddScalar(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(x=Continuous.partial(sigma=10))
    DEFAULTS = dict(x=0)

    def forward(self, state):
        return self.get_operand(0, state) + self.get_hyperparam("x")

    def __repr__(self):
        x = _format_hyperparam(self.get_hyperparam('x'))
        return f"({self.operands[0]} + {x})"

@decorator_common
class MulScalar(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(x=Continuous.partial(sigma=10))
    DEFAULTS = dict(x=1)

    def forward(self, state):
        return self.get_operand(0, state) * self.get_hyperparam("x")

    def __repr__(self):
        x = _format_hyperparam(self.get_hyperparam('x'))
        return f"({self.operands[0]} * {x})"

@decorator_common
class DivScalar(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(x=Continuous.partial(sigma=10))
    DEFAULTS = dict(x=1)

    def forward(self, state):
        x = self.get_hyperparam("x")
        if x < torch.finfo(x.dtype).tiny * 2:
            x = torch.finfo(x.dtype).tiny * x.sign()

        return self.get_operand(0, state) / x

    def __repr__(self):
        x = self.get_hyperparam('x')
        if x < torch.finfo(x.dtype).tiny * 2:
            x = torch.finfo(x.dtype).tiny * x.sign()

        return f"({self.operands[0]} / {_format_hyperparam(x)})"



@decorator_common
class EpsilonRDivScalar(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(x=Continuous.partial(sigma=10), eps=Log.partial(sigma=8), clip=Choice.partial(True, False))
    DEFAULTS = dict(x=1, eps=1e-7, clip=False)

    def forward(self, state):
        num = self.get_hyperparam("x")
        eps = self.get_hyperparam("eps")
        clip = self.get_hyperparam("clip")
        denom = self.get_operand(0, state)

        if clip:
            eps_denom = denom.abs().clip(min=eps).copysign(denom)
        else:
            eps_denom = (denom.abs() + eps).copysign(denom)

        return num / eps_denom

    def __repr__(self):
        eps = _format_hyperparam(self.get_hyperparam('eps'))
        clip = self.get_hyperparam("clip")

        num = _format_hyperparam(self.get_hyperparam("x"))
        denom = f'{self.operands[0]}'

        return f"({num} / eps({denom}, {eps}, clip={clip}))"


@decorator_common
class AbsPowScalar(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(x=Continuous.partial(sigma=2))
    DEFAULTS = dict(x=1)

    def forward(self, state):
        return self.get_operand(0, state).abs() ** self.get_hyperparam("x")

    def __repr__(self):
        x = _format_hyperparam(self.get_hyperparam('x'))
        return f"(abs({self.operands[0]})^{x})"

@decorator_common
class SignedPowScalar(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(x=Continuous.partial(sigma=2))
    DEFAULTS = dict(x=1)

    def forward(self, state):
        operand = self.get_operand(0, state)
        return (operand.abs() ** self.get_hyperparam("x")).copysign(operand)


@decorator_common
class ExpScalar(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(x=Continuous.partial(min=0, sigma=2))
    DEFAULTS = dict(x=1)

    def forward(self, state):
        return self.get_hyperparam("x") ** self.get_operand(0, state)

    def __repr__(self):
        x = _format_hyperparam(self.get_hyperparam('x'))
        return f"({x}^{self.operands[0]})"

@decorator_common
class MaxScalar(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(x=Continuous.partial(sigma=10))
    DEFAULTS = dict(x=1)

    def forward(self, state):
        return self.get_operand(0, state).clip(min = self.get_hyperparam("x"))

    def __repr__(self):
        x = _format_hyperparam(self.get_hyperparam('x'))
        return f"max({self.operands[0]}, {x})"

@decorator_common
class MinScalar(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(x=Continuous.partial(sigma=10))
    DEFAULTS = dict(x=1)

    def forward(self, state):
        return self.get_operand(0, state).clip(max = self.get_hyperparam("x"))

    def __repr__(self):
        x = _format_hyperparam(self.get_hyperparam('x'))
        return f"min({self.operands[0]}, {x})"


@decorator_common
class ClipValue(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(x=Log.partial(sigma=4))
    DEFAULTS = dict(x=1)

    def forward(self, state):
        x = self.get_hyperparam("x")
        return self.get_operand(0, state).clip(min = -x, max = x)

    def __repr__(self):
        x = _format_hyperparam(self.get_hyperparam('x'))
        return f"clip({self.operands[0]}, -{x}, {x})"


@decorator_common
class ModuloScalar(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(x=Continuous.partial(sigma=10))
    DEFAULTS = dict(x=1)

    def forward(self, state):
        x = self.get_hyperparam("x")
        if x < torch.finfo(x.dtype).tiny * 2:
            x = torch.finfo(x.dtype).tiny * x.sign()

        return self.get_operand(0, state) % x

    def __repr__(self):
        x = self.get_hyperparam('x')
        if x < torch.finfo(x.dtype).tiny * 2:
            x = torch.finfo(x.dtype).tiny * x.sign()

        return f"({self.operands[0]} % {_format_hyperparam(x)})"



@decorator_common
class EpsilonRModuloScalar(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(x=Continuous.partial(sigma=10), eps=Log.partial(sigma=8), clip=Choice.partial(True, False))
    DEFAULTS = dict(x=1, eps=1e-7, clip=False)

    def forward(self, state):
        num = self.get_hyperparam("x")
        eps = self.get_hyperparam("eps")
        clip = self.get_hyperparam("clip")
        denom = self.get_operand(0, state)

        if clip:
            eps_denom = denom.abs().clip(min=eps).copysign(denom)
        else:
            eps_denom = (denom.abs() + eps).copysign(denom)

        return num % eps_denom

    def __repr__(self):
        eps = _format_hyperparam(self.get_hyperparam('eps'))
        clip = self.get_hyperparam("clip")

        num = _format_hyperparam(self.get_hyperparam("x"))
        denom = f'{self.operands[0]}'

        return f"({num} % eps({denom}, {eps}, clip={clip}))"

