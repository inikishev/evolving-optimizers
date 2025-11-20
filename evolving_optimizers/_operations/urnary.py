import torch
from torch.nn import functional as F

from .._bases import BaseOperation, _format_hyperparam
from .._hyperparameter import Choice, Continuous, Log
from .pools import decorator_common


@decorator_common
class Square(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False

    def forward(self, state):
        return self.get_operand(0, state).square()

    def __repr__(self):
        return f"({self.operands[0]}^2)"

@decorator_common
class SqrtAbs(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False

    def forward(self, state):
        return self.get_operand(0, state).abs().sqrt()


@decorator_common
class Neg(BaseOperation):
    N_OPERANDS = 1

    def forward(self, state):
        return self.get_operand(0, state).abs().sqrt()

    def __repr__(self):
        return f"(-{self.operands[0]})"


@decorator_common
class Abs(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False

    def forward(self, state):
        return self.get_operand(0, state).abs()

    def __repr__(self):
        return f"|{self.operands[0]}|"

@decorator_common
class Sign(BaseOperation):
    N_OPERANDS = 1

    def forward(self, state):
        return self.get_operand(0, state).sign()

    def __repr__(self):
        return f"sgn({self.operands[0]})"

@decorator_common
class Normalize(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(p=Continuous.partial(sigma=4), clip=Choice.partial(True, False))
    DEFAULTS = dict(p=2, clip=False)

    def forward(self, state):
        operand = self.get_operand(0, state)
        p = self.get_hyperparam("p")
        clip = self.get_hyperparam("clip")
        norm = torch.linalg.vector_norm(operand, ord=p, dim=-1, keepdim=True) # pylint:disable=not-callable
        if clip: norm = norm.clip(min=1)
        return operand / norm.clip(min=torch.finfo(operand.dtype).eps)


@decorator_common
class Centralize(BaseOperation):
    N_OPERANDS = 1

    def forward(self, state):
        operand = self.get_operand(0, state)
        return operand - operand.mean(dim=-1, keepdim=True)

@decorator_common
class Rescale(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False

    def forward(self, state):
        operand = self.get_operand(0, state)

        operand = operand - operand.amin(dim=-1, keepdim=True)
        maxv = operand.amax(dim=-1, keepdim=True).clip(min=torch.finfo(operand.dtype).tiny * 2)
        operand = operand / maxv
        return operand

@decorator_common
class Norm(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(p=Continuous.partial(sigma=4))
    DEFAULTS = dict(p=2)
    ALLOW_FIRST = False

    def forward(self, state):
        operand = self.get_operand(0, state)
        p = self.get_hyperparam("p")
        norm = torch.linalg.vector_norm(operand, ord=p, dim=-1, keepdim=True) # pylint:disable=not-callable
        return torch.ones_like(operand) * norm

    def __repr__(self):
        p = _format_hyperparam(self.get_hyperparam("p"))
        return f'||{self.operands[0]}||_{p}'

@decorator_common
class Mean(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False

    def forward(self, state):
        operand = self.get_operand(0, state)
        return torch.ones_like(operand) * operand.mean(dim=-1, keepdim=True)

    def __repr__(self):
        return f'mean1({self.operands[0]})'


@decorator_common
class Min(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False

    def forward(self, state):
        operand = self.get_operand(0, state)
        return torch.ones_like(operand) * operand.amin(dim=-1, keepdim=True)

    def __repr__(self):
        return f'min1({self.operands[0]})'

@decorator_common
class Max(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False

    def forward(self, state):
        operand = self.get_operand(0, state)
        return torch.ones_like(operand) * operand.amax(dim=-1, keepdim=True)

    def __repr__(self):
        return f'max1({self.operands[0]})'


@decorator_common
class Sigmoid(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False

    def forward(self, state):
        return self.get_operand(0, state).sigmoid()

    def __repr__(self):
        return f"sigmoid({self.operands[0]})"

@decorator_common
class Tanh(BaseOperation):
    N_OPERANDS = 1

    def forward(self, state):
        return self.get_operand(0, state).tanh()

    def __repr__(self):
        return f"tanh({self.operands[0]})"


@decorator_common
class Sin(BaseOperation):
    N_OPERANDS = 1

    def forward(self, state):
        return self.get_operand(0, state).sin()

    def __repr__(self):
        return f"sin({self.operands[0]})"

@decorator_common
class Softplus(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False

    def forward(self, state):
        return F.softplus(self.get_operand(0, state)) # pylint:disable=not-callable

    def __repr__(self):
        return f"softplus({self.operands[0]})"

@decorator_common
class Softmax(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False

    def forward(self, state):
        return self.get_operand(0, state).softmax(-1)

    def __repr__(self):
        return f"softmax({self.operands[0]})"

