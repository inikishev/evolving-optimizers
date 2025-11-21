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
        return self.get_operand(0, state).neg()

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

@decorator_common
class Tanh(BaseOperation):
    N_OPERANDS = 1

    def forward(self, state):
        return self.get_operand(0, state).tanh()


@decorator_common
class Sin(BaseOperation):
    N_OPERANDS = 1

    def forward(self, state):
        return self.get_operand(0, state).sin()

@decorator_common
class Cos(BaseOperation):
    N_OPERANDS = 1

    def forward(self, state):
        return self.get_operand(0, state).cos()

@decorator_common
class Sinc(BaseOperation):
    N_OPERANDS = 1

    def forward(self, state):
        return self.get_operand(0, state).sinc()

@decorator_common
class Arctan(BaseOperation):
    N_OPERANDS = 1

    def forward(self, state):
        return self.get_operand(0, state).arctan()

@decorator_common
class Arcsinh(BaseOperation):
    N_OPERANDS = 1

    def forward(self, state):
        return self.get_operand(0, state).arcsinh()

@decorator_common
class Softplus(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False
    HYPERPARAMS = dict(beta=Continuous.partial(sigma=10), threshold=Continuous.partial(sigma=50))
    DEFAULTS = dict(beta=1, threshold=20)

    def forward(self, state):
        beta = self.get_hyperparam("beta")
        threshold = self.get_hyperparam("threshold")
        return F.softplus(self.get_operand(0, state), beta, threshold) # pylint:disable=not-callable

@decorator_common
class Softmax(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False

    def forward(self, state):
        return self.get_operand(0, state).softmax(-1)

@decorator_common
class Softmin(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False

    def forward(self, state):
        return F.softmin(self.get_operand(0, state), -1)

@decorator_common
class Softsign(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False

    def forward(self, state):
        return F.softsign(self.get_operand(0, state))

@decorator_common
class LogSigmoid(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False

    def forward(self, state):
        return F.logsigmoid(self.get_operand(0, state)) # pylint:disable=not-callable

@decorator_common
class NormalizeMAD(BaseOperation):
    N_OPERANDS = 1

    def forward(self, state):
        operand = self.get_operand(0, state)
        mad = operand.abs().mean(dim=-1, keepdim=True)
        return operand / mad.clip(min=torch.finfo(mad.dtype).tiny * 2)

@decorator_common
class NormalizeLInf(BaseOperation):
    N_OPERANDS = 1

    def forward(self, state):
        operand = self.get_operand(0, state)
        linf = operand.amax(dim=-1, keepdim=True)
        return operand / linf.clip(min=torch.finfo(linf.dtype).tiny * 2)

@decorator_common
class SubFromMax(BaseOperation):
    N_OPERANDS = 1

    def forward(self, state):
        operand = self.get_operand(0, state)
        return operand.amax(dim=-1, keepdim=True) - operand

@decorator_common
class Dropout(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False
    HYPERPARAMS = dict(p=Continuous.partial(min=0, max=1))
    DEFAULTS = dict(p=1)

    def forward(self, state):
        operand = self.get_operand(0, state)
        p = self.get_hyperparam("p")
        mask = torch.bernoulli(torch.full_like(operand, p))
        return operand * mask

