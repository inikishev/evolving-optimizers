import torch
from torch.nn import functional as F

from .._bases import BaseOperation, _format_hyperparam
from .._hyperparameter import Choice, Continuous, Log
from .pools import decorator_common


@decorator_common
class Add2(BaseOperation):
    N_OPERANDS = 2
    def forward(self, state):
        return self.get_operand(0, state) + self.get_operand(1, state)
    def __repr__(self):
        return f"({self.operands[0]} + {self.operands[1]})"

@decorator_common
class Mul2(BaseOperation):
    N_OPERANDS = 2
    def forward(self, state):
        return self.get_operand(0, state) * self.get_operand(1, state)
    def __repr__(self):
        return f"({self.operands[0]} * {self.operands[1]})"

@decorator_common
class Sub(BaseOperation):
    N_OPERANDS = 2
    def forward(self, state):
        return self.get_operand(0, state) - self.get_operand(1, state)
    def __repr__(self):
        return f"({self.operands[0]} - {self.operands[1]})"

@decorator_common
class Lerp(BaseOperation):
    N_OPERANDS = 2
    HYPERPARAMS = dict(weight=Continuous.partial(sigma=2))
    DEFAULTS = dict(weight=0.5)
    def forward(self, state):
        weight = self.get_hyperparam("weight")
        return self.get_operand(0, state).lerp(self.get_operand(1, state), weight=weight)


@decorator_common
class AbsCExp(BaseOperation):
    N_OPERANDS = 2
    def forward(self, state):
        return self.get_operand(0, state).abs() ** self.get_operand(1, state)
    def __repr__(self):
        return f"(abs({self.operands[0]})^{self.operands[1]})"



@decorator_common
class SignedExp(BaseOperation):
    N_OPERANDS = 2
    def forward(self, state):
        op0 = self.get_operand(0, state)
        op1 = self.get_operand(1, state)
        return (op0.abs() ** op1).copysign(op0)


@decorator_common
class EpsilonDiv(BaseOperation):
    N_OPERANDS = 2
    HYPERPARAMS = dict(eps=Log.partial(sigma=8), clip=Choice.partial(True, False))
    DEFAULTS = dict(eps=1e-7, clip=False)

    def forward(self, state):
        num = self.get_operand(0, state)
        denom = self.get_operand(1, state)
        eps = self.get_hyperparam("eps")
        clip = self.get_hyperparam("clip")

        if clip:
            eps_denom = denom.abs().clip(min=eps).copysign(denom)
        else:
            eps_denom = (denom.abs() + eps).copysign(denom)

        return num / eps_denom

    def __repr__(self):
        eps = _format_hyperparam(self.get_hyperparam('eps'))
        clip = self.get_hyperparam("clip")

        num = f'{self.operands[0]}'
        denom = f'{self.operands[1]}'

        return f"({num} / eps({denom}, {eps}, clip={clip}))"


@decorator_common
class Copysign(BaseOperation):
    N_OPERANDS = 2
    def forward(self, state):
        return self.get_operand(0, state).copysign(self.get_operand(1, state))


@decorator_common
class Graft(BaseOperation):
    N_OPERANDS = 2
    HYPERPARAMS = dict(p = Continuous.partial(sigma=4))
    DEFAULTS = dict(p = 2)

    def forward(self, state):
        direction = self.get_operand(0, state)
        magnitude = self.get_operand(1, state)
        p = self.get_hyperparam("p")
        dp = torch.linalg.vector_norm(direction, ord=p, dim=-1, keepdim=True) # pylint:disable=not-callable
        mp = torch.linalg.vector_norm(magnitude, ord=p, dim=-1, keepdim=True) # pylint:disable=not-callable
        scale = mp / dp.clip(torch.finfo(mp.dtype).eps)
        return direction * scale




@decorator_common
class Maximum2(BaseOperation):
    N_OPERANDS = 2
    def forward(self, state):
        return self.get_operand(0, state).maximum(self.get_operand(1, state))
    def __repr__(self):
        return f"max2({self.operands[0]}, {self.operands[1]})"

@decorator_common
class Minimum2(BaseOperation):
    N_OPERANDS = 2
    def forward(self, state):
        return self.get_operand(0, state).minimum(self.get_operand(1, state))
    def __repr__(self):
        return f"min2({self.operands[0]}, {self.operands[1]})"



@decorator_common
class Add3(BaseOperation):
    N_OPERANDS = 3
    WEIGHT = 0.5
    def forward(self, state):
        return self.get_operand(0, state) + self.get_operand(1, state) + self.get_operand(2, state)
    def __repr__(self):
        return f"({self.operands[0]} + {self.operands[1]} + {self.operands[2]})"

@decorator_common
class Mul3(BaseOperation):
    N_OPERANDS = 3
    WEIGHT = 0.5
    def forward(self, state):
        return self.get_operand(0, state) * self.get_operand(1, state) * self.get_operand(2, state)
    def __repr__(self):
        return f"({self.operands[0]} * {self.operands[1]} * {self.operands[2]})"


@decorator_common
class Maximum3(BaseOperation):
    N_OPERANDS = 3
    WEIGHT = 0.5
    def forward(self, state):
        return self.get_operand(0, state).maximum(self.get_operand(1, state)).maximum(self.get_operand(2, state))
    def __repr__(self):
        return f"max3({self.operands[0]}, {self.operands[1]}, {self.operands[2]})"

@decorator_common
class Minimum3(BaseOperation):
    N_OPERANDS = 3
    WEIGHT = 0.5
    def forward(self, state):
        return self.get_operand(0, state).minimum(self.get_operand(1, state)).minimum(self.get_operand(2, state))
    def __repr__(self):
        return f"min3({self.operands[0]}, {self.operands[1]}, {self.operands[2]})"



@decorator_common
class Alternate2(BaseOperation):
    N_OPERANDS = 2

    def initialize(self, ref):
        self.current_step = -1

    def forward(self, state):
        self.current_step += 1

        if self.current_step % 2 == 0:
            return self.get_operand(0, state)

        return self.get_operand(1, state)


@decorator_common
class MaskedMix2(BaseOperation):
    N_OPERANDS = 2
    HYPERPARAMS = dict(p=Continuous.partial(min=0, max=1))
    DEFAULTS = dict(p=0.5)
    def forward(self, state):
        op1 = self.get_operand(0, state)
        op2 = self.get_operand(1, state)
        p = self.get_hyperparam("p")

        mask = torch.bernoulli(torch.full_like(op1, p)).bool()
        return torch.where(mask, op1, op2)


@decorator_common
class EpsilonModulo(BaseOperation):
    N_OPERANDS = 2
    HYPERPARAMS = dict(eps=Log.partial(sigma=8), clip=Choice.partial(True, False))
    DEFAULTS = dict(eps=1e-7, clip=False)

    def forward(self, state):
        num = self.get_operand(0, state)
        denom = self.get_operand(1, state)
        eps = self.get_hyperparam("eps")
        clip = self.get_hyperparam("clip")

        if clip:
            eps_denom = denom.abs().clip(min=eps).copysign(denom)
        else:
            eps_denom = (denom.abs() + eps).copysign(denom)

        return num % eps_denom

    def __repr__(self):
        eps = _format_hyperparam(self.get_hyperparam('eps'))
        clip = self.get_hyperparam("clip")

        num = f'{self.operands[0]}'
        denom = f'{self.operands[1]}'

        return f"({num} % eps({denom}, {eps}, clip={clip}))"


@decorator_common
class CosineSimilarity(BaseOperation):
    N_OPERANDS = 2
    ALLOW_FIRST = False
    def forward(self, state):
        op1 = self.get_operand(0, state)
        op2 = self.get_operand(1, state)
        cossim = F.cosine_similarity(op1, op2, dim=-1) # pylint:disable=not-callable
        if op1.ndim == 2: cossim = cossim.unsqueeze(-1)
        return torch.ones_like(op1) * cossim

@decorator_common
class Dot(BaseOperation):
    N_OPERANDS = 2
    ALLOW_FIRST = False
    def forward(self, state):
        op1 = self.get_operand(0, state)
        op2 = self.get_operand(1, state)
        dot = (op1 * op2).sum(-1, keepdim=True)
        return torch.ones_like(op1) * dot

    def __repr__(self):
        return f"({self.operands[0]}^T {self.operands[1]})"

@decorator_common
class SelfDot(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False
    def forward(self, state):
        op1 = self.get_operand(0, state)
        dot = (op1 * op1).sum(-1, keepdim=True)
        return torch.ones_like(op1) * dot

    def __repr__(self):
        return f"||{self.operands[0]}||^2"



@decorator_common
class Atan2(BaseOperation):
    N_OPERANDS = 2
    def forward(self, state):
        return self.get_operand(0, state).atan2(self.get_operand(1, state))
