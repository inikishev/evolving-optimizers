import torch

from .._bases import BaseOperation
from .._operations import *
from .._utils import vec_to_tensors, vec_to_tensors_

def flatten_jacobian(jacs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    n_out = jacs[0].shape[0]
    return torch.cat([j.reshape(n_out, -1) for j in jacs], dim=1)

class TreeOpt(torch.optim.Optimizer):
    def __init__(self, params, lr: float, tree: BaseOperation):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.tree = tree

        params = [p for g in self.param_groups for p in g["params"]]
        ref = torch.cat([p.ravel() for p in params])

        self.tree.prepare_(ref)


    @torch.no_grad
    def step(self, closure): # type:ignore pylint:disable=signature-differs
        """returns ``(loss, n_forwards, n_backwards, n_hvps)``"""
        self.n_forwards = 0
        self.n_backwards = 0
        self.n_hvps = 0

        params = []
        lrs = []
        for g in self.param_groups:
            for p in g["params"]:
                if p.requires_grad:
                    params.append(p)
                    lrs.append(g["lr"])

        init_params = [p.clone() for p in params]

        def gather_grads():
            return torch.cat([p.grad.ravel() if p.grad is not None else torch.zeros_like(p) for p in params])

        def loss_fn(x):
            self.n_forwards += 1

            vec_to_tensors_(x, params)
            return closure(False)

        def loss_grad_fn(x):
            self.n_forwards += 1
            self.n_backwards += 1

            vec_to_tensors_(x, params)
            with torch.enable_grad():
                loss = closure()

            return loss, gather_grads()

        def hessp_fn(x, z):
            self.n_hvps += 1

            vec_to_tensors_(x, params)
            with torch.enable_grad():
                self.zero_grad()
                loss = closure(False).sum()
                grads = torch.autograd.grad(loss, params, materialize_grads=True, allow_unused=True, create_graph=True)
                Hz = torch.autograd.grad(grads, params, vec_to_tensors(z, params), materialize_grads=True, allow_unused=True)
                return torch.cat([t.ravel() for t in Hz])

        p = torch.cat([t.ravel() for t in params])
        state = {"loss_fn": loss_fn, "loss_grad_fn": loss_grad_fn, "hessp_fn": hessp_fn, "p": p, "ref": p}

        dir = self.tree.forward(state)
        dir_list = vec_to_tensors(dir, params)

        for param, init_param in zip(params, init_params):
            param.copy_(init_param)

        torch._foreach_mul_(dir_list, lrs)
        torch._foreach_sub_(params, dir_list)

        return state.get("f", None), self.n_forwards, self.n_backwards, self.n_hvps




class Grad(BaseOperation):
    WEIGHT = 4
    def forward(self, state):
        if "g" not in state:
            state["f"], state["g"] = state["loss_grad_fn"](state["p"]) # pyright:ignore[reportGeneralTypeIssues, reportCallIssue]

        return state["g"]

    def __repr__(self):
        return "g"

def _broadcast_loss(f: torch.Tensor, p: torch.Tensor):
    if f.shape == p.shape: return f

    if f.numel() == 1:
        return torch.ones_like(p) * f.squeeze()

    if f.ndim != 1:
        raise RuntimeError(f"if f is batched it must be a vec, got {f.shape = }")

    batch_size = f.numel()
    p_batched = p.view(batch_size, -1)
    f_batched = f.unsqueeze(-1).broadcast_to(p_batched.shape)
    return f_batched.flatten()

class Loss(BaseOperation):
    ALLOW_FIRST = False
    def forward(self, state):
        if "f" not in state:
            state["f"] = state["loss_fn"](state["p"]) # pyright:ignore[reportGeneralTypeIssues, reportCallIssue]

        return _broadcast_loss(state["f"], state["p"])

    def __repr__(self):
        return "f"

class Params(BaseOperation):
    def forward(self, state):
        return state["p"]

    def __repr__(self):
        return "p"

class Hessp(BaseOperation):
    N_OPERANDS = 1
    def forward(self, state):
        z = self.get_operand(0, state)
        return state["hessp_fn"](state["p"], z) # pyright:ignore[reportCallIssue]

    def __repr__(self):
        return f"H×({self.operands[0]})"

class GradAt(BaseOperation):
    N_OPERANDS = 1
    @torch.no_grad
    def forward(self, state):
        operand = self.get_operand(0, state)
        _, g = state["loss_grad_fn"](operand) # pyright:ignore[reportCallIssue]
        return g

class LossAt(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False
    @torch.no_grad
    def forward(self, state):
        operand = self.get_operand(0, state)
        f = state["loss_fn"](operand) # pyright:ignore[reportGeneralTypeIssues, reportCallIssue]
        return _broadcast_loss(f, state["p"])


class HesspAt(BaseOperation):
    N_OPERANDS = 2
    @torch.no_grad
    def forward(self, state):
        operand = self.get_operand(0, state)
        z = self.get_operand(1, state)
        Hz = state["hessp_fn"](operand, z) # pyright:ignore[reportCallIssue]
        return Hz

class DirectionalGrad(BaseOperation):
    N_OPERANDS = 1
    @torch.no_grad
    def forward(self, state):
        operand = self.get_operand(0, state)
        _, g = state["loss_grad_fn"](state["p"] + operand) # pyright:ignore[reportGeneralTypeIssues, reportCallIssue]
        return g

class DirectionalLoss(BaseOperation):
    N_OPERANDS = 1
    ALLOW_FIRST = False
    @torch.no_grad
    def forward(self, state):
        operand = self.get_operand(0, state)
        f = state["loss_fn"](state["p"] + operand) # pyright:ignore[reportGeneralTypeIssues, reportCallIssue]
        return _broadcast_loss(f, state["p"])

class DirectionalHessp(BaseOperation):
    N_OPERANDS = 2
    @torch.no_grad
    def forward(self, state):
        operand = self.get_operand(0, state)
        z = self.get_operand(1, state)
        Hz = state["hessp_fn"](state["p"] + operand, z) # pyright:ignore[reportCallIssue]
        return Hz

OPTIMIZER_POOL = [Grad, Loss, Params, Hessp, LossAt, GradAt, HesspAt, DirectionalGrad, DirectionalLoss, DirectionalHessp]
def gd():
    return Grad()

def normgd():
    return Normalize(Grad(), p=2, clip=False)

def heavyball():
    return Momentum(Grad(), momentum=0.9, nesterov=False)

def adagrad():
    return EpsilonDiv(
        Grad(),
        SqrtAbs(AccumulateSum(Square(Grad()), decay=0)),
        eps=1e-7,
        clip=False,
    )

def rmsprop():
    return EpsilonDiv(
        Grad(),
        SqrtAbs(EMA(Square(Grad()), beta=0.99, debias=True)),
        eps=1e-7,
        clip=False,
    )

def adam():
    return EpsilonDiv(
        EMA(Grad(), beta=0.9, debias=True),
        SqrtAbs(EMA(Square(Grad()), beta=0.99, debias=True)),
        eps=1e-7,
        clip=False,
    )


def amsgrad():
    return EpsilonDiv(
        EMA(Grad(), beta=0.9, debias=True),
        SqrtAbs(AccumulateMaximum(EMA(Square(Grad()), beta=0.99, debias=True), decay=0)),
        eps=1e-7,
        clip=False,
    )

def adamw():
    return Add2(
        adam(),
        MulScalar(Params(), x=1e-3),
    )


def cautious_adam():
    return EpsilonDiv(
        Mul2(
            EMA(Grad(), beta=0.9, debias=True),
            SignConsistency(Grad(), EMA(Grad(), beta=0.9, debias=True))
        ),
        SqrtAbs(EMA(Square(Grad()), beta=0.99, debias=True)),
        eps=1e-7,
        clip=False,
    )

def laprop():
    return EMA(rmsprop(), beta=0.9, debias=True)

def signgd():
    return Sign(Grad())

def signum():
    return Sign(EMA(Grad(), beta=0.9, debias=True))

def adahessian():
    return EpsilonDiv(
        EMA(Grad(), beta=0.9, debias=True),
        SqrtAbs(EMA(Square(Hessp(Random(distribution="rademacher"))), beta=0.99, debias=True)),
        eps=1e-7,
        clip=False,
    )

def sophiag():
    return ClipValue(
        EpsilonDiv(
            EMA(Grad(), beta=0.96, debias=False),
            MaxScalar(EMA(Hessp(Random(distribution="normal")), beta=0.99, debias=False), x=1e-12),
            eps=1e-7,
            clip=False,
        ),
        x=1
    )

# not quite but close
def lion():
    return Sign(
        Lerp(
            EMA(Grad(), beta=0.99, debias=False),
            Grad(),
            weight=0.1
        ),
    )


def mars_adam():
    def mars_g():
        return Normalize(Add2(Grad(), MulScalar(LastDifference(Grad(), n=1), x=0.3)), p=2, clip=True)
    return EpsilonDiv(
        EMA(mars_g(), beta=0.9, debias=True),
        SqrtAbs(EMA(Square(mars_g()), beta=0.99, debias=True)),
        eps=1e-7,
        clip=False,
    )

def bb_long():
    num = SelfDot(LastDifference(Params(), n=1))
    denom = Dot(LastDifference(Params(), n=1), LastDifference(Grad(), n=1))
    return Mul2(Grad(), EpsilonDiv(num, denom, eps=1e-7, clip=False))

def bb_short():
    num = Dot(LastDifference(Params(), n=1), LastDifference(Grad(), n=1))
    denom = SelfDot(LastDifference(Grad(), n=1))
    return Mul2(Grad(), EpsilonDiv(num, denom, eps=1e-7, clip=False))

def polyak():
    return Mul2(Grad(), EpsilonDiv(Loss(), SelfDot(Grad()), eps=1e-7, clip=False))

def init_population(lr=None):
    opts = [
        gd(),
        normgd(),
        heavyball(),
        adagrad(),
        rmsprop(),
        adam(),
        amsgrad(),
        adamw(),
        cautious_adam(),
        laprop(),
        signgd(),
        signum(),
        adahessian(),
        sophiag(),
        lion(),
        mars_adam(),
        bb_long(),
        bb_short(),
        polyak(),
    ]

    if lr is not None:
        opts = [MulScalar(opt, x=lr) for opt in opts]
    return opts