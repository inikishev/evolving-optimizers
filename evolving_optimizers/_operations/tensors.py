import torch
from .._bases import BaseOperation, _format_hyperparam
from .._hyperparameter import Continuous, Choice
from .pools import decorator_common

# we shall have "ref" in all states which is any tensor of correct shape device etc

@decorator_common
class Random(BaseOperation):
    ALLOW_FIRST = False
    HYPERPARAMS = dict(distribution = Choice.partial("normal", "rademacher", "uniform"))
    DEFAULTS = dict(distribution = "normal")
    def forward(self, state):
        distribution = self.get_hyperparam("distribution")

        if distribution == "normal":
            return torch.randn_like(state["ref"])

        if distribution == "rademacher":
            return torch.randint_like(state["ref"], 0, 2) * 2 - 1

        if distribution == "uniform":
            with torch.no_grad():
                return torch.empty_like(state["ref"]).uniform_(-1,1)

        raise ValueError(distribution)

    def __repr__(self): return self.get_hyperparam("distribution")

@decorator_common
class Full(BaseOperation):
    HYPERPARAMS = dict(x=Continuous.partial(sigma=10))
    DEFAULTS = dict(x=1)
    ALLOW_FIRST = False

    def forward(self, state):
        x = self.get_hyperparam("x")
        return torch.full_like(state["ref"], fill_value=x)

    def __repr__(self):
        x = _format_hyperparam(self.get_hyperparam("x"))
        return f'{x}'

