import torch

from .._bases import BaseOperation
from .._operations import *

class B(BaseOperation):
    def forward(self, state):
        return state["b"]
    def __repr__(self): return "b"

class X(BaseOperation):
    def forward(self, state):
        return state["x"]
    def __repr__(self): return "x"

class Ax(BaseOperation):
    WEIGHT=2 # make it more likely to be picked because its an important one
    def forward(self, state):
        if "Ax" not in state:
            state["Ax"] = (state["A"] @ state["x"].unsqueeze(-1)).squeeze(-1)
            state["n_matvecs"] += 1
        return state["Ax"]
    def __repr__(self): return "Ax"

class Residual(BaseOperation):
    WEIGHT=2
    def forward(self, state):
        if "Ax" not in state:
            state["Ax"] = (state["A"] @ state["x"].unsqueeze(-1)).squeeze(-1)
            state["n_matvecs"] += 1
        return state["Ax"] - state["b"]
    def __repr__(self): return "r"

class Matvec(BaseOperation):
    N_OPERANDS=1
    def forward(self, state):
        state["n_matvecs"] += 1
        v = self.get_operand(0, state)
        return (state["A"] @ v.unsqueeze(-1)).squeeze(-1)
    def __repr__(self): return f"(A{self.operands[0]})"

class MatvecResidual(BaseOperation):
    N_OPERANDS=1
    def forward(self, state):
        state["n_matvecs"] += 1
        v = self.get_operand(0, state)
        return (state["A"] @ v.unsqueeze(-1)).squeeze(-1) - state["b"]
    def __repr__(self): return f"(A{self.operands[0]} - b)"

def richardson(): # for run_tree
    return Sub(X(), MulScalar(Residual(), x=5e-2))

SOLVER_POOL = (X, Ax, B, Matvec, Residual, MatvecResidual)
SOLVER_MUST_CONTAIN_ANY_OF = (Ax, Matvec, Residual, MatvecResidual)

def run_tree(tree: BaseOperation, A: torch.Tensor, b: torch.Tensor, matvecs:int):
    """A is ``(nsamples, m, n)``; b is ``(nsamples, n)``.

    Runs ``tree`` where x_{k+1} = tree(x_k)."""
    tree.prepare_(b)

    x = torch.zeros_like(b)

    # state contains all base tensors
    state = {"A": A, "x": x, "b": b, "n_matvecs": 0}

    while state["n_matvecs"] < matvecs:
        state["x"] = tree.forward(state)
        state.pop("Ax", None)

        if state["n_matvecs"] == 0:
            break # means tree has no matvecs so it is useless

    return (A @ state["x"].unsqueeze(-1)).squeeze(-1)

def run_tree_additive(tree: BaseOperation, A: torch.Tensor, b: torch.Tensor, matvecs:int):
    """A is ``(nsamples, m, n)``; b is ``(nsamples, n)``.

    Runs ``tree`` where x_{k+1} = x_k + tree(x_k)."""
    tree.prepare_(b)
    x = torch.zeros_like(b)

    # state contains all base tensors
    state = {"A": A, "x": x, "b": b, "n_matvecs": 0}

    while state["n_matvecs"] < matvecs:
        state["x"] = state["x"] + tree.forward(state)
        state.pop("Ax", None)

        if state["n_matvecs"] == 0:
            break # means tree has no matvecs so it is useless

    return (A @ state["x"].unsqueeze(-1)).squeeze(-1)


def run_tree_stepsize(tree: BaseOperation, A: torch.Tensor, b: torch.Tensor, matvecs:int):
    """A is ``(nsamples, m, n)``; b is ``(nsamples, n)``.

    Runs ``tree`` where x_{k+1} = x_k - r * tree(x_k)."""
    tree.prepare_(b)
    x = torch.zeros_like(b)

    # state contains all base tensors
    state = {"A": A, "x": x, "b": b, "n_matvecs": 0}

    while state["n_matvecs"] < matvecs:
        state["Ax"] = (state["A"] @ state["x"].unsqueeze(-1)).squeeze(-1)
        r = state["Ax"] - state["b"]
        state["x"] = state["x"] - r  * tree.forward(state)
        state.pop("Ax", None)

        if state["n_matvecs"] == 0:
            break # means tree has no matvecs so it is useless

    return (A @ state["x"].unsqueeze(-1)).squeeze(-1)
