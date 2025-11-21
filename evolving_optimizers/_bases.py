import random
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import torch

from ._utils import clone, to_


class BaseHyperparameter(ABC):
    """hyperparameter, continuous ones are stored in tensors so as to allow differentiability"""
    @abstractmethod
    def get_value(self) -> Any:
        """"""

    @abstractmethod
    def mutate_point_(self) -> None:
        """"""

    @abstractmethod
    def mutate_perturb_(self, sigma:float) -> None:
        """"""

    def to_(self, device: torch.types.Device = None, dtype: torch.dtype | None = None):
        to_(self, device=device, dtype=dtype)

    @classmethod
    def partial(cls, *args, **kwargs):
        return lambda x: cls(x, *args, **kwargs) # pyright:ignore[reportCallIssue]

    def clone(self):
        return clone(self)




def _init_hyperparam(cls: Callable[[Any], BaseHyperparameter], value: Any):
    if isinstance(value, BaseHyperparameter): return value
    return cls(value)

def _format_hyperparam(x: Any):
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        v = x.item()
        if v < 1:
            return f"{x.item():.3g}" # pylint:disable=consider-using-f-string
        if isinstance(v, int):
            return v
        return f"{v:.3f}"
    return x

class BaseOperation(ABC):
    """Operation in the algorithm"""

    WEIGHT: float = 1.0
    """weight for choosing this operation from the pool"""

    N_OPERANDS: int = 0
    """number of operands (any other operations) passed to this operation"""

    HYPERPARAMS: dict[str, Callable[[Any], BaseHyperparameter]] = {}
    """Maps string keys to callables which take initial value (like one from ``DEFAULTS``)
    and return a hyperparameter object"""

    DEFAULTS: dict[str, Any] = {}
    """Maps hyperparameter keys to default values"""

    ALLOW_FIRST: bool = True

    def __init__(self, *operands: "BaseOperation", **hyperparams: Any):
        """hyperparams can either be values or hyperparameter objects"""

        if len(operands) != self.N_OPERANDS:
            raise RuntimeError(
                f"{self.__class__.__name__} received {len(operands)} operands, while it has {self.N_OPERANDS = }")

        if not (tuple(hyperparams.keys()) == tuple(self.HYPERPARAMS.keys()) == tuple(self.DEFAULTS.keys())):
            raise RuntimeError(
                f"{self.__class__.__name__} received unmatching hyperparameter keys:\n"
                f"{tuple(hyperparams.keys()) = }\n"
                f"{tuple(self.HYPERPARAMS.keys()) = }\n"
                f"{tuple(self.DEFAULTS.keys()) = }"
                )


        self.operands: list[BaseOperation] = list(operands)
        self.hyperparams: dict[str, BaseHyperparameter] = {k: _init_hyperparam(self.HYPERPARAMS[k], v) for k,v in hyperparams.items()}

    def initialize(self, ref: torch.Tensor) -> None:
        """initializes state such as buffers etc, this will be called multiple times to reset and reinitialize.
        ``ref`` is a reference tensor of the same shape as all operands"""

    def prepare_(self, ref:torch.Tensor):
        for b in self.flat_branches():
            b.initialize(ref)

        self.to_(ref.device)


    @abstractmethod
    def forward(self, state: dict[str, torch.Tensor | Any]) -> torch.Tensor:
        """what this operation does"""

    def get_operand(self, idx: int, state: dict[str, torch.Tensor | Any]):
        """returns output of specified operand"""
        return self.operands[idx].forward(state)

    def get_hyperparam(self, key: str):
        """returns value of specified hyperparameter"""
        return self.hyperparams[key].get_value()

    def flat_branches(self, include_self:bool=True):
        """flat operands"""
        flat: list[BaseOperation] = []

        if include_self:
            flat.append(self)

        for op in self.operands:
            flat.extend(op.flat_branches())

        return flat

    def flat_idxs(self, include_self:bool=True) -> list[list[int]]:
        """returns indexes of each operation in the entire tree"""
        tree_idxs = []

        if include_self:
            tree_idxs.insert(0, [])

        for i, op in enumerate(self.operands):
            for idxs in op.flat_idxs(include_self=True):
                tree_idxs.append([i, *idxs])

        return tree_idxs

    # def flat_idxs(self, include_self:bool=True) -> list[list[int]]:
    #     """returns indexes of each operation in the entire tree"""
    #     tree_idxs = [[i] for i in range(self.N_OPERANDS)]

    #     if include_self:
    #         tree_idxs.insert(0, [])

    #     for i, op in enumerate(self.operands):
    #         for idxs in op.flat_idxs(include_self=False):
    #             tree_idxs.append([i, *idxs])

    #     return tree_idxs

    def depth(self) -> int:
        return max(len(i) for i in self.flat_idxs())

    def total_operands(self) -> int:
        return len(self.flat_branches(include_self=False))

    def select_branch_by_idx(self, idx: list[int]):
        """select operation by its index"""
        c = self
        for v in idx:
            c = c.operands[v]
        return c

    def replace_branch_by_idx_(self, idx: list[int], new: "BaseOperation"):
        """replace operation under indexes with another one (in-place)"""
        if len(idx) == 0:
            raise RuntimeError("Idxs is an empty list, can't replace self!")

        c = self
        for i, v in enumerate(idx):
            if i == (len(idx) - 1):
                c.operands[v] = new
                return

            c = c.operands[v]

        raise RuntimeError("can't happen")

    def pick_via_random_walk(self, include_self: bool = True) -> "tuple[list[BaseOperation], list[list[int]]]":
        """random walk on the tree until it hits a leaf, returns ``(branches, idxs)``"""
        if include_self:
            branches: list[BaseOperation] = [self]
            idxs: list[list[int]] = [[]]
        else:
            branches = []
            idxs = []

        current = self
        path = []

        while True:
            if current.N_OPERANDS == 0:
                return branches, idxs

            idx = random.randrange(0, current.N_OPERANDS)
            branches.append(current.operands[idx])
            path.append(idx)
            idxs.append(path.copy())

            current = current.operands[idx]


    def pick_random_branch(
        self,
        target_rdepth: float | None,
        weight_fn: "Callable[[BaseOperation, list[int]], float] | None" = None,
        self_weight: float | None = None,
        unbiased: bool = True,
        eps: float = 0.01,
        exp: float = 2,
        shift: float = 0.0,
    ) -> "tuple[BaseOperation, list[int]]":
        """Select a random branch weighted by target relative depth. Returns ``(branch, idx)``.

        Args:
            target_rdepth (float): target depth from 0 to 1.
            weight_fn (Callable[[BaseOperation, list[int]], float] | None, optional):
                optional function which accepts ``(operation, idx)`` and returns
                multiplier to weight of the operation. Defaults to None.
            self_weight (bool, optional):
                if specified, can pick self with weight multiplied by this value. Defaults to None
            unbiased (bool, optional):
                removes bias where some branches have so many operands that
                they are significantly more likely to be picked. Defaults to 0.
            eps (float, optional): lowest possible weight. Defaults to 0.01.
            exp (float, optional): exponential applied to probabilities to make them stronger. Defaults to 2
            shift (float, optional): shifts probabilities. Defaults to 0.
        """
        include_self = self_weight is not None
        if unbiased:
            branches, idxs = self.pick_via_random_walk(include_self=include_self)

        else:
            branches = self.flat_branches(include_self=include_self)
            idxs = self.flat_idxs(include_self=include_self)
            assert len(branches) == len(idxs), f"{branches = }, {idxs = }"

        if target_rdepth is None:
            weights = [1. for _ in branches]

        else:
            dists_to_root = [len(i) for i in idxs]
            dists_to_leaf = [b.depth() for b in branches]

            # relative depths in [0, 1] range
            rdepths = [r / max(r + l, 1) for r, l in zip(dists_to_root, dists_to_leaf)]

            # weigths are distances from depths to targets
            weights = [(1 - abs(target_rdepth - rd)) ** exp for rd in rdepths]
            weights = [max(w + shift, eps) for w in weights]

        if self_weight is not None:
            weights[0] = weights[0] * self_weight

        # apply weight fn
        if weight_fn is not None:
            weights = [w * weight_fn(branch, idx) for w, branch, idx in zip(weights, branches, idxs)]

        # select branch by weights
        if sum(weights) == 0:
            weights = None

        # if weights is not None:
        #     for w, b in zip(weights, branches):
        #         print(f'{w = }, {b = }')

        i_sel = random.choices(list(range(len(branches))), weights, k=1)[0]
        return branches[i_sel], idxs[i_sel]


    def to_(self, device: torch.types.Device = None, dtype: torch.dtype | None = None):
        """move to device and dtype in-place"""
        to_(self, device=device, dtype=dtype)

        for op in self.operands:
            op.to_(device=device, dtype=dtype)

        for hp in self.hyperparams.values():
            hp.to_(device=device, dtype=dtype)

    def clone(self):
        """clone"""
        return type(self)(
            *[op.clone() for op in self.operands],
            **{k: v.clone() for k,v in self.hyperparams.items()}
        )

    def _tostring(self, fn=None):
        if fn is None: fn = lambda x: x
        s = f"{self.__class__.__name__}"

        if len(self.operands) > 0:
            operands_s = ', '.join([f"{fn(o)}" for o in self.operands])
            s = f'{s}({operands_s})'

        if len(self.hyperparams) > 0:
            hyperparams_s = ""
            for k in self.hyperparams:
                hyperparams_s = f'{hyperparams_s}{k}={_format_hyperparam(self.get_hyperparam(k))}, '

            if len(self.operands) == 0: s = f'{s}({hyperparams_s[:-2]})'
            else: s = f'{s[:-1]}, {hyperparams_s[:-2]})'

        return s

    def __repr__(self):
        return self._tostring()

    def string(self):
        if len(self.operands) == 0: return self._tostring()
        return self._tostring(lambda x: x.string())

class BaseCrossover(ABC):
    """combine two trees into children"""
    @abstractmethod
    def cross(self, pool: "BasePool", tree1: BaseOperation, tree2: BaseOperation) -> tuple[BaseOperation, ...]:
        """Crossover given roots and return tuple of children. This must clone roots!

        Args:
            pool (BaseOperationPool): _description_
            root1 (BaseOperation): _description_
            root2 (BaseOperation): _description_

        Returns:
            tuple[BaseOperation, ...]: _description_
        """


class BasePool(ABC):
    """randomly selects operations from the pool
    Args:
        must_contain (Sequence[type[BaseOperation]]): if cur=0, tree will require to contain any of specified operands.
    """
    def __init__(self, must_contain: Sequence[type[BaseOperation]]):
        self.must_contain = must_contain

    @abstractmethod
    def select(self, cur: int, weight_fn: Callable[[type[BaseOperation]], float] = lambda x: 1) -> type[BaseOperation]:
        """selects a random operation class, cur is number of operations already in the tree"""

    def random_tree(self, cur: int = 0, must_contain: Sequence[type[BaseOperation]] = (), weight_fn: Callable[[type[BaseOperation]], float] = lambda x: 1) -> BaseOperation:

        if cur == 0:
            must_contain = list(must_contain) + list(self.must_contain)

        tree, cur = random_tree(self, cur=cur, must_contain=must_contain, weight_fn=weight_fn)
        return tree

def _init_random_hyperparams(operation: type[BaseOperation]):
    hyperparams = {}
    for k, v in operation.HYPERPARAMS.items():
        if k not in operation.DEFAULTS:
            raise KeyError(f'{operation} doesn\'t have key "{k}" in DEFAULTS')
        hyperparam = v(operation.DEFAULTS[k])
        hyperparam.mutate_perturb_(random.triangular(0, 1, 0))
        hyperparams[k] = hyperparam
    return hyperparams


def random_tree(pool: BasePool, cur: int = 0, must_contain: Sequence[type[BaseOperation]] = (), weight_fn: Callable[[type[BaseOperation]], float] = lambda x: 1, *, _n_attempts=0) -> tuple[BaseOperation, int]:
    """returns ``(tree, cur)``

    Args:
        pool (BasePool): pool of operations
        cur (int, optional): number of operations that have already been selected to penalize large trees. Defaults to 0.
        must_contain (Sequence[type[BaseOperation]], optional):
            will generate until tree contains at least one of any of the specified operations. Defaults to ().
        weight_fn (_type_, optional):
            function that accepts type[BaseOperation] and returns multiplier to probability of picking that operation.
            Defaults to ``lambda x: 1``.
        _n_attempts (int, optional):
            number of attempts generating a tree that contains ``must_contain`` (to avoid infinite recursion). Defaults to 0.
    """
    init_cur = cur

    root = pool.select(cur, weight_fn=weight_fn)
    cur = cur + 1

    operands: list[BaseOperation] = []
    for _ in range(root.N_OPERANDS):
        operand, cur = random_tree(pool, cur)
        operands.append(operand)

    if len(must_contain) > 0:
        ops = [type(t) for op in operands for t in op.flat_branches()] + [root]
        if not any(mc in ops for mc in must_contain):
            if _n_attempts > 1000:
                warnings.warn(f"failed to generate tree containing {must_contain}")
                return random_tree(pool=pool, cur=init_cur, weight_fn=weight_fn)

            return random_tree(pool=pool, cur=init_cur, must_contain=must_contain, weight_fn=weight_fn, _n_attempts=_n_attempts+1)

    return root(*operands, **_init_random_hyperparams(root)), cur

class BaseMutation(ABC):
    """mutates a tree"""
    @abstractmethod
    def mutate(self, pool: BasePool, tree: BaseOperation, sigma: float) -> BaseOperation:
        """mutate ``root``, this must clone it first!"""

