import copy
from collections import UserDict, UserList
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


def totensor(x, device=None, dtype=None):
    if device is None and dtype is None:
        if isinstance(x, torch.Tensor): return x
        if isinstance(x, np.ndarray): return torch.from_numpy(x)
        return torch.from_numpy(np.asarray(x))

    if isinstance(x, torch.Tensor): return x.to(device=device, dtype=dtype)
    if isinstance(x, np.ndarray): return torch.as_tensor(x, device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)

# we use similar thing to torch.ParameterList/ModuleList
# except no restrictions on overwriting it etc
class TList(UserList):
    def to(self, device: torch.types.Device = None, dtype: torch.dtype | None = None):
        return type(self)(i.to(device=device, dtype=dtype) if isinstance(i, torch.Tensor) else i for i in self)
    def clone(self):
        return type(self)(i.clone() for i in self)

class TDict(UserDict):
    def to(self, device: torch.types.Device = None, dtype: torch.dtype | None = None):
        return type(self)({k: (v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) else v) for k,v in self.items()})
    def clone(self):
        return type(self)({k: v.clone() for k,v in self.items()})

def to_(x_: Any, device: torch.types.Device = None, dtype: torch.dtype | None = None):
    for k in dir(x_):
        v = getattr(x_, k)
        if isinstance(v, (torch.Tensor, TList, TDict)):
            setattr(x_, k, v.to(device=device, dtype=dtype))

def clone(x: Any):
    """clones tensros and deepcopies rest"""
    c = copy.copy(x)
    for k in dir(c):
        v = getattr(c, k)
        if isinstance(v, (torch.Tensor, TList, TDict)):
            setattr(c, k, v.clone())
        elif hasattr(v, "copy") and callable(getattr(v, "copy")):
            setattr(c, k, getattr(v, "copy")())
    return c


def vec_to_tensors(vec: torch.Tensor, reference: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    tensors = []
    cur = 0
    for r in reference:
        numel = r.numel()
        tensors.append(vec[cur:cur+numel].reshape_as(r))
        cur += numel
    return tensors


def vec_to_tensors_(vec: torch.Tensor, tensors_: Iterable[torch.Tensor]):
    cur = 0
    for t in tensors_:
        numel = t.numel()
        t.set_(vec[cur:cur+numel].view_as(t)) # pyright: ignore[reportArgumentType]
        cur += numel
