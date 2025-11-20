"""pool is a set of operations, the only difference between pools is base tensors like grad for optimizer or Ax for solver"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._bases import BaseOperation

COMMON_POOL = []

def decorator_common(x: "type[BaseOperation]"):
    assert x not in COMMON_POOL
    COMMON_POOL.append(x)
    return x
