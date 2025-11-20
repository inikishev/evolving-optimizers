import math
import random
from collections.abc import Sequence
from typing import Any

import torch

from ._bases import BaseHyperparameter
from ._utils import totensor


class Continuous(BaseHyperparameter):
    def __init__(self, x: Any, min: float | None = None, max: float | None = None, sigma: float = 1):
        self.x = totensor(x, dtype=torch.float32)
        self.min = min
        self.max = max
        self.sigma = sigma

    def get_value(self):
        return self.x

    def mutate_point_(self):
        if (self.min is None) or (self.max is None):
            self.mutate_perturb_(1)

        else:
            with torch.no_grad():
                self.x.uniform_(self.min, self.max)

    def mutate_perturb_(self, sigma):
        # absolute mutation
        if (self.x == 0).all() or (random.random() > 0.5):
            sigma = sigma * self.sigma
            if (self.min is not None) and (self.max is not None):
                sigma = sigma * (self.max - self.min)

            self.x = self.x + torch.randn_like(self.x) * sigma

        # relative mutation
        # this basically solves the issue of having to tune sigma
        else:
            self.x = self.x * (1 + torch.randn_like(self.x) * sigma)

        # clip
        if (self.min is not None) or (self.max is not None):
            self.x = self.x.clip(min=self.min, max=self.max)

class Log(Continuous):
    def __init__(self, x: Any, min: float | None = None, max: float | None = None, sigma: float = 1, base: float = 10):
        x = math.log(x, base)
        if min is not None: min = math.log(min, base)
        if max is not None: max = math.log(max, base)
        super().__init__(x=x, min=min, max=max, sigma=sigma)
        self.base = base

    def get_value(self):
        return self.base ** self.x


class Choice(BaseHyperparameter):
    def __init__(self, x: Any, *choices: Any):
        self.choices = list(choices)
        self.x = totensor(self.choices.index(x), dtype=torch.uint64)

    def get_value(self):
        return self.choices[self.x]

    def mutate_point_(self):
        self.x = torch.randint_like(self.x, low=0, high=len(self.choices))

    def mutate_perturb_(self, sigma):
        if random.triangular(0, 1, 1) < sigma:
            self.mutate_point_()



class Integer(BaseHyperparameter):
    def __init__(self, x: Any, min: int | None = None, max: int | None = None, sigma: float = 1):
        self.x = totensor(x, dtype=torch.int64)
        self.min = min
        self.max = max
        self.sigma = sigma

    def get_value(self):
        return self.x.round()

    def mutate_point_(self):
        if (self.min is None) or (self.max is None):
            self.mutate_perturb_(1)

        else:
            with torch.no_grad():
                self.x = torch.randint_like(self.x, low=self.min, high=self.max+1)

    def mutate_perturb_(self, sigma):
        sigma = sigma * self.sigma
        if (self.min is not None) and (self.max is not None):
            sigma = sigma * (self.max - self.min)

        if sigma > 1:
            rsigma = round(sigma)

            self.x = self.x + torch.randint_like(self.x, low=-rsigma, high=rsigma+1)

        else:
            if random.random() < sigma:
                self.x = self.x + torch.randint_like(self.x, low=-1, high=2)

        if (self.min is not None) or (self.max is not None):
            self.x = self.x.clip(min=self.min, max=self.max)