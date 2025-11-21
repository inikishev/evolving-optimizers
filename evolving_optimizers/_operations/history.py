import torch
from .._bases import BaseOperation, _format_hyperparam
from .._utils import TList
from .._hyperparameter import Continuous, Log, Choice, Integer
from .pools import decorator_common


@decorator_common
class Previous(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(n=Integer.partial(min=1, max=20))
    DEFAULTS = dict(n=1)

    def initialize(self, ref):
        self.history = TList()

    def forward(self, state):
        operand = self.get_operand(0, state)
        n = self.get_hyperparam("n")

        assert len(self.history) <= n

        if len(self.history) == n:
            value = self.history.pop(0)
            self.history.append(operand)
            return value

        self.history.append(operand)
        return self.history[0]

@decorator_common
class LastDifference(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(n=Integer.partial(min=1, max=20))
    DEFAULTS = dict(n=1)

    def initialize(self, ref):
        self.history = TList()

    def forward(self, state):
        operand = self.get_operand(0, state)
        n = self.get_hyperparam("n")

        assert len(self.history) <= n

        if len(self.history) == n:
            value = self.history.pop(0)
            self.history.append(operand)
            return operand - value

        self.history.append(operand)

        if len(self.history) == 1:
            return operand

        return operand - self.history[0]


@decorator_common
class LastProduct(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(n=Integer.partial(min=1, max=20))
    DEFAULTS = dict(n=1)

    def initialize(self, ref):
        self.history = TList()

    def forward(self, state):
        operand = self.get_operand(0, state)
        n = self.get_hyperparam("n")

        assert len(self.history) <= n

        if len(self.history) == n:
            value = self.history.pop(0)
            self.history.append(operand)
            return operand * value

        self.history.append(operand)

        if len(self.history) == 1:
            return operand.square()

        return operand * self.history[0]


@decorator_common
class HistorySum(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(n=Integer.partial(min=2, max=20))
    DEFAULTS = dict(n=10)

    def initialize(self, ref):
        self.history = TList()

    def forward(self, state):
        operand = self.get_operand(0, state)
        n = self.get_hyperparam("n")

        assert len(self.history) <= n

        if len(self.history) == n:
            self.history.pop(0)

        self.history.append(operand)

        return torch.stack(tuple(self.history), dim=0).sum(dim=0)

@decorator_common
class HistoryProduct(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(n=Integer.partial(min=2, max=20))
    DEFAULTS = dict(n=10)

    def initialize(self, ref):
        self.history = TList()

    def forward(self, state):
        operand = self.get_operand(0, state)
        n = self.get_hyperparam("n")

        assert len(self.history) <= n

        if len(self.history) == n:
            self.history.pop(0)

        self.history.append(operand)

        return torch.stack(tuple(self.history), dim=0).prod(dim=0)

@decorator_common
class HistoryMean(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(n=Integer.partial(min=2, max=20))
    DEFAULTS = dict(n=10)

    def initialize(self, ref):
        self.history = TList()

    def forward(self, state):
        operand = self.get_operand(0, state)
        n = self.get_hyperparam("n")

        assert len(self.history) <= n

        if len(self.history) == n:
            self.history.pop(0)

        self.history.append(operand)

        return torch.stack(tuple(self.history), dim=0).mean(dim=0)

@decorator_common
class HistoryMedian(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(n=Integer.partial(min=2, max=20))
    DEFAULTS = dict(n=10)

    def initialize(self, ref):
        self.history = TList()

    def forward(self, state):
        operand = self.get_operand(0, state)
        n = self.get_hyperparam("n")

        assert len(self.history) <= n

        if len(self.history) == n:
            self.history.pop(0)

        self.history.append(operand)

        return torch.stack(tuple(self.history), dim=0).median(dim=0).values

@decorator_common
class HistoryMedianIndices(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(n=Integer.partial(min=2, max=20))
    DEFAULTS = dict(n=10)

    def initialize(self, ref):
        self.history = TList()

    def forward(self, state):
        operand = self.get_operand(0, state)
        n = self.get_hyperparam("n")

        assert len(self.history) <= n

        if len(self.history) == n:
            self.history.pop(0)

        self.history.append(operand)

        return torch.stack(tuple(self.history), dim=0).median(dim=0).indices.to(operand)

@decorator_common
class HistoryArgmax(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(n=Integer.partial(min=2, max=20))
    DEFAULTS = dict(n=10)

    def initialize(self, ref):
        self.history = TList()

    def forward(self, state):
        operand = self.get_operand(0, state)
        n = self.get_hyperparam("n")

        assert len(self.history) <= n

        if len(self.history) == n:
            self.history.pop(0)

        self.history.append(operand)

        return torch.stack(tuple(self.history), dim=0).argmax(dim=0).to(operand)

@decorator_common
class HistoryMaximum(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(n=Integer.partial(min=2, max=20))
    DEFAULTS = dict(n=10)

    def initialize(self, ref):
        self.history = TList()

    def forward(self, state):
        operand = self.get_operand(0, state)
        n = self.get_hyperparam("n")

        assert len(self.history) <= n

        if len(self.history) == n:
            self.history.pop(0)

        self.history.append(operand)

        return torch.stack(tuple(self.history), dim=0).amax(dim=0)

@decorator_common
class HistoryMinimum(BaseOperation):
    N_OPERANDS = 1
    HYPERPARAMS = dict(n=Integer.partial(min=2, max=20))
    DEFAULTS = dict(n=10)

    def initialize(self, ref):
        self.history = TList()

    def forward(self, state):
        operand = self.get_operand(0, state)
        n = self.get_hyperparam("n")

        assert len(self.history) <= n

        if len(self.history) == n:
            self.history.pop(0)

        self.history.append(operand)

        return torch.stack(tuple(self.history), dim=0).amin(dim=0)