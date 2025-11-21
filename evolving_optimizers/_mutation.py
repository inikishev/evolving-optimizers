import math
import random
import warnings

import numpy as np

from ._bases import BaseMutation, BaseOperation


class MutateRandomizeHyperparam(BaseMutation):
    """picks random hyperparameter and sets it to a random value"""
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        branches = [o for o in tree.flat_branches() if len(o.hyperparams) > 0]
        if len(branches) == 0:
            return RandomMutationCombo().mutate(pool, tree, sigma)

        branch = random.choice(branches)
        hyperparam = random.choice(tuple(branch.hyperparams.values()))
        if random.random() < sigma:
            hyperparam.mutate_point_()

        else:
            hyperparam.mutate_perturb_(sigma)

        return tree

def _tree_mutate_perturb_(tree: BaseOperation, sigma):
    perturbed = False

    for v in tree.hyperparams.values():

        # here we use 0 mode to make more perturbs
        if random.triangular(0, 1, 0) < math.sqrt(sigma):
            perturbed = True
            v.mutate_perturb_(sigma)

    for branch in tree.operands:
        if _tree_mutate_perturb_(branch, sigma):
            perturbed = True

    return perturbed

class MutatePerturbHyperparam(BaseMutation):
    """perturbs all hyperparams slightly"""
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()
        perturbed = _tree_mutate_perturb_(tree, sigma)

        # make sure at least 1 perturbed
        if not perturbed:

            branches = [o for o in tree.flat_branches() if len(o.hyperparams) > 0]
            if len(branches) == 0:
                return RandomMutationCombo().mutate(pool, tree, sigma)

            branch = random.choice(branches)
            hyperparam = random.choice(tuple(branch.hyperparams.values()))
            hyperparam.mutate_perturb_(sigma)

        return tree

class MutateReplaceBranch(BaseMutation):
    """replaces a branch with a randomly generated"""
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        if tree.N_OPERANDS == 0:
            return pool.random_tree(0)

        _, idx = tree.pick_random_branch(1-sigma, self_weight=None)
        tree.replace_branch_by_idx_(idx, pool.random_tree(len(idx)))

        return tree

class MutateInsertIntoRandomBranch(BaseMutation):
    """picks a branch and inserts it as operand into a randomly generated branch, which replaces it"""
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        if tree.N_OPERANDS == 0:
            new_root = pool.random_tree(0, weight_fn=lambda x: x.N_OPERANDS > 0)
            idx = random.choice(list(range(new_root.N_OPERANDS)))
            new_root.operands[idx] = tree
            return new_root

        # pick a branch to insert and replace
        branch, idx = tree.pick_random_branch(1-sigma, self_weight=0.5)

        # generate the new branch
        new_branch = pool.random_tree(len(idx), weight_fn=lambda x: x.N_OPERANDS > 0)

        if new_branch.N_OPERANDS > 0:
            # select a random operand in the new branch
            # and replace it with current branch
            new_idx = random.randrange(0, new_branch.N_OPERANDS)
            new_branch.operands[new_idx] = branch
        else:
            warnings.warn("MutateInsertIntoRandomBranch picked a branch with 0 operands")

        # replace current branch with one with new operation inserted
        if len(idx) == 0: # means it picked root
            return new_branch

        tree.replace_branch_by_idx_(idx, new_branch)
        return tree

class MutateTruncateBranch(BaseMutation):
    """same as replace except it always replaces a branch with a leaf to counteract insert"""
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        if tree.N_OPERANDS == 0:
            return pool.random_tree(0)

        # pick a branch with more than 0 operands
        def no_leaf(branch: BaseOperation, idx: list[int]):
            if branch.N_OPERANDS == 0: return 0
            return 1

        _, idx = tree.pick_random_branch(1-sigma, weight_fn=no_leaf, self_weight=None)

        # replace with a random 0 operand branch
        leaf = pool.random_tree(len(idx), weight_fn=lambda x: x.N_OPERANDS == 0)
        tree.replace_branch_by_idx_(idx, leaf)

        return tree

class MutatePickBranch(BaseMutation):
    """Picks a branch with more than 0 operands and returns it."""
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        if tree.N_OPERANDS == 0:
            return pool.random_tree(0)

        # we want to cut a branch with more than 0 operands
        def no_leaf(branch: BaseOperation, idx: list[int]):
            if branch.N_OPERANDS == 0: return 0
            return 1

        branch, _ = tree.pick_random_branch(sigma, weight_fn=no_leaf, self_weight=None)
        return branch

class MutateCutMiddle(BaseMutation):
    """removes an operand in the middle"""
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        if tree.N_OPERANDS == 0:
            return pool.random_tree(0)

        # we need a branch with another branch as an operand
        def weight_fn(branch: BaseOperation, idx: list[int]):
            if branch.N_OPERANDS == 0:
                return 0

            if sum(op.N_OPERANDS for op in branch.operands) == 0:
                return 0

            return 1

        branch, idx = tree.pick_random_branch((1-sigma), weight_fn=weight_fn, self_weight=1)

        # pick an operand which is a branch
        sub_branches = [(i,b) for i,b in enumerate(branch.operands) if b.N_OPERANDS != 0]
        if len(sub_branches) == 0:
            return RandomMutationCombo().mutate(pool, tree, sigma)

        # pick a random operand from the sub branch
        sub_idx, sub_branch = random.choice(sub_branches)
        operand = random.choice(sub_branch.operands)

        # replace operand with its subbranch
        tree.replace_branch_by_idx_(idx + [sub_idx], operand)
        return tree



class MutateReplaceOperand(BaseMutation):
    """replaces operand in a branch with another random operand with same number of operands"""
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        if tree.N_OPERANDS == 0:
            return pool.random_tree(0)

        # pick a branch to replace
        branch, idx = tree.pick_random_branch(1-sigma, self_weight=1)

        # generate new random tree with same number of operands
        def weight_fn(operand: type[BaseOperation]):
            if operand.N_OPERANDS != branch.N_OPERANDS: return 0
            if isinstance(branch, operand): return 0.1
            return 1

        new_branch = pool.random_tree(len(idx), weight_fn=weight_fn)

        # and set operands to current ones
        if new_branch.N_OPERANDS == branch.N_OPERANDS:
            new_branch.operands = branch.operands.copy()

        if len(idx) == 0:
            return new_branch

        tree.replace_branch_by_idx_(idx, new_branch)
        return tree

class MutateSwapBranches(BaseMutation):
    """swaps two branches"""
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        if tree.N_OPERANDS == 0:
            return pool.random_tree(0)

        branch1, idx1 = tree.pick_random_branch(1-sigma, self_weight=None)

        def no_same_branch(branch: BaseOperation, idx: list[int]):
            if idx == idx1[:len(idx)]: return 0
            if idx1 == idx[:len(idx1)]: return 0
            return 1

        branch2, idx2 = tree.pick_random_branch(1-sigma, weight_fn=no_same_branch, self_weight=None, unbiased=False)

        if no_same_branch(branch2, idx2) == 0:
            return RandomMutationCombo().mutate(pool, tree, sigma)

        tree.replace_branch_by_idx_(idx1, branch2.clone())
        tree.replace_branch_by_idx_(idx2, branch1)

        return tree


MUTATIONS = (
    MutateRandomizeHyperparam(),
    MutatePerturbHyperparam(),
    MutateReplaceBranch(),
    MutateInsertIntoRandomBranch(),
    MutateTruncateBranch(),
    MutateReplaceOperand(),
    MutateCutMiddle(),
    MutatePickBranch(),
    MutateSwapBranches(),
)

class RandomMutation(BaseMutation):
    def __init__(self, *mutations: BaseMutation):
        if len(mutations) == 0:
            mutations = MUTATIONS

        self.mutations = mutations

    def mutate(self, pool, tree, sigma):
        string = tree.string()
        for _ in range(100):
            mutation = random.choice(self.mutations)
            mutated = mutation.mutate(pool, tree, sigma)
            if mutated.string() != string:
                return mutated

        warnings.warn("failed to mutate after 100 attempts, returning random tree")
        return pool.random_tree(0)

class RandomMutationCombo(BaseMutation):
    """has a chance to apply multiple mutations distributing sigma across them."""
    def __init__(self, *mutations: BaseMutation):
        if len(mutations) == 0:
            mutations = MUTATIONS

        self.mutations = mutations

    def mutate(self, pool, tree, sigma):
        string = tree.string()

        choices = list(range(1, len(self.mutations)+1))
        weights = [2**(c-1) for c in choices]
        weights.reverse()

        n_mutations = random.choices(choices, weights=weights)

        sigmas = np.random.triangular(0, 0, 1, size=n_mutations)
        sigmas = sigmas / sigmas.sum()
        sigmas = sigmas * sigma

        for _ in range(100):
            mutated = tree

            for s in sigmas:
                mutation = random.choice(self.mutations)
                mutated = mutation.mutate(pool, mutated, sigma=s)

            if mutated.string() != string:
                return mutated

        warnings.warn("failed to mutate after 100 attempts, returning random tree")
        return pool.random_tree(0)
