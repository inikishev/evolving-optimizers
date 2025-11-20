import math
import warnings
import random

from ._bases import BaseMutation, BaseOperation


class PointMutation(BaseMutation):
    """picks random hyperparameter and sets it to a random value"""
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        branches = [o for o in tree.flat_branches() if len(o.hyperparams) > 0]
        if len(branches) == 0:
            return RandomMutation().mutate(pool, tree, sigma)

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

class PerturbMutation(BaseMutation):
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()
        perturbed = _tree_mutate_perturb_(tree, sigma)

        # make sure at least 1 perturbed
        if not perturbed:

            branches = [o for o in tree.flat_branches() if len(o.hyperparams) > 0]
            if len(branches) == 0:
                return RandomMutation().mutate(pool, tree, sigma)

            branch = random.choice(branches)
            hyperparam = random.choice(tuple(branch.hyperparams.values()))
            hyperparam.mutate_perturb_(sigma)

        return tree

class BranchMutation(BaseMutation):
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        if tree.N_OPERANDS == 0:
            return pool.random_tree(0)

        # pick index of a branch to replace
        idxs = tree.get_flat_idxs()
        depth = max(len(i) for i in idxs)
        target_depth = depth * (1-sigma)

        inv_weights = [abs(len(i) - target_depth) for i in idxs]
        maxv = max(inv_weights)
        weights = [max((maxv - iw), 0.5)**2 for iw in inv_weights]

        idx = random.choices(idxs, weights, k=1)[0]

        # replace branch with a new random tree
        tree.replace_branch_by_idxs_(idx, pool.random_tree(len(idx)))

        return tree

class InsertMutation(BaseMutation):
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        if tree.N_OPERANDS == 0:
            new_root = pool.random_tree(0, weight_fn=lambda x: x.N_OPERANDS > 0)
            idx = random.choice(list(range(new_root.N_OPERANDS)))
            new_root.operands[idx] = tree
            return new_root

        # pick index of a branch to insert a new operation before it
        idxs = [[]] + tree.get_flat_idxs()
        depth = max(len(i) for i in idxs)
        target_depth = depth * (1-sigma)

        inv_weights = [abs(len(i) - target_depth) for i in idxs]
        maxv = max(inv_weights)
        weights = [max((maxv - iw), 0.5)**2 for iw in inv_weights]

        idx = random.choices(idxs, weights, k=1)[0]

        # generate the new branch
        new_branch = pool.random_tree(len(idx), weight_fn=lambda x: x.N_OPERANDS > 0)

        # select a random operand in the new branch
        # and replace it with current branch
        new_idx = random.choice(list(range(new_branch.N_OPERANDS)))
        new_branch.operands[new_idx] = tree.select_branch_by_idxs(idx)

        # replace current branch with one with new operation inserted
        if len(idx) == 0:
            return new_branch

        tree.replace_branch_by_idxs_(idx, new_branch)
        return tree

class TruncateMutation(BaseMutation):
    """same as replace except it always replaces a branch with a leaf to counteract insert"""
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        if tree.N_OPERANDS == 0:
            return pool.random_tree(0)

        # pick an index to replace with a leaf, it must have operands with children
        idxs = tree.get_flat_idxs()
        depth = max(len(i) for i in idxs)
        target_depth = depth * (1-sigma)

        inv_weights = [abs(len(i) - target_depth) for i in idxs]
        maxv = max(inv_weights)

        def idx_weight(idx):
            branch = tree.select_branch_by_idxs(idx)
            if branch.N_OPERANDS == 0:
                return 1e-4

            dist = abs(len(idx) - target_depth)
            return max((maxv - dist), 0.5) ** 2

        idx, _ = tree.select_random_branch(weight_fn=idx_weight)

        # generate a leaf to replace the branch with
        leaf = pool.random_tree(0, weight_fn=lambda x: (x.N_OPERANDS == 0) + 1e-4)
        tree.replace_branch_by_idxs_(idx, leaf)

        return tree

class CutMutation(BaseMutation):
    """picks a random branch"""
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        if tree.N_OPERANDS == 0:
            return pool.random_tree(0)

        # select an index of a branch to pick
        idxs = tree.get_flat_idxs()
        depth = max(len(i) for i in idxs)
        target_depth = depth * sigma

        inv_weights = [abs(len(i) - target_depth) for i in idxs]
        maxv = max(inv_weights)

        def idx_weight(idx):
            branch = tree.select_branch_by_idxs(idx)
            if branch.N_OPERANDS == 0:
                return 1e-4

            dist = abs(len(idx) - target_depth)
            return max((maxv - dist), 0.5) ** 2

        _, branch = tree.select_random_branch(weight_fn=idx_weight)
        return branch

class SimplifyMutation(BaseMutation):
    """removes an operand in the middle"""
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        if tree.N_OPERANDS == 0:
            return pool.random_tree(0)

        # we need a branch with another branch as an operand
        idxs = tree.get_flat_idxs()
        depth = max(len(i) for i in idxs)
        target_depth = depth * (1 - sigma)

        inv_weights = [abs(len(i) - target_depth) for i in idxs]
        maxv = max(inv_weights)

        def idx_weight(idx):
            branch = tree.select_branch_by_idxs(idx)
            if branch.N_OPERANDS == 0:
                return 1e-4
            if sum(op.N_OPERANDS for op in branch.operands) == 0:
                return 1e-4

            dist = abs(len(idx) - target_depth)
            return max((maxv - dist), 0.5) ** 2

        idx, branch = tree.select_random_branch(weight_fn=idx_weight)

        # pick an operand which is a branch
        sub_branches = [(i,b) for i,b in enumerate(branch.operands) if b.N_OPERANDS != 0]
        if len(sub_branches) == 0:
            return RandomMutation().mutate(pool, tree, sigma)

        # pick a random operand from the sub branch
        sub_branch = random.choice(sub_branches)
        operand = random.choice(sub_branch[1].operands)

        # replace operand with its subbranch
        tree.replace_branch_by_idxs_(idx + [sub_branch[0]], operand)
        return tree



class ReplaceMutation(BaseMutation):
    def mutate(self, pool, tree, sigma):
        tree = tree.clone()

        if tree.N_OPERANDS == 0:
            return pool.random_tree(0)

        # pick a branch to replace
        idxs = tree.get_flat_idxs()
        depth = max(len(i) for i in idxs)
        target_depth = depth * (1-sigma)

        inv_weights = [abs(len(i) - target_depth) for i in idxs]
        maxv = max(inv_weights)
        weights = [max((maxv - iw), 0.5)**2 for iw in inv_weights]

        idx = random.choices(idxs, weights, k=1)[0]
        cur_branch = tree.select_branch_by_idxs(idx)

        # generate new random tree with same number of immediate operands
        # and set operands to current ones
        new_branch = pool.random_tree(len(idx), weight_fn=lambda x: (x.N_OPERANDS == cur_branch.N_OPERANDS)+1e-4)
        if new_branch.N_OPERANDS == cur_branch.N_OPERANDS:
            new_branch.operands = cur_branch.operands.copy()

        tree.replace_branch_by_idxs_(idx, new_branch)

        return tree

class RandomMutation(BaseMutation):
    def __init__(self, *mutations: BaseMutation):
        if len(mutations) == 0:
            mutations = (
                PointMutation(), PerturbMutation(),
                BranchMutation(),  InsertMutation(),
                TruncateMutation(), ReplaceMutation(),
                SimplifyMutation(), CutMutation(),
            )

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

