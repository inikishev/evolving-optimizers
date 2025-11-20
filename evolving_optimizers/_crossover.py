import warnings
import random

from ._bases import BaseCrossover, BaseOperation


def _select_low_depth_branch_idxs(root: BaseOperation):
    idxs = [[]] + root.get_flat_idxs()
    depth = max(len(i) for i in idxs)
    weights = [(depth - len(i)) + 1 for i in idxs]
    weights[0] = 1
    return random.choices(idxs, weights, k=1)[0]

def _select_high_depth_branch_idxs(root: BaseOperation):
    idxs = [[]] + root.get_flat_idxs()
    weights = [len(i) for i in idxs]
    return random.choices(idxs, weights, k=1)[0]

class CrossoverSwap(BaseCrossover):
    def cross(self, pool, tree1, tree2):
        # we take a random branch from each and swap them
        # if either operation has no operands, return a random tree
        if tree1.N_OPERANDS == 0 or tree2.N_OPERANDS == 0:
            return (pool.random_tree(0), )

        # select branches
        for _ in range(100):
            mode = random.choice([0,1,2])
            if mode == 0:
                idxs1 = _select_low_depth_branch_idxs(tree1)
                idxs2 = _select_low_depth_branch_idxs(tree2)
            elif mode == 1:
                idxs1 = _select_high_depth_branch_idxs(tree1)
                idxs2 = _select_high_depth_branch_idxs(tree2)
            else:
                idxs1 = _select_low_depth_branch_idxs(tree1)
                idxs2 = _select_high_depth_branch_idxs(tree2)

            if len(idxs1) == len(idxs2) == 0: continue

            children: list[BaseOperation] = []
            if len(idxs1) > 0:
                child1 = tree1.clone()
                child1.replace_branch_by_idxs_(idxs1, tree2.select_branch_by_idxs(idxs2))
                children.append(child1)

            if len(idxs2) > 0:
                child2 = tree2.clone()
                child2.replace_branch_by_idxs_(idxs2, tree1.select_branch_by_idxs(idxs1))
                children.append(child2)

            if len(children) == 2 and (children[0].string() == children[1].string()):
                continue

            return tuple(children)

        warnings.warn("failed to cross after 100 tries, returning random tree")
        return (pool.random_tree(0), )

