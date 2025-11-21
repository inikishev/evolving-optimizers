import warnings
import random

from ._bases import BaseCrossover, BaseOperation

class CrossoverSwap(BaseCrossover):
    def cross(self, pool, tree1, tree2):
        # we take a random branch from each and swap them
        # if either operation has no operands, return a random tree
        if tree1.N_OPERANDS == 0 or tree2.N_OPERANDS == 0:
            return (pool.random_tree(0), )

        parent_strings = [tree1.string(), tree2.string]

        # select branches
        for _ in range(100):

            target_rdepth = random.uniform(0, 1)
            branch1, idxs1 = tree1.pick_random_branch(
                target_rdepth = target_rdepth,
                self_weight = random.choice([0.25, None]),
            )

            branch2, idxs2 = tree2.pick_random_branch(
                target_rdepth = target_rdepth,
                self_weight = random.choice([0.25, None]),
            )

            if len(idxs1) == len(idxs2) == 0: continue

            children: list[BaseOperation] = []
            if len(idxs1) > 0:
                child1 = tree1.clone()
                child1.replace_branch_by_idx_(idxs1, branch2)
                children.append(child1)

            if len(idxs2) > 0:
                child2 = tree2.clone()
                child2.replace_branch_by_idx_(idxs2, branch1)
                children.append(child2)

            # make sure crossover did something
            strings = [c.string() for c in children]
            strings = [s for s in strings if s not in parent_strings]

            # if childs match pick first one
            if (len(strings) == 2) and (strings[0] == strings[1]):
                children = [children[0]]

            # all childs are same as parents discard them
            if len(strings) == 0:
                continue

            return tuple(children)

        warnings.warn("failed to cross after 100 tries, returning random tree")
        return (pool.random_tree(0), )

