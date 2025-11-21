import os
import warnings
import math
import random
from abc import ABC, abstractmethod
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, NamedTuple

import numpy as np
import torch

from ._bases import BaseCrossover, BaseMutation, BaseOperation, BasePool
from ._crossover import CrossoverSwap
from ._mutation import RandomMutation


class Solution:
    def __init__(self, tree: BaseOperation, fitness: float, data: Any = None):
        self.tree: BaseOperation = tree
        self.fitness: float = fitness
        """lower is better"""

        self.data: Any = data
        self.optimizer_data: dict = {}

    def copy(self):
        "doesnt copy tree but shallow copies optimizer data"
        sol_copy = Solution(self.tree, self.fitness, self.data)
        sol_copy.optimizer_data = self.optimizer_data.copy()
        return sol_copy

def _solution_to_cpu(solution: Solution):
    tree_cpu = solution.tree.clone()
    tree_cpu.to_(device="cpu")
    sol_cpu = Solution(tree_cpu, solution.fitness, solution.data)
    sol_cpu.optimizer_data = solution.optimizer_data
    return sol_cpu

def _remove_duplicates(solutions: list[Solution]) -> list[Solution]:
    strings = []
    filtered = []

    solutions = solutions.copy()
    random.shuffle(solutions)

    for solution in solutions:
        string = solution.tree.string()
        if string in strings: continue
        filtered.append(solution)
        strings.append(string)

    return filtered

class Optimizer(ABC):
    @abstractmethod
    def step(self, objective: Callable[[list[BaseOperation]], list[Solution]], population: list[Solution], pool: BasePool) -> list[Solution]:
        """one step"""


class BaseSelectionStrategy(ABC):
    @abstractmethod
    def select(self, solutions: list[Solution], n: int) -> tuple[list[Solution], list[Solution]]:
        """selects two populations. First will carry over to next generation as is, and will participate in crossover/mutation. Second is only for crossover/mutation. n is number of individuals in second population (doesn't have to be exactly that number)"""

class RankWeightedSelection(BaseSelectionStrategy):
    def __init__(self, topk: int, proc: Callable[[np.ndarray], np.ndarray] = lambda x: x):
        self.topk = topk
        self.proc = proc

    def select(self, solutions, n: int):
        solutions = [p for p in solutions if math.isfinite(p.fitness)]
        solutions.sort(key=lambda x: (x.fitness, len(x.tree.flat_branches())))
        topk = solutions[:self.topk]

        under_topk = solutions[self.topk:]
        weights = np.array(list(range(len(under_topk))), dtype=np.float64)
        weights = self.proc((weights.max() - weights) + 1)

        other = random.choices(under_topk, weights.tolist(), k=n)

        return topk, other

class TopkGA(Optimizer):
    """here is what goes into next generation:
    1. solutions from full population selected by selection strategy with mutation and crossover based on ``mutation_rate`` and ``crossover_rate``, crossover includes topk solutions.
    2. top k solutions
    3. top k solutions with mandatory mutations
    4. top k solutions crossed with other solutions from top k
    5. top k solutions crossed with solutions from main population
    6. ``n_random`` new random soltions

    note that selection doesn't have much effect here because it only removes overflow solutions.
    """
    def __init__(self, pop_size: int = 100, n_random: int = 10, crossover_rate: float=0.5, mutation_rate: float = 1e-1, mutation_exp: float = 1, selection: BaseSelectionStrategy = RankWeightedSelection(5), crossover: BaseCrossover = CrossoverSwap(), mutation: BaseMutation = RandomMutation()):
        self.pop_size = pop_size
        self.n_random = n_random
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_exp = mutation_exp

        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def step(self, objective, population, pool):
        population = _remove_duplicates(population)

        if len(population) < self.pop_size:
            extra = [pool.random_tree(0) for _ in range(self.pop_size - len(population))]
            population.extend(objective(extra))

        topk, candidates = self.evolve(pool, population)
        return topk + objective(candidates)

    def evolve(self, pool: BasePool, solutions: list[Solution]) -> tuple[list[Solution], list[BaseOperation]]:
        top_sol, main_sol = self.selection.select(solutions, n=self.pop_size)
        new_pop = []

        # top k with mutations
        mutated = [self.mutation.mutate(pool, p.tree, random.triangular(0,1,0)**self.mutation_exp) for p in top_sol]
        new_pop.extend(mutated)

        # cross topk with topk
        for sol in top_sol:
            sol2 = random.choice(top_sol)
            new_pop.append(random.choice(self.crossover.cross(pool, sol.tree, sol2.tree)))

        # cross topk with all
        all_sol = top_sol + main_sol
        for sol in top_sol:
            sol2 = random.choice(all_sol)
            new_pop.append(random.choice(self.crossover.cross(pool, sol.tree, sol2.tree)))

        # random
        for _ in range(self.n_random):
            new_pop.append(pool.random_tree(0))

        # main
        for sol in main_sol:
            if random.random() < self.crossover_rate:
                sol2 = random.choice(all_sol)
                trees = self.crossover.cross(pool, sol.tree, sol2.tree)
            else:
                trees = (sol.tree, )

            muta: list[BaseOperation] = []
            for tree in trees:
                if random.random() < self.mutation_rate:
                    sigma = random.triangular(0,1,0)**self.mutation_exp
                    tree = self.mutation.mutate(pool, tree, sigma)
                muta.append(tree)

            new_pop.extend(muta)

        return top_sol, new_pop

class OnePlusOne(Optimizer):
    """mutates and accepts or rejects depending on whether mutation helped"""
    def __init__(self, n_candidates: int = 1, mutation: BaseMutation = RandomMutation(), mutation_exp:float = 1):
        self.mutation = mutation
        self.n_candidates = n_candidates
        self.mutation_exp = mutation_exp

    def step(self, objective, population, pool):
        population = _remove_duplicates(population)

        if len(population) == 0:
            population = objective([pool.random_tree(0)])

        mutated_trees = []
        for _ in range(self.n_candidates):
            for ind in population:
                sigma = random.triangular(0,1,0)**self.mutation_exp
                mutated_trees.append(self.mutation.mutate(pool, ind.tree, sigma))

        mutated = objective(mutated_trees)

        full = sorted(population + mutated, key=lambda x: (x.fitness, len(x.tree.flat_branches())))
        return [full[0]]

class ARS(Optimizer):
    """ARS"""
    def __init__(self, mul=0.9, maxiter=50, mutation: BaseMutation = RandomMutation()):
        self.mutation = mutation
        self.mul = mul
        self.maxiter = maxiter

    def step(self, objective, population, pool):
        population = _remove_duplicates(population)

        if len(population) == 0:
            population = objective([pool.random_tree(0)])

        if len(population) > 1:
            warnings.warn("ARS only supports population size of 1 so only best solution is kept from initial population")
            population = [population[np.argmin([i.fitness for i in population])]]

        solution = population[0]
        sigma = 1

        # start with completely random
        candidate = objective([pool.random_tree(0)])[0]
        if candidate.fitness < solution.fitness:
            return [candidate]

        # local-er search
        for i in range(self.maxiter):
            candidate = objective([self.mutation.mutate(pool, solution.tree, sigma)])[0]

            if candidate.fitness < solution.fitness:
                return [candidate]

            sigma *= self.mul

        return [solution]

class K1Plus1(Optimizer):
    """we maintain ``pop_size`` solutions, update each solution with one plus one and replace bottom ``k`` solutions with random/crossovers.

    If initial population is larger than ``pop_size``, it will remove ``k`` worst solutions per step until it reaches ``pop_size``.
    """
    def __init__(self, pop_size:int = 10, k:int = 5, mutation_exp:float = 1, crossover_prob:float=0.5, mutation: BaseMutation = RandomMutation(), crossover: BaseCrossover = CrossoverSwap()):
        self.pop_size = pop_size
        self.k = k
        self.crossover_prob = crossover_prob

        self.mutation = mutation
        self.mutation_exp = mutation_exp
        self.crossover = crossover

    def step(self, objective, population, pool):
        population = _remove_duplicates(population)

        if len(population) < self.pop_size:
            extra = [pool.random_tree(0) for _ in range(self.pop_size - len(population))]
            population.extend(objective(extra))

        # mutate all solutions
        mutated_trees = []
        for ind in population:
            sigma = random.triangular(0,1,0)**self.mutation_exp
            mutated_trees.append(self.mutation.mutate(pool, ind.tree, sigma))

        # evaluate mutated
        mutated = objective(mutated_trees)

        # accept or reject
        best: list[Solution] = []
        for orig, muta in zip(population, mutated):
            if orig.fitness <= muta.fitness:
                best.append(orig)
            else:
                best.append(muta)

        # replace bottom k
        best.sort(key=lambda x: (x.fitness, len(x.tree.flat_branches())))

        bestk = best[:-self.k]
        new_pop: list[BaseOperation] = []

        while (len(new_pop) + len(bestk)) < self.pop_size:
            if random.random() < self.crossover_prob:
                ind1, ind2 = random.choices(bestk, weights=list(range(len(bestk), 0, -1)), k=2)
                new_pop.extend(self.crossover.cross(pool, ind1.tree, ind2.tree))
            else:
                new_pop.append(pool.random_tree(0))

        if len(new_pop) == 0:
            return bestk

        return bestk + objective(new_pop)

class IslandGA(Optimizer):
    """Runs multiple optimizers and every once in a while exchanges individuals from their populations

    Args:
        optimizers (Sequence[Optimizer]): optimizers.
        exchange_prob (float, optional):
            probability rolled for each optimizer to exchange one solution with another optimizer,
            if a roll succeeds, it rolls again. Defaults to 0.05.
        crossover_prob (float, optional):
            probability of crossing the exchanged solutions from two optimizers. Defaults to 0.1.
        mutation_prob (float, optional): probability of mutating the exchanged solutions. Defaults to 0.1.
        two_way (bool, optional):
            if True, during an exchange between two optimizers, both receive one solution from another optimizer.
            If False, only first optimizer receives a solution from second. Defaults to False.
        random_init (int | None, optional):
            if specified, will generate this many random trees per each optimizer on first step to initilize it.
            Initial population will be appended to random trees. Defaults to None.
        mutation (BaseMutation, optional): type of mutation. Defaults to RandomMutation().
        crossover (BaseCrossover, optional): type of crossover. Defaults to CrossoverSwap().
    """

    def __init__(
        self,
        optimizers: Sequence[Optimizer],
        exchange_prob: float = 0.05,
        crossover_prob: float = 0.5,
        mutation_prob: float = 0.1,
        two_way: bool = False,
        remove_duplicates: bool = True,
        random_init: int | None = None,
        mutation: BaseMutation = RandomMutation(),
        crossover: BaseCrossover = CrossoverSwap(),
    ):
        self.optimizers = list(optimizers)
        self.exchange_prob = exchange_prob
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.two_way = two_way
        self.random_init = random_init
        self.mutation = mutation
        self.crossover = crossover
        self.remove_duplicates = remove_duplicates


    def step(self, objective, population, pool):
        if self.remove_duplicates:
            population = _remove_duplicates(population)

        islands: list[list[Solution]] = []

        # clear optimizers that do not exist (like if population is initialized to previous set of opts)
        ids = [id(opt) for opt in self.optimizers]
        for ind in population:
            if "optimizer_id" in ind.optimizer_data:
                if ind.optimizer_data["optimizer_id"] not in ids:
                    warnings.warn(f"optimizer {ind.optimizer_data['optimizer_id']} is not in optimizers, deleting it")
                    del ind.optimizer_data["optimizer_id"]

        # initialize islands on 1st step
        if all("optimizer_id" not in ind.optimizer_data for ind in population):
            for opt in self.optimizers:
                island = [s.copy() for s in population]
                if self.random_init is not None:
                    island = population + objective([pool.random_tree() for _ in range(self.random_init)])

                island = opt.step(objective, island, pool)
                if len(island) == 0: warnings.warn(f"{opt} returned an empty island")

                for ind in island:
                    ind.optimizer_data["optimizer_id"] = id(opt)

                islands.append([s.copy() for s in island])

        else:
            # if there are any individuals without island distributed them randomly
            no_island = [ind for ind in population if "optimizer_id" not in ind.optimizer_data]
            if len(no_island) > 0:
                warnings.warn("IslandGA received new solutions")
                for ind in no_island:
                    ind.optimizer_data["optimizer_id"] = random.choice(ids)

            # step each optimizer with its island
            for opt in self.optimizers:
                island = [s.copy() for s in population if s.optimizer_data["optimizer_id"] == id(opt)]

                if len(island) == 0:
                    warnings.warn(f"island for {opt} is empty")
                    if self.random_init is not None:
                        island = objective([pool.random_tree() for _ in range(self.random_init)])

                island = opt.step(objective, island, pool)

                for ind in island:
                    ind.optimizer_data["optimizer_id"] = id(opt)

                islands.append([s.copy() for s in island])

        # exchange
        orig_islands = islands.copy()

        for j, island in enumerate(orig_islands):
            while random.random() < self.exchange_prob:
                other_idxs = list(range(len(orig_islands)))
                del other_idxs[j]

                island2 = orig_islands[random.choice(other_idxs)]
                ind1 = random.choice(island)
                ind2 = random.choice(island2)

                opt_id1 = ind1.optimizer_data["optimizer_id"]
                opt_id2 = ind2.optimizer_data["optimizer_id"]
                assert opt_id1 != opt_id2

                # if crossed over both go into both
                # random crossover
                if random.random() < self.crossover_prob:
                    into_opt1 = into_opt2 = self.crossover.cross(pool, ind1.tree, ind2.tree)
                else:
                    into_opt1 = (ind2.tree, ) # tree from second goes into first
                    into_opt2 = (ind1.tree, ) # tree from first goes into second

                # random mutation
                def mutate(ind: BaseOperation):
                    if random.random() < self.mutation_prob:
                        sigma = random.triangular(0,1,0)
                        return self.mutation.mutate(pool, ind, sigma)
                    return ind.clone()

                # evaluate new trees and add them
                if self.two_way:
                    into_opt1 = [mutate(ind) for ind in into_opt1]
                    into_opt2 = [mutate(ind) for ind in into_opt2]

                    sols = objective(into_opt1 + into_opt2)
                    sols_into_opt1 = sols[:len(into_opt1)]
                    sols_into_opt2 = sols[len(into_opt1):]

                    # move 2nd to 1st
                    for sol in sols_into_opt1:
                        sol.optimizer_data["optimizer_id"] = opt_id1

                    # move 1st to 2nd
                    for sol in sols_into_opt2:
                        sol.optimizer_data["optimizer_id"] = opt_id2

                    islands.append(sols_into_opt1)
                    islands.append(sols_into_opt2)

                else:
                    into_opt1 = [mutate(ind) for ind in into_opt1]
                    sols_into_opt1 = objective(into_opt1)

                    # move 2nd to 1st
                    for sol in sols_into_opt1:
                        sol.optimizer_data["optimizer_id"] = opt_id1

                    islands.append(sols_into_opt1)

        return [sol for island in islands for sol in island]


class Runner:
    def __init__(self, store_k_best: int | float = 100):
        self.best_solutions: list[Solution] = []
        self._best_strings: list[str] = []

        self.store_k_best = store_k_best
        self.fitness_history = []

    def evaluate(self, population: list[BaseOperation], objective: Callable[[list[BaseOperation]], list[Solution]]):
        solutions = objective(population)
        for sol in solutions:
            self.fitness_history.append(sol.fitness)

            string = sol.tree.string()

            if string not in self._best_strings:
                cur = (sol.fitness, len(sol.tree.flat_branches()))
                idx = bisect_right(self.best_solutions, cur, key=lambda x: (x.fitness, len(x.tree.flat_branches())))
                self.best_solutions.insert(idx, _solution_to_cpu(sol))
                self._best_strings.insert(idx, string)

            if len(self.best_solutions) > self.store_k_best:
                self.best_solutions.pop(-1)
                self._best_strings.pop(-1)

        return solutions

    def step(self, objective: Callable[[list[BaseOperation]], list[Solution]], optimizer: Optimizer, population: list[Solution], pool: BasePool):
        solutions = optimizer.step(partial(self.evaluate, objective=objective), population=population, pool=pool)
        return solutions

def map_objective(objective: Callable[[BaseOperation], float]):
    def mapped(trees: list[BaseOperation]):
        losses = [objective(tree) for tree in trees]
        return [Solution(tree, float(loss)) for tree, loss in zip(trees, losses)]
    return mapped


class CachedObjective:
    def __init__(self, objective: Callable[[list[BaseOperation]], list[Solution]]):
        self.objective = objective
        self.values: dict[str, float] = {}
        self.datas: dict[str, Any] = {}

    def __call__(self, trees: list[BaseOperation]) -> list[Solution]:
        strings = [t.string() for t in trees]

        new_pairs = [(tree, string) for tree, string in zip(trees, strings) if string not in self.values]
        new_sols = self.objective([tree for tree, string in new_pairs])

        for (_, string), sol in zip(new_pairs, new_sols):
            self.values[string] = sol.fitness
            self.datas[string] = sol.data

        return [Solution(tree, self.values[string], self.datas[string]) for tree, string in zip(trees, strings)]


class DiskCachedObjective:
    def __init__(self, file: os.PathLike | str, objective: Callable[[list[BaseOperation]], list[Solution]]):
        self.file = file
        self._open_file = None
        self.objective = objective
        self.values: dict[str, float] = {}
        self.datas: dict[str, Any] = {}

        if os.path.exists(self.file):
            with open(self.file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if len(line) > 0:
                        string, value = line.rsplit(": ", 1)
                        self.values[string] = float(value)

    def open(self):
        assert self._open_file is None
        self._open_file = open(self.file, "a", encoding='utf-8')
        return self

    def close(self):
        assert self._open_file is not None
        self._open_file.close()
        self._open_file = None

        # sort
        lines = []
        for k,v in sorted(self.values.items(), key=lambda x: x[1]):
            lines.append(f'{k}: {v}\n')

        with open(self.file, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def __enter__(self):
        self.open()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __call__(self, trees: list[BaseOperation]) -> list[Solution]:
        if self._open_file is None:
            raise RuntimeError("Use `with open DiskCachedObjectvive(...) as objective:`.")

        strings = [t.string() for t in trees]
        new_pairs = [(tree, string) for tree, string in zip(trees, strings) if string not in self.values]
        new_sols = self.objective([tree for tree, string in new_pairs])

        lines = []
        for (_, string), sol in zip(new_pairs, new_sols):
            self.values[string] = sol.fitness
            self.datas[string] = sol.data # we can't store data in text so its just not stored
            lines.append(f'{string}: {sol.fitness}\n')

        self._open_file.writelines(lines)
        return [Solution(tree, self.values[string], self.datas.get(string, None)) for tree, string in zip(trees, strings)]
