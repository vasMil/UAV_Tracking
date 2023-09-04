from typing import List, Dict, Tuple
from itertools import product

import numpy as np
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination

from project_types import Pruned_model_stats_t

class PruningSpeedupAccuracyProblem(Problem):
    def __init__(self,
                 stats: List[Pruned_model_stats_t],
                 diff_sparsities: List[float] = [0.8, 0.85, 0.9, 0.95],
                 num_group_layers: int = 4,
                 map_key: str = "map_75"
    ):
        super().__init__(n_var=1, n_obj=2, n_constr=0, xl=[-100., -200.], xu=[0, 0], elementwise_evaluation=True)
        self.diff_sparsities = diff_sparsities
        self.num_group_layers = num_group_layers
        self.sparsity_combinations = list(product(diff_sparsities, repeat=num_group_layers))
        self.map_key = map_key

        # Manipulate the stats data structure so you can search for a specific sparsity set fast
        self.indexed_stats: Dict[Tuple[float], Pruned_model_stats_t] = {}
        for stat in stats:
            sparsities = tuple(stat["layer_sparsity"])
            if sparsities in self.indexed_stats.values():
                raise Exception(f"The setup of sparsities {sparsities} found twice in the given stats list")
            self.indexed_stats[sparsities] = stat

    def _evaluate(self, xs, out, *args, **kwargs) -> None:
        res = []
        for x in xs:
            stat = self.indexed_stats.get(tuple(x.squeeze().tolist()))
            if not stat:
                raise Exception(f"sparsity {x} was not found in indexed stats")
            if stat["sparsity"] is None or stat["theoretical_speedup"] is None:
                raise Exception(f"Missing sparsity of theoretical speedup for {x}")
            res.append((-stat["map_dict"][self.map_key], -stat["theoretical_speedup"]))
        out['F'] = np.array(res)

class SparsitySampling(Sampling):

    def _do(self,
            problem: PruningSpeedupAccuracyProblem,
            n_samples: int,
            **kwargs
        ):
        X = np.full((n_samples, 1), None, dtype=tuple)
        for i in range(n_samples):
            X[i, 0] = problem.sparsity_combinations[np.random.randint(0, len(problem.sparsity_combinations))]
        
        return X

class SparsityCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem: PruningSpeedupAccuracyProblem, X, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # Initialize the output of shape (n_offsprings, n_matings, n_var)
        # Y = np.full([self.n_offsprings, n_matings, n_var], None, dtype=object)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):

            # get the first and the second parent
            a, b = X[0, k, 0], X[1, k, 0]

            # select the crossover point
            off_a = []
            off_b = []
            cross_point = np.random.randint(0, problem.num_group_layers)
            for i in range(problem.num_group_layers):
                off_a.append(a[i] if cross_point <= i else b[i])
                off_b.append(b[i] if cross_point <= i else a[i])

            Y[0, k, 0], Y[1, k, 0] = tuple(off_a), tuple(off_b)

        return Y

class SparsityMutation(Mutation):
    def __init__(self) -> None:
        super().__init__()

    def _do(self, problem: PruningSpeedupAccuracyProblem, X, **kwargs):
        # 50% chance for each group-layer to choose a different sparsity
        for i in range(len(X)):
            sparsities = list(X[i, 0])
            for spi in range(len(sparsities)):
                if np.random.randint(0, 2):
                    diff_sparsities = problem.diff_sparsities.copy()
                    diff_sparsities.remove(sparsities[spi])
                    sparsities[spi] = np.random.choice(diff_sparsities, 1).item()
            X[i, 0] = tuple(sparsities)
        return X

class SparsityDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return a.X[0] == b.X[0]
