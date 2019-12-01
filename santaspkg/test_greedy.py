from santaspkg.greedy import random_greedy
from santaspkg.dataset import desired, family_size, penalties
from santaspkg.cost_function.crescenzi_cost_function import jited_cost

import numpy as np


def test_random_greedy():
    rs = np.random.RandomState(500)
    best_score = 10**8
    for x in range(100):
        assignment = random_greedy(rs)
        score, n_out_of_range = jited_cost(assignment, desired, family_size, penalties)
        assert n_out_of_range == 0
        best_score = min(score, best_score)
    print('Best score during testing:', best_score)
