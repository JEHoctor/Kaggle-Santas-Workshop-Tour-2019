from santaspkg.cost_function.crescenzi import soft_cost_function as crescenzi_soft_cost_function
from santaspkg.cost_function.xhlulu import soft_cost_function as cost_function_under_test
from santaspkg.dataset import desired, sample_assignment

import numpy as np


def both_cost_functions(assignment):
    a = crescenzi_soft_cost_function(assignment)
    b = cost_function_under_test(assignment)
    assert np.abs(a - b) < 0.0005
    return a


def test_xhlulu_cost_function():
    best = np.copy(sample_assignment)
    best_score = both_cost_functions(best)
    
    # loop over each family
    for fam_id in range(len(best)):
        # loop over each family choice
        for day in desired[fam_id]:
            new = np.copy(best)
            new[fam_id] = day # add in the new pick
            new_score = both_cost_functions(new)
            if new_score < best_score:
                best = new
                best_score = new_score

    assert best_score == 672254.0276683343
