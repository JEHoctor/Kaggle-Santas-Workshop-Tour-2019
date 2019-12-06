from santaspkg.cost_function.crescenzi import soft_cost_function as cost_function_under_test
from santaspkg.cost_function.reference import reference_cost_function, choice_dict
from santaspkg.dataset import sample_submission

import numpy as np


def both_cost_functions(prediction):
    a = reference_cost_function(prediction)
    b = cost_function_under_test(prediction)
    assert abs(a-b) < 0.0005
    return a


def test_crescenzi_cost_function():
    best = sample_submission['assigned_day'].values
    start_score = both_cost_functions(best)
    
    new = np.copy(best)
    # loop over each family
    for fam_id, _ in enumerate(best):
        # loop over each family choice
        for pick in range(10):
            day = choice_dict[f'choice_{pick}'][fam_id]
            temp = np.copy(new)
            temp[fam_id] = day # add in the new pick
            if both_cost_functions(temp) < start_score:
                new = np.copy(temp)
                start_score = both_cost_functions(new)
    
    score = both_cost_functions(new)
    assert score == 672254.0276683343
