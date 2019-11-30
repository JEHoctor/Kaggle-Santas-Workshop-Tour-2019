from santaspkg.cost_function.four_x_faster import cost_function as four_x_faster_cost_function
from santaspkg.cost_function.crescenzi_cost_function import jited_cost, desired, family_size, penalties
from santaspkg.cost_function.reference_cost_function import reference_cost_function, choice_dict
from santaspkg.dataset import sample_submission
import numpy as np


# This should always return the same value as the reference implementation. If there
# are any days out of range, it will re-calculate using the 4x-faster implementation.
def cost_function_under_test(prediction):
    penalty, n_out_of_range = jited_cost(np.asarray(prediction), desired, family_size, penalties)
    if n_out_of_range > 0:
        return four_x_faster_cost_function(prediction)
    else:
        return penalty


def both_cost_functions(prediction):
    a = reference_cost_function(prediction)
    b = cost_function_under_test(prediction)
    assert a-b < 0.5 #assert a == b
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
