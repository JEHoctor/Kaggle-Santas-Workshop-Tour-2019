from santaspkg.cost_function.four_x_faster import cost_function as cost_function_under_test
from santaspkg.cost_function.reference_cost_function import reference_cost_function, choice_dict
from santaspkg import dataset
import pandas as pd


def both_cost_functions(prediction):
    a = reference_cost_function(prediction)
    b = cost_function_under_test(prediction)
    assert a == b
    return a


def test_four_x_faster_cost_function():
    submission = pd.read_csv(dataset.sample_submission_file, index_col='family_id')

    best = submission['assigned_day'].tolist()
    start_score = both_cost_functions(best)
    
    new = best.copy()
    # loop over each family
    for fam_id, _ in enumerate(best):
        # loop over each family choice
        for pick in range(10):
            day = choice_dict[f'choice_{pick}'][fam_id]
            temp = new.copy()
            temp[fam_id] = day # add in the new pick
            if both_cost_functions(temp) < start_score:
                new = temp.copy()
                start_score = both_cost_functions(new)
    
    score = both_cost_functions(new)
    assert score == 672254.0276683343
