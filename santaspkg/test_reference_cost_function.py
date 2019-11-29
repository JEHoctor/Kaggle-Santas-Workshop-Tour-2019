import pandas as pd
from santaspkg import dataset
from santaspkg.reference_cost_function import reference_cost_function, choice_dict

def test_reference_cost_function():
    submission = pd.read_csv(dataset.sample_submission_file, index_col='family_id')

    best = submission['assigned_day'].tolist()
    start_score = reference_cost_function(best)
    
    new = best.copy()
    # loop over each family
    for fam_id, _ in enumerate(best):
        # loop over each family choice
        for pick in range(10):
            day = choice_dict[f'choice_{pick}'][fam_id]
            temp = new.copy()
            temp[fam_id] = day # add in the new pick
            if reference_cost_function(temp) < start_score:
                new = temp.copy()
                start_score = reference_cost_function(new)
    
    #submission['assigned_day'] = new
    score = reference_cost_function(new)
    #submission.to_csv(f'submission_{score}.csv')
    #print(f'Score: {score}')
    assert score == 672254.0276683343
