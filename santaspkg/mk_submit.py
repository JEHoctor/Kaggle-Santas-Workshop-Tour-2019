from santaspkg.cost_function import reference_cost_function
from santaspkg.dataset import sample_submission
import pandas as pd


submission_template = sample_submission.copy(deep=True)


def mk_submit(assignment):
    score = reference_cost_function(assignment)
    filename = f'submission_{score}.csv'
    submission_template['assigned_day'] = assignment
    submission_template.to_csv(filename)
