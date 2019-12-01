from santaspkg.cost_function import reference_cost_function
from santaspkg.dataset import sample_submission

from pathlib import Path


submission_template = sample_submission.copy(deep=True)


submissions_dir = Path(__file__).parent.parent / 'submissions'
submissions_dir.mkdir(exist_ok=True)


def mk_submit(assignment):
    score = reference_cost_function(assignment)
    filename = str(submissions_dir / f'submission_{score}.csv')
    submission_template['assigned_day'] = assignment
    submission_template.to_csv(filename)
