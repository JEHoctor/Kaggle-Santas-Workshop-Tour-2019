from pathlib import Path

data_dir = Path(__file__).parent / 'santa-workshop-tour-2019'
family_data_file = data_dir / 'family_data.csv'
sample_submission_file = data_dir / 'sample_submission.csv'

assert family_data_file.is_file() and sample_submission_file.is_file(), \
    ('The dataset must be downloaded from https://www.kaggle.com/c/santa-workshop-tour-2019/data'
     ' and extracted into '+str(data_dir))
