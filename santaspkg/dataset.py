from pathlib import Path
import pandas as pd

data_dir = Path(__file__).parent.parent / 'santa-workshop-tour-2019'
family_data_file = data_dir / 'family_data.csv'
sample_submission_file = data_dir / 'sample_submission.csv'

assert family_data_file.is_file() and sample_submission_file.is_file(), \
    ('The dataset must be downloaded from https://www.kaggle.com/c/santa-workshop-tour-2019/data'
     ' and extracted into '+str(data_dir))

data = pd.read_csv(family_data_file, index_col='family_id')
sample_submission = pd.read_csv(sample_submission_file, index_col='family_id')
