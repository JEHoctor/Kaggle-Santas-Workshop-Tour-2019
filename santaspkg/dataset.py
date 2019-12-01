from pathlib import Path
import pandas as pd
import numpy as np


data_dir = Path(__file__).parent.parent / 'santa-workshop-tour-2019'
family_data_file = data_dir / 'family_data.csv'
sample_submission_file = data_dir / 'sample_submission.csv'


assert family_data_file.is_file() and sample_submission_file.is_file(), \
    ('The dataset must be downloaded from https://www.kaggle.com/c/santa-workshop-tour-2019/data'
     ' and extracted into '+str(data_dir))


data = pd.read_csv(family_data_file, index_col='family_id')
sample_submission = pd.read_csv(sample_submission_file, index_col='family_id')


# Inspired by Crescenzi's cost function implementation.
# https://www.kaggle.com/nickel/250x-faster-cost-function-with-numba-jit
desired = np.copy(data.values[:, :-1])
family_size = np.copy(data.n_people.values)
penalties = np.asarray([
    [
        0,
        50,
        50 + 9 * n,
        100 + 9 * n,
        200 + 9 * n,
        200 + 18 * n,
        300 + 18 * n,
        300 + 36 * n,
        400 + 36 * n,
        500 + 36 * n + 199 * n,
        500 + 36 * n + 398 * n
    ] for n in range(family_size.max() + 1)
])

desired.flags.writeable = False
family_size.flags.writeable = False
penalties.flags.writeable = False
