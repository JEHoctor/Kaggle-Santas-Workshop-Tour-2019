# This module is for determining which cost function implementation is the fastest. Correctness
# of each implementation is checked in a test_*.py file.
# run with `python -m santaspkg.cost_function.benchmark`

import timeit
import numpy as np
import pandas as pd
from santaspkg import dataset

from santaspkg.cost_function.reference_cost_function import reference_cost_function
from santaspkg.cost_function.four_x_faster import cost_function as four_x_faster_cost_function
from santaspkg.cost_function.crescenzi_cost_function import cost_function as crescenzi_cost_function

named_functions = [
    ('reference', reference_cost_function),
    ('4x faster', four_x_faster_cost_function),
    ('crescenzi', crescenzi_cost_function),
]


def main():
    submission = pd.read_csv(
        dataset.sample_submission_file, index_col='family_id')
    prediction = submission['assigned_day'].tolist()

    for function_name, f in named_functions:
        if function_name == 'crescenzi':
            # This is a bit of a mess honestly, but necessary so that each method can
            # be measured with its preferred input data type.
            prediction = np.asarray(prediction)
            # Also the function seems to need to be run once to be JIT compiled.
            f(prediction)
        times = timeit.repeat(
            'f(prediction)',
            number=10**2,
            globals={
                'f': f,
                'prediction': prediction
            })
        print(function_name, np.mean(times), np.std(times))


if __name__ == '__main__':
    main()
